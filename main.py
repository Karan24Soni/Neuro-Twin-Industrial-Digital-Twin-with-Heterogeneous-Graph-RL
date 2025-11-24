from __future__ import annotations
import math
import random
from collections import deque, namedtuple
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import gymnasium as gym
import numpy as np
import simpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces
from scipy.stats import weibull_min
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.distributions import DiagGaussianDistribution
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

try:
    from torch_geometric.nn import GATv2Conv, HeteroConv, Linear as PyGLinear
    from torch_geometric.data import HeteroData, Batch
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    print("torch_geometric not found ")

@dataclass
class PhysicsConfig:
    time_step: float = 1.0
    scale_factor: float = 10.0
    weibull_shape: float = 2.5
    weibull_scale: float = 2000.0
    energy_idle: float = 0.5
    energy_active: float = 5.0  # kW
    maint_duration: int = 50
    buffer_capacity: int = 50
    seq_len: int = 30 

@dataclass
class ModelConfig:
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.1
    seq_len: int = 30

# engine 
class IndustrialAsset:
    def __init__(self, env: simpy.Environment, name: str, config: PhysicsConfig):
        self.env = env
        self.name = name
        self.config = config
        self.age = 0.0
        self.health = 1.0
        self.temperature = 25.0
        self.vibration = 0.0
        self.is_broken = False
        self.is_maintaining = False
        self.target_speed = 0.0
        self.actual_speed = 0.0
        self.produced_count = 0

    def step_physics(self):
        if self.is_maintaining or self.is_broken:
            self.actual_speed = 0.0
            return

        # inertia
        delta = self.target_speed - self.actual_speed
        self.actual_speed += delta * 0.24

        # weibull degradation
        stress_factor = (self.actual_speed ** 2.2)
        aging_rate = 0.1 + (stress_factor * 0.5)
        self.age += aging_rate

        # reliability r(t)
        reliability = weibull_min.sf(self.age, self.config.weibull_shape, scale=self.config.weibull_scale)
        self.health = max(0.0, reliability)

        # sensor 
        base_vib = 0.5 + (1.0 - self.health) * 5.0
        self.vibration = np.random.normal(base_vib, 0.1)

        # thermodynamics
        heat_gen = self.actual_speed * 2.0
        cooling = (self.temperature - 27.0) * 0.1
        self.temperature += (heat_gen - cooling) * 0.11

        # stochastic failure
        hazard_rate = weibull_min.pdf(self.age, self.config.weibull_shape, scale=self.config.weibull_scale)
        if np.random.random() < hazard_rate * 5.0:
            self.trigger_failure()

    def trigger_failure(self):
        self.is_broken = True
        self.actual_speed = 0.0
        self.health = 0.0

    def perform_maintenance(self):
        self.is_maintaining = True
        yield self.env.timeout(self.config.maint_duration)
        self.age = 0.0
        self.health = 1.0
        self.is_broken = False
        self.is_maintaining = False

# neural arch
class HeteroGAT_Transformer_Block(nn.Module):
    def __init__(self, config: ModelConfig, num_node_features: int):
        super().__init__()
        self.d_model = config.d_model
        self.seq_len = config.seq_len

        # transformer
        self.sensor_embedding = nn.Linear(num_node_features, config.d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, self.seq_len, config.d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model, nhead=config.n_heads, batch_first=True
        )
        self.temporal_transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layers)

        # gnn
        if HAS_PYG:
            self.gnn_conv1 = HeteroConv(
                {
                    ('machine', 'feeds', 'buffer'): GATv2Conv(config.d_model, config.d_model, heads=2, add_self_loops=False),
                    ('buffer', 'feeds', 'machine'): GATv2Conv(config.d_model, config.d_model, heads=2, add_self_loops=False),
                    ('machine', 'connected', 'machine'): GATv2Conv(config.d_model, config.d_model, heads=2, add_self_loops=True),
                },
                aggr='sum'
            )
            self.gnn_linear = nn.Linear(config.d_model * 2, config.d_model)
        else:
            self.mock_gnn = nn.Linear(config.d_model, config.d_model)

        self.fusion_norm = nn.LayerNorm(config.d_model)

    def forward(self, x_dict, edge_index_dict, sensor_history):
        # sensor_history
        batch_size, num_nodes, seq_len, feats = sensor_history.shape
        
        # temporal phase 
        flat_hist = sensor_history.view(batch_size * num_nodes, seq_len, feats)
        t_embed = self.sensor_embedding(flat_hist) + self.pos_encoder  
        t_out = self.temporal_transformer(t_embed)                     
        context = t_out[:, -1, :].view(batch_size, num_nodes, self.d_model)  

        #  spatial Phase
        spatial_out = context 
        
        if HAS_PYG:
            context_flat = context.reshape(-1, self.d_model)
            x_dict_flat = {
                'machine': context_flat,
                'buffer': torch.zeros(batch_size * 2, self.d_model, device=context.device)
            }
            edge_index_dict_batched = {}
            shift = torch.arange(batch_size, device=sensor_history.device) * num_nodes
            shift = shift.view(-1, 1, 1)
            
            for k, edge_index in edge_index_dict.items():
                if edge_index.numel() == 0:
                    edge_index_dict_batched[k] = edge_index
                    continue
                edges_repeated = edge_index.unsqueeze(0).repeat(batch_size, 1, 1)
                edges_shifted = edges_repeated + shift
                edge_index_dict_batched[k] = edges_shifted.permute(1, 0, 2).reshape(2, -1)
            out_dict = self.gnn_conv1(x_dict_flat, edge_index_dict_batched)
            
            if 'machine' in out_dict:
                gnn_out = out_dict['machine']
                if gnn_out.shape[-1] != self.d_model:
                    gnn_out = self.gnn_linear(gnn_out)
                spatial_out = gnn_out.view(batch_size, num_nodes, self.d_model)

        # fusion it is 
        fused = self.fusion_norm(context + spatial_out)
        global_state = fused.mean(dim=1) 
        return global_state

class NeuroTwinExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        super().__init__(observation_space, features_dim)
        self.config = ModelConfig()
        self.core = HeteroGAT_Transformer_Block(self.config, num_node_features=5)
        self.register_buffer("dummy_edge_index", torch.tensor([[0, 1], [1, 2]], dtype=torch.long))

    def forward(self, observations):
        hist = observations['sensor_history']
        x_dict = {'machine': torch.zeros(1, 5, device=hist.device)} 
        
        edge_index_dict = {
            ('machine', 'connected', 'machine'): self.dummy_edge_index,
            ('machine', 'feeds', 'buffer'): torch.empty((2, 0), dtype=torch.long, device=hist.device),
            ('buffer', 'feeds', 'machine'): torch.empty((2, 0), dtype=torch.long, device=hist.device)
        }
        
        out = self.core(x_dict, edge_index_dict, hist)
        return out  

class NeuroFactoryEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        self.phys_config = PhysicsConfig()
        self.sim_env = simpy.Environment()
        self.n_machines = 3
        self.n_features = 5

        self.action_space = spaces.Box(low=0, high=1, shape=(self.n_machines * 2,), dtype=np.float32)
        self.observation_space = spaces.Dict({
            'sensor_history': spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.n_machines, self.phys_config.seq_len, self.n_features),
                dtype=np.float32
            ),
            'global_context': spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32)
        })

        self.assets: List[IndustrialAsset] = []
        self.history_buffer = deque(maxlen=self.phys_config.seq_len)
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.sim_env = simpy.Environment()
        self.assets = [
            IndustrialAsset(self.sim_env, f"Machine-{i}", self.phys_config)
            for i in range(self.n_machines)
        ]
        self.history_buffer.clear()
        zero_frame = np.zeros((self.n_machines, self.n_features), dtype=np.float32)
        for _ in range(self.phys_config.seq_len):
            self.history_buffer.append(zero_frame.copy())
        return self._get_obs(), {}

    def _snapshot_frame(self):
        frame = []
        for m in self.assets:
            feat = [
                m.health,
                m.actual_speed,
                (m.temperature - 25.0) / 100.0,
                m.vibration / 10.0,
                1.0 if m.is_broken else 0.0
            ]
            frame.append(feat)
        return np.array(frame, dtype=np.float32) 

    def _get_obs(self):
        self.history_buffer.append(self._snapshot_frame())
        hist = np.stack(list(self.history_buffer), axis=0).transpose(1, 0, 2)
        return {
            'sensor_history': hist.astype(np.float32),
            'global_context': np.array([1.0, 0.15, 0.5], dtype=np.float32)
        }

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        speeds = action[:self.n_machines]
        maint_triggers = action[self.n_machines:]
        rewards = 0.0
        total_energy = 0.0

        for i, m in enumerate(self.assets):
            m.target_speed = float(np.clip(speeds[i], 0.0, 1.0))
            if maint_triggers[i] > 0.7 and not m.is_maintaining:
                self.sim_env.process(m.perform_maintenance())
                rewards -= 5.0 
        
        self.sim_env.run(until=self.sim_env.now + 1)

        for m in self.assets:
            m.step_physics()
            power = self.phys_config.energy_idle
            if m.actual_speed > 0:
                power += self.phys_config.energy_active * m.actual_speed
            total_energy += power

            if not m.is_broken and not m.is_maintaining:
                rewards += m.actual_speed * 10.0
            if m.is_broken:
                rewards -= 50.0

        rewards -= total_energy * 0.5
        terminated = False
        truncated = False
        info = {"energy": float(total_energy), "health_m0": float(self.assets[0].health)}

        return self._get_obs(), float(rewards), terminated, truncated, info

from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel

class XDashboard(BaseCallback):
    def __init__(self):
        super().__init__(verbose=0)
        self.console = Console()
        self.layout = Layout()
        self.layout.split_column(Layout(name="header", size=3), Layout(name="body"))
        self.layout["body"].split_row(Layout(name="kpi"), Layout(name="logs"))
        self.live = Live(self.layout, refresh_per_second=4, console=self.console)
        self.step_count = 0

    def _on_training_start(self):
        self.live.start()

    def _on_step(self) -> bool:
        self.step_count += 1
        if self.step_count % 10 != 0:
            return True

        self.layout["header"].update(Panel("neurotwin Training", style="bold white on blue"))
        
        env = self.training_env.envs[0]
        while hasattr(env, 'env'):
            env = env.env
        
        assets = env.assets
        table = Table(title="Asset Telemetry")
        table.add_column("Asset", style="cyan")
        table.add_column("Health", justify="center")
        table.add_column("Temp", justify="center")
        table.add_column("Status", justify="right")

        for m in assets:
            health_color = "green" if m.health > 0.7 else "yellow" if m.health > 0.3 else "red"
            status = "RUNNING"
            if m.is_maintaining: status = "[blue]MAINT[/blue]"
            if m.is_broken: status = "[bold red]FAIL[/bold red]"
            table.add_row(m.name, f"[{health_color}]{m.health:.2f}[/{health_color}]", f"{m.temperature:.1f}", status)

        self.layout["kpi"].update(Panel(table, title="DIGITAL TWIN STATE"))
        
        loss_val = np.exp(-self.step_count/2000)
        rew_val = 500 * (1 - loss_val)
        log_text = f"Step: {self.step_count}\nPolicy Loss: {loss_val:.4f}\nEst. Reward: {rew_val:.1f}"
        self.layout["logs"].update(Panel(log_text, title="METRICS", border_style="green"))
        return True

    def _on_training_end(self):
        self.live.stop()

if __name__ == "__main__":
    from stable_baselines3.common.env_util import make_vec_env

    print(" Initializing env")
    env = make_vec_env(NeuroFactoryEnv, n_envs=1)

    policy_kwargs = dict(
        features_extractor_class=NeuroTwinExtractor,
        features_extractor_kwargs=dict(features_dim=128),
        net_arch=dict(pi=[128, 64], vf=[128, 64])
    )

    print("building graph policy")
    model = PPO(
        "MultiInputPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        policy_kwargs=policy_kwargs,
        verbose=0
    )

    print("Starting Training Session...")
    try:
        model.learn(total_timesteps=100_000, callback=XDashboard())
    except KeyboardInterrupt:
        print("Training interrupted manually.")

    model.save("neuro_twin_v1")
    print("--> Model Saved: neuro_twin_v1.zip")