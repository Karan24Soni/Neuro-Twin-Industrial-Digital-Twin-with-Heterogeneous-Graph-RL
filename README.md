# Neuro-Twin: Industrial Digital Twin with Heterogeneous Graph RL

A production-grade reinforcement learning system for predictive maintenance in industrial environments. Combines physics-based simulation, heterogeneous graph neural networks, and transformer architectures to optimize manufacturing operations.

## Overview

This project implements an AI agent that learns to manage a simulated factory floor with multiple machines. The agent monitors sensor data in real-time, predicts equipment failures using Weibull reliability models, and decides when to perform maintenance to minimize downtime while maximizing production output.

**Key Innovation:** Uses heterogeneous graph attention networks to model machine-to-machine dependencies and temporal transformers to capture time-series patterns from sensor data - essentially creating a "brain" for the factory that understands both spatial relationships and temporal dynamics.

## Technical Architecture

### Physics Engine
- **Weibull Degradation Model**: Realistic equipment aging based on stress factors (speed, temperature)
- **Thermodynamic Simulation**: Heat generation and cooling dynamics
- **Stochastic Failure**: Probabilistic breakdowns based on hazard rates
- **SimPy Integration**: Discrete-event simulation for maintenance scheduling

### Neural Architecture
1. **Temporal Encoder**: Transformer with positional encoding processes 30-step sensor history
2. **Spatial Encoder**: Heterogeneous GNN (GATv2Conv) models machine connectivity and dependencies
3. **Feature Fusion**: Combines temporal context with spatial relationships
4. **Policy Network**: PPO-based RL agent optimizes maintenance scheduling and production speed

### State Space (5 sensors per machine)
- Health (Weibull reliability score)
- Operating speed (normalized)
- Temperature (normalized)
- Vibration levels
- Failure status

### Action Space
- Speed control for each machine [0, 1]
- Maintenance trigger for each machine [0, 1]

## Results

Trained for 100,000 timesteps using Proximal Policy Optimization (PPO):
- Successfully learned to maintain machine health above 0.95 on average
- Reduced catastrophic failures through predictive maintenance
- Balanced production throughput against maintenance costs

## Business Impact

This approach addresses real industrial challenges:
- **Predictive Maintenance**: Reduces unplanned downtime by 40-60% compared to reactive strategies
- **Energy Optimization**: Balances production speed with power consumption
- **Production Continuity**: Maintains stable output by preventing cascade failures

Perfect fit for Industry 4.0 applications where edge AI needs to coordinate complex manufacturing systems in real-time.

## Installation

```bash
# Create virtual environment
python3 -m venv verl-env
source verl-env/bin/activate  # On Windows: verl-env\Scripts\activate

# Install dependencies
pip install gymnasium numpy scipy simpy torch stable-baselines3 rich
pip install torch-geometric  # Optional, for full GNN support
```

## Usage

```bash
# Train the model
python3 main.py

# The dashboard will show real-time telemetry:
# - Machine health scores
# - Temperature readings
# - Operational status (RUNNING/MAINT/FAIL)
```

Training takes approximately 15-25 minutes on CPU, 5-10 minutes with GPU.

## Model Architecture Details

**Input Processing:**
- Raw sensor readings → Linear embedding (5D → 128D)
- Positional encoding for temporal context
- Multi-head attention across time steps

**Graph Processing:**
- Node types: machines, buffers
- Edge types: feeds, connected
- Multi-relational message passing with attention

**Output:**
- Continuous actions for speed control
- Binary decisions for maintenance triggers

## Files

- `main.py`: Complete implementation (neural nets, environment, training loop)
- `neuro_twin_v1.zip`: Trained model weights (100k timesteps)

## Technical Notes

- Uses stable-baselines3 for PPO implementation
- PyTorch Geometric for heterogeneous graph operations
- Falls back to linear layers if PyG not available
- Rich library provides DeepMind-style live training dashboard

## Future Work

- Add maintenance cooldown periods to prevent over-maintenance
- Implement multi-agent coordination for larger factory networks
- Integrate with real PLC/SCADA systems via OPC-UA
- Add cost-based reward shaping for better economic optimization

## Requirements

```
Python 3.8+
gymnasium
numpy
scipy
simpy
torch
stable-baselines3
rich
torch-geometric (optional)
```

