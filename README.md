# Lunar Lander Agent using Genetic Algorithm

A Genetic Algorithm (GA) based Reinforcement Learning agent that learns to solve the **Gymnasium LunarLander-v3** environment by evolving neural network parameters through selection, crossover, and mutation.

## Features

- Genetic Algorithm based optimization
- Feedforward neural network policy
- Elite selection, crossover, and Gaussian mutation
- Multi-episode fitness evaluation
- Policy evaluation and manual gameplay support

## Technologies

- Python
- NumPy
- Gymnasium (Box2D)
- Pygame

## Project Structure

```
lunar-lander-genetic-algorithm/
├── src/
│   ├── train_agent.py
│   ├── evaluate_agent.py
│   ├── play_lunar_lander.py
│   └── policy.py
├── models/
│   └── best_policy_gen_5000.npy
├── README.md
├── requirements.txt
└── .gitignore
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Train the agent:

```bash
python src/train_agent.py
```

Evaluate a trained policy:

```bash
python src/evaluate_agent.py --filename models/best_policy_gen_5000.npy --policy_module policy
```

Play manually:

```bash
python src/play_lunar_lander.py
```


Nishant Garg