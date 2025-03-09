# Bomberman Double DQN Project

## Overview
This project implements a Double DQN for a character in a game environment, utilizing deep reinforcement learning techniques such as PER. The character makes decisions based on its surroundings and attempts to avoid dangers while trying to reach the exit.

## Requirements
To run this project, you need to have Python 3 and the following package installed:

- **PyTorch**: You can install it using pip:
```bash
pip3 install torch
```

## Running Trials
If you want to run ten trials and gather statistics, you should execute the `train_deep_q.py` script located in the `team02` folder. This script will initiate the training process and log the results.

### Command to Run Trials
Navigate to the `team02` directory and run the following command:

```bash
python3 train_deep_q.py
```