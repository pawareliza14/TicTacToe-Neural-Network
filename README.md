
# TicTacToe Neural Network Model

## Introduction

This project implements a neural network model that predicts optimal moves in Tic-Tac-Toe games. The model is built using TensorFlow/Keras and is designed to learn optimal play strategies from synthetic data that mimics expert gameplay decisions.

Unlike traditional Tic-Tac-Toe algorithms that use minimax or other deterministic approaches, this neural network learns patterns from data, allowing it to generalize its strategy and potentially handle board states it hasn't explicitly seen before.

## Features

- **Synthetic Data Generation**: Creates realistic Tic-Tac-Toe board states with a strategic heuristic for optimal move selection
- **Optimized Neural Network Architecture**: Includes batch normalization, dropout, and regularization for better performance
- **Data Preprocessing**: Normalizes input data and uses proper train/validation/test splits
- **Early Stopping**: Prevents overfitting by monitoring validation performance
- **Move Validation**: Ensures predicted moves are only made on empty board positions
- **Board Visualization**: Includes utility functions to display the current board state

## Technical Implementation

The model architecture consists of:
- An input layer accepting 9-dimensional vectors (representing the board state)
- Two hidden layers with LeakyReLU activation and dropout for regularization
- Batch normalization layers to stabilize training
- L2 regularization to prevent overfitting
- Softmax output layer to predict probabilities for each possible move

## Summary

This project demonstrates how machine learning can be applied to simple game strategy. While Tic-Tac-Toe can be solved deterministically, this neural network approach provides a foundation for applying similar techniques to more complex games where traditional algorithms become computationally infeasible.

The model achieves good performance by learning from synthetic "expert" gameplay data and employing modern neural network optimization techniques. The provided code includes data generation, model training, evaluation, and prediction functionalities, making it a complete solution for Tic-Tac-Toe move prediction.

## Usage

```python
# Load the model
from tensorflow import keras
model = keras.models.load_model("tictactoe_best_move_optimized.h5")

# Make a prediction on an empty board
board_state = [0, 0, 0, 0, 0, 0, 0, 0, 0]
best_move = predict_move(board_state, model)
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/tictactoe-neural-network.git
cd tictactoe-neural-network

# Install dependencies
pip install tensorflow numpy scikit-learn
```

## Running the Code

```bash
# Train the model
python tictactoe_model.py

# To use in your own project
# Import the necessary functions from the script
from tictactoe_model import predict_move, print_board
```

## Future Improvements

Potential enhancements include:
- Training on real expert gameplay data
- Expanding to larger board games or more complex variants
- Implementing reinforcement learning approaches
- Adding a web/UI interface for interactive play against the model



