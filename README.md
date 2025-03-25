# Chess-UE-Advanced-ML
Projet du groupe 4 de l'UE Advanced Machine Learning pour la compétition Kaggle (https://www.kaggle.com/competitions/fide-google-efficiency-chess-ai-challenge/overview). Février - Mars 2025

## Description 
Three different techniques were explored : minimax, deep learning with cnn, and reinforcement learning with alphazero.

The chess engines are stored in ```/engines/``` based on the ```ChessEngine``` class in ```/engines/chess_engine.py```. 
There you can find the different minimax used (```minimax_1.py``` being the simplest model and ```minimax_5.py``` the most complete but without the transposition tables because it was slower)

The CNN models like ```deep_nn_3.py``` use .pth models stored in ```/cnn/models/```.
For training the CNN, you need to put some data in ```/cnn/data```. This dataset was used to train the model https://database.nikonoel.fr/.
You do not need to download the dataset if you don't want to train it.

```/stockfish/``` and ```/cfish/``` contain the compiled engines used to compare our models to competitions submissions.

## Files
```tournament.py``` : functions to make the chess engines play together. 

```mem_usage.py``` : script to measure RAM consumption of the different models.

```useful_scripts.py``` : scripts to visualize games are there.