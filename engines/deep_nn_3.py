from chess import Board, pgn
from cnn.engines.torch.auxiliary_func import board_to_matrix
import torch
from cnn.engines.torch.model import ChessModel1
import pickle
import numpy as np
from engines.chess_engine import ChessEngine

torch.set_warn_always(False)

class DeepNN3(ChessEngine):
    
    def __init__(self):
        # Load the mapping
        with open("./cnn/models/heavy_move_to_int_mathias", "rb") as file:
            move_to_int = pickle.load(file)
        self.int_to_move = {v: k for k, v in move_to_int.items()}

        # Check for GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f'Using device: {self.device}')

        # Load the model
        self.model = ChessModel1(num_classes=len(move_to_int))
        self.model.load_state_dict(torch.load("./cnn/models/TORCH_20_100EPOCHS_mathias.pth", map_location=self.device, weights_only=False))
        self.model.to(self.device)
        self.model.eval()  # Set the model to evaluation mode (it may be reductant)

    def prepare_input(self, board: Board):
        matrix = board_to_matrix(board)
        X_tensor = torch.tensor(matrix, dtype=torch.float32).unsqueeze(0)
        return X_tensor
    
    # Function to make predictions
    def next_move(self, board):
        X_tensor = self.prepare_input(board).to(self.device)
        
        with torch.no_grad():
            logits = self.model(X_tensor)
        
        logits = logits.squeeze(0)  # Remove batch dimension
        
        probabilities = torch.softmax(logits, dim=0).cpu().numpy()  # Convert to probabilities
        legal_moves = list(board.legal_moves)
        legal_moves_uci = [move.uci() for move in legal_moves]
        sorted_indices = np.argsort(probabilities)[::-1]
        for move_index in sorted_indices:
            move = self.int_to_move[move_index]
            if move in legal_moves_uci:
                return move
        
        return None

