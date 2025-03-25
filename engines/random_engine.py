import chess
import random
from engines.chess_engine import ChessEngine

class RandomEngine(ChessEngine):
    def __init__(self):
        pass

    def next_move(self, board: chess.Board):
        return random.choice(list(board.legal_moves)).uci()