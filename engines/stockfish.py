from stockfish import Stockfish
from engines.chess_engine import ChessEngine

class StockfishEngine(ChessEngine):
    def __init__(self):
        self.stockfish = Stockfish("./stockfish/src/stockfish.exe")

    def next_move(self, board):
        self.stockfish.set_fen_position(board.fen())
        return self.stockfish.get_best_move()