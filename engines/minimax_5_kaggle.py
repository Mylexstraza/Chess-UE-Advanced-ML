
from engines.minimax_5 import Minimax5 as ChessEngine
import chess

chess_engine = ChessEngine(2)

def chess_bot(obs):
    board = chess.Board(obs.board)
    return chess_engine.next_move(board)
