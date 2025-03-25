
from engines.minimax_4 import Minimax4 as ChessEngine
import chess

chess_engine = ChessEngine(1)

def chess_bot(obs):
    board = chess.Board(obs.board)
    return chess_engine.next_move(board)
