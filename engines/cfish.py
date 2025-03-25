import chess.engine
from engines.chess_engine import ChessEngine

class CfishEngine(ChessEngine):
    def __init__(self):
        
        self.engine = chess.engine.SimpleEngine.popen_uci("./cfish/src/cfish.exe")
        self.engine.configure({"EvalFile": "./cfish/src/nn-62ef826d1a6d.nnue"})


    def next_move(self, board):
        res = self.engine.play(board, chess.engine.Limit(time=0.1))
        return res.move.uci()