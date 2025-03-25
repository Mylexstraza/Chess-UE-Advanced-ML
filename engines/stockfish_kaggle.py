from stockfish import Stockfish

stockfish = Stockfish("./stockfish-windows-x86-64-avx2\stockfish\src\stockfish.exe")

def next_move(obs):
    board = obs.board
    stockfish.set_fen_position(obs.board)
    return stockfish.get_best_move()