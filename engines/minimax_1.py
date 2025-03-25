import chess
import chess.svg
from engines.chess_engine import ChessEngine
import numpy as np

class Minimax1(ChessEngine):
    def __init__(self, max_depth=3):
        self.max_depth = max_depth

    def evaluate_board(self, board : chess.Board):
        """
        Evaluate the board using a simple material count.
        Positive scores favor White; negative scores favor Black.
        """
        piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 0
        }
        
        value = 0
        for piece_type, piece_value in piece_values.items():
            value += len(board.pieces(piece_type, chess.WHITE)) * piece_value
            value -= len(board.pieces(piece_type, chess.BLACK)) * piece_value

        if board.is_checkmate():
            if board.turn == chess.WHITE:
                value = -10000
            else:
                value = 10000
        
        elif board.is_check():
            if board.turn == chess.WHITE:
                value -= 1000
            else:
                value += 1000

        return value

    def minimax_alpha_beta(self, board, depth, alpha, beta, maximizing_player):
        """
        Minimax algorithm with alpha-beta pruning.
        
        Args:
            board (chess.Board): The current board state.
            depth (int): The depth of the search.
            alpha (float): The best value that the maximizer currently can guarantee.
            beta (float): The best value that the minimizer currently can guarantee.
            maximizing_player (bool): True if the current move is for the maximizing player.
            
        Returns:
            int: The evaluation score of the board.
        """
        if depth == 0 or board.is_game_over():
            return self.evaluate_board(board)
        
        if maximizing_player:
            max_eval = float('-inf')
            for move in board.legal_moves:
                board.push(move)
                eval = self.minimax_alpha_beta(board, depth - 1, alpha, beta, False)
                board.pop()
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    return max_eval
            return max_eval
        else:
            min_eval = float('inf')
            for move in board.legal_moves:
                board.push(move)
                eval = self.minimax_alpha_beta(board, depth - 1, alpha, beta, True)
                board.pop()
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    return min_eval
            return min_eval

    def find_best_move(self, board, depth):
        """
        Determine the best move for the current player using the minimax algorithm
        with alpha-beta pruning.
        
        Args:
            board (chess.Board): The current board state.
            depth (int): The search depth.
            
        Returns:
            chess.Move: The best move found.
        """
        if board.turn == chess.WHITE:
            best_value = float('-inf')
            for move in board.legal_moves:
                board.push(move)
                move_value = self.minimax_alpha_beta(board, depth - 1, float('-inf'), float('inf'), False)
                board.pop()
                if move_value > best_value:
                    best_value = move_value
                    best_move = move
        else:
            best_value = float('inf')
            for move in board.legal_moves:
                board.push(move)
                move_value = self.minimax_alpha_beta(board, depth - 1, float('-inf'), float('inf'), True)
                board.pop()
                if move_value < best_value:
                    best_value = move_value
                    best_move = move

        return best_move

    def next_move(self, board):
        """
        Chess bot that uses minimax with alpha-beta pruning to determine the best move
        """
        best_move = self.find_best_move(board, self.max_depth)
        return best_move.uci()