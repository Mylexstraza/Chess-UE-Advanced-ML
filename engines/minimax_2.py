import chess
import threading
import concurrent.futures
import time
import numpy as np
from engines.chess_engine import ChessEngine

class Minimax2(ChessEngine):
    def __init__(self, max_depth=3):
        self.transposition_table = {}
        self.max_depth = max_depth
        self.piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 0
        }

    def reset(self):
        self.transposition_table.clear()

    def evaluate_board(self, board: chess.Board):
        """
        Optimized static evaluation function using bitboards.
        """
        if board.is_checkmate():
            return float('-inf') if board.turn == chess.WHITE else float('inf')
        elif board.is_stalemate() or board.is_insufficient_material():
            return 0
        elif board.is_check():
            return -100 if board.turn == chess.WHITE else 100

        score = 0
        for piece_type, value in self.piece_values.items():
            white_pieces = board.pieces_mask(piece_type, chess.WHITE)
            black_pieces = board.pieces_mask(piece_type, chess.BLACK)

            score += value * white_pieces.bit_count()
            score -= value * black_pieces.bit_count()
        
        # print(f"evaluate_board: {score}")
        return score

    def move_ordering(self, board: chess.Board):
        """
        Orders moves based on Most Valuable Victim - Least Valuable Attacker (MVV-LVA).
        Captures are prioritized.
        """
        def score_move(move):
            if board.is_capture(move):
                victim = board.piece_at(move.to_square)
                attacker = board.piece_at(move.from_square)
                if victim and attacker:
                    return (self.piece_values[victim.piece_type] - self.piece_values[attacker.piece_type])
            return 0  # Non-capture moves are lower priority

        return sorted(board.legal_moves, key=score_move, reverse=True)
        # moves = list(board.legal_moves)
        # np.random.shuffle(moves)
        # return moves

    def negamax(self, board: chess.Board, depth, alpha, beta, color):
        """
        Negamax function with Alpha-Beta Pruning and Transposition Table.
        """
        
        board_key = board.fen()  # Unique board position
        if board_key in self.transposition_table:
            return self.transposition_table[board_key]

        if depth == 0 or board.is_game_over():
            eval_score = color * self.evaluate_board(board)

            self.transposition_table[board_key] = eval_score
            return eval_score
        
        best_value = float('-inf')
        moves = self.move_ordering(board)
        for move in moves:
            board.push(move)
            value = -self.negamax(board, depth - 1, -beta, -alpha, -color)
            board.pop()
            best_value = max(best_value, value)
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        
        self.transposition_table[board_key] = best_value
        return best_value


    def evaluate_move(self, board: chess.Board, move, depth, color):
        board.push(move)
        score = -self.negamax(board, depth - 1, float('-inf'), float('inf'), -color)
        board.pop()
        return move, score

    def find_best_move(self, board:chess.Board, depth):
        """
        Determines the best move using Negamax.
        """
        best_score = float('-inf')
        moves = self.move_ordering(board)
        best_move = moves[0]
        for move in moves:
            board.push(move)
            score = -self.negamax(board, depth - 1, float('-inf'), float('inf'), 1 if board.turn == chess.WHITE else -1)
            board.pop()
            if score > best_score:
                best_score = score
                best_move = move
        return best_move

    def best_move_threaded_1(self, board, depth):
        """
        Determines the best move using threading.
        """
        legal_moves = list(board.legal_moves)
        results = [None] * len(legal_moves)
        threads = []
        
        def evaluate_and_store(index, move):
            results[index] = self.evaluate_move(board.copy(), move, depth, 1 if board.turn == chess.WHITE else -1)
        
        for i, move in enumerate(legal_moves):
            thread = threading.Thread(target=evaluate_and_store, args=(i, move))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        best_move = max(results, key=lambda x: x[1])[0]
        return best_move

    def best_move_threaded_2(self, board, depth):
        """
        Determines the best move using efficient threading with ThreadPoolExecutor.
        """
        ordered_moves = self.move_ordering(board)

        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(lambda move: self.evaluate_move(board.copy(), move, depth, 1 if board.turn == chess.WHITE else -1), ordered_moves))
        
        best_move = max(results, key=lambda x: x[1])[0]
        return best_move

    def best_move_iterative_1(self, board, max_depth, time_limit=3.0):
        start_time = time.time()
        best_move = None
        depth_reached = 0

        for depth in range(1, max_depth + 1):
            self.transposition_table = {}  # Clear transposition table for each depth
            depth_reached = depth
            if time.time() - start_time > time_limit:
                break  # Stop searching if out of time
            move = self.best_move_threaded_2(board, depth)

            if move:
                best_move = move

        print(f"best_move_iterative: depth reached: {depth_reached}")
        return best_move if best_move else self.best_move_threaded_2(board, 1)  # Return best move found

    def best_move_iterative_2(self, board, max_depth, time_limit=3.0):
        """
        Performs Iterative Deepening Search with move ordering and time control.
        Ensures deeper searches use results from previous depths.
        """
        start_time = time.time()
        best_move = None
        depth_reached = 0
        move_order = list(board.legal_moves)  # Start with basic move order

        for depth in range(1, max_depth + 1):
            depth_reached = depth
            self.transposition_table = {}  # Clear transposition table for each depth
            if time.time() - start_time > time_limit:
                break  # Stop searching if time limit is reached
            
            best_value = float('-inf')
            ordered_moves = []  # To store moves for next depth

            for move in move_order:
                board.push(move)
                score = -self.negamax(board, depth - 1, float('-inf'), float('inf'), -1 if board.turn == chess.WHITE else 1)
                board.pop()

                ordered_moves.append((move, score))

                if score > best_value:
                    best_value = score
                    best_move = move  # Update the best move at this depth

            # Sort moves by best scores for the next iteration (best first)
            ordered_moves.sort(key=lambda x: x[1], reverse=True)
            move_order = [m[0] for m in ordered_moves]  # Update move order

        print(f"best_move_iterative: depth reached: {depth_reached}")
        return best_move if best_move else move_order[0]  # Ensure a move is always returned

    def next_move(self, board):
        # board = chess.Board(obs.board)
        # move = best_move_threaded_1(board, depth=MAX_DEPTH)
        move = self.best_move_threaded_2(board, depth=self.max_depth)
        # move = find_best_move(board, 3)
        # move = best_move_iterative_1(board, max_depth=100, time_limit=10)
        return move.uci()