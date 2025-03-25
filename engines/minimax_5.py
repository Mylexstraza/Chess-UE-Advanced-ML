import chess
import concurrent.futures
import time
import numpy as np
import random
from engines.chess_engine import ChessEngine

class Minimax5(ChessEngine):
    def __init__(self, max_depth=3):
        self.max_depth = max_depth
        # Mapping piece types to their corresponding tables
        self.piece_tables = {
            chess.PAWN: np.array([
                0,  0,  0,  0,  0,  0,  0,  0,
                5, 10, 10,-20,-20, 10, 10,  5,
                5, -5,-10,  0,  0,-10, -5,  5,
                0,  0,  0, 20, 20,  0,  0,  0,
                5,  5, 10, 25, 25, 10,  5,  5,
                10, 10, 20, 30, 30, 20, 10, 10,
                50, 50, 50, 50, 50, 50, 50, 50,
                0,  0,  0,  0,  0,  0,  0,  0
            ]),
            chess.KNIGHT: np.array([
                -50,-40,-30,-30,-30,-30,-40,-50,
                -40,-20,  0,  0,  0,  0,-20,-40,
                -30,  0, 10, 15, 15, 10,  0,-30,
                -30,  5, 15, 20, 20, 15,  5,-30,
                -30,  0, 15, 20, 20, 15,  0,-30,
                -30,  5, 10, 15, 15, 10,  5,-30,
                -40,-20,  0,  5,  5,  0,-20,-40,
                -50,-40,-30,-30,-30,-30,-40,-50
            ]),
            chess.BISHOP: np.array([
                -20,-10,-10,-10,-10,-10,-10,-20,
                -10,  5,  0,  0,  0,  0,  5,-10,
                -10, 10, 10, 10, 10, 10, 10,-10,
                -10,  0, 10, 10, 10, 10,  0,-10,
                -10,  5,  5, 10, 10,  5,  5,-10,
                -10,  0,  5, 10, 10,  5,  0,-10,
                -10,  0,  0,  0,  0,  0,  0,-10,
                -20,-10,-10,-10,-10,-10,-10,-20
            ]),
            chess.ROOK: np.array([
                0,  0,  0,  5,  5,  0,  0,  0,
                -5,  0,  0,  0,  0,  0,  0, -5,
                -5,  0,  0,  0,  0,  0,  0, -5,
                -5,  0,  0,  0,  0,  0,  0, -5,
                -5,  0,  0,  0,  0,  0,  0, -5,
                -5,  0,  0,  0,  0,  0,  0, -5,
                5, 10, 10, 10, 10, 10, 10,  5,
                0,  0,  0,  0,  0,  0,  0,  0
            ]),
            chess.QUEEN: np.array([
                -20,-10,-10, -5, -5,-10,-10,-20,
                -10,  0,  0,  0,  0,  0,  0,-10,
                -10,  0,  5,  5,  5,  5,  0,-10,
                -5,  0,  5,  5,  5,  5,  0, -5,
                0,  0,  5,  5,  5,  5,  0, -5,
                -10,  5,  5,  5,  5,  5,  0,-10,
                -10,  0,  5,  0,  0,  0,  0,-10,
                -20,-10,-10, -5, -5,-10,-10,-20
            ]),
            chess.KING: np.array([
                -30,-40,-40,-50,-50,-40,-40,-30,
                -30,-40,-40,-50,-50,-40,-40,-30,
                -30,-40,-40,-50,-50,-40,-40,-30,
                -30,-40,-40,-50,-50,-40,-40,-30,
                -20,-30,-30,-40,-40,-30,-30,-20,
                -10,-20,-20,-20,-20,-20,-20,-10,
                20, 20,  0,  0,  0,  0, 20, 20,
                20, 30, 10,  0,  0, 10, 30, 20
            ])
        }

    def evaluate_board(self, board):
        """
        Evaluates the board state using piece-square tables and piece values.
        """
        score =  0
        piece_map = board.piece_map()  # Get a dictionary of {square: piece}

        # Precompute mirrored squares for black pieces
        mirrored_squares = {sq: chess.square_mirror(sq) for sq in piece_map.keys()}

        # Iterate over pieces and calculate scores
        for square, piece in piece_map.items():
            piece_type = piece.piece_type
            color_factor = 1 if piece.color == chess.WHITE else -1
            position_value = self.piece_tables[piece_type][square] if piece.color == chess.WHITE else self.piece_tables[piece_type][mirrored_squares[square]]
            score += color_factor * (self.piece_value(piece_type) + position_value)

        return score

    # Valeurs des pièces
    def piece_value(self, piece_type):
        values = {chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330, chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 0}
        return values.get(piece_type, 0)

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
                    return (self.piece_value(victim.piece_type) - self.piece_value(attacker.piece_type))
            return 0  # Non-capture moves are lower priority

        
        moves = list(board.legal_moves)
        scored_moves = [(move, score_move(move)) for move in moves]
        scored_moves.sort(key=lambda x: x[1], reverse=True)

        if not scored_moves:
            return moves
        
        # Extract moves with the best score
        best_score = scored_moves[0][1]
        best_moves = [move for move, score in scored_moves if score == best_score]
        random.shuffle(best_moves)

        # Combine best moves with the rest
        sorted_moves = best_moves + [move for move, score in scored_moves if score < best_score]
        return sorted_moves

    def negamax(self, board: chess.Board, depth, alpha, beta, color):
        """
        Negamax function with Alpha-Beta Pruning and Transposition Table.
        """

        if depth == 0 or board.is_game_over():
            eval_score = color * self.evaluate_board(board)
            return eval_score
        
        best_value = float('-inf')
        moves = self.move_ordering(board)
        for move in moves:  # Use sorted moves
            board.push(move)
            value = -self.negamax(board, depth - 1, -beta, -alpha, -color)
            board.pop()
            best_value = max(best_value, value)
            alpha = max(alpha, value)
            if alpha >= beta:
                break

        return best_value


    def evaluate_move(self, board: chess.Board, move, depth, color):
        board.push(move)
        score = -self.negamax(board, depth - 1, float('-inf'), float('inf'), -color)
        board.pop()
        return move, score

    def best_move_threaded(self, board, depth):
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
        move = self.best_move_threaded(board, depth=self.max_depth)
        # move = find_best_move(board, 3)
        # move = best_move_iterative_1(board, max_depth=100, time_limit=10)
        return move.uci()