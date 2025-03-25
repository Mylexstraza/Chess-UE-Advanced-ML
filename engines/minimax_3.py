import chess
import threading
import concurrent.futures
import time
import numpy as np
from engines.chess_engine import ChessEngine

class Minimax3(ChessEngine):
    def __init__(self, max_depth=3):
        self.transposition_table = {}
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

    def reset(self):
        self.transposition_table.clear()

    def evaluate_board(self, board):
        """
        A more optimized evaluation that combines material, piece-square tables,
        and mobility.
        
        Returns a score in centipawns (positive favors White, negative favors Black).
        """
        # Structure des pions
        def pawn_structure(board, color):
            pawns = board.pieces(chess.PAWN, color)
            score = 0
            
            # Pions doublés
            for file in range(8):
                if chess.SquareSet(chess.BB_FILES[file]) & pawns:
                    count = bin(int(chess.SquareSet(chess.BB_FILES[file]) & pawns)).count('1')
                    if count > 1:
                        score -= 20 * (count - 1)
            
            # Pions isolés
            for square in pawns:
                file = chess.square_file(square)
                left = chess.BB_FILES[file - 1] if file > 0 else 0
                right = chess.BB_FILES[file + 1] if file < 7 else 0
                if not (pawns & (left | right)):
                    score -= 15
            
            return score

        # Sécurité du roi
        def king_safety(board, color):
            king_square = board.king(color)
            king_zone = chess.BB_KING_ATTACKS[king_square]
            safe_squares = bin(king_zone & ~board.occupied_co[color]).count('1')
            return safe_squares * 5
        
        
        if board.is_checkmate():
            return -100_000 if board.turn else 100_000
        if board.is_stalemate() or board.is_insufficient_material():
            return 0
        if board.is_check():
            return -1_000 if board.turn else 1_000

        score = 0
        piece_map = board.piece_map()  # Get a dictionary of {square: piece}

        # Precompute mirrored squares for black pieces
        mirrored_squares = {sq: chess.square_mirror(sq) for sq in piece_map.keys()}

        # Iterate over pieces and calculate scores
        for square, piece in piece_map.items():
            piece_type = piece.piece_type
            color_factor = 1 if piece.color == chess.WHITE else -1
            position_value = self.piece_tables[piece_type][square] if piece.color == chess.WHITE else self.piece_tables[piece_type][mirrored_squares[square]]
            score += color_factor * (self.piece_value(piece_type) + position_value)

        # Mobility bonus
        mobility_bonus = 10
        white_moves = len(list(board.legal_moves))
        board.push(chess.Move.null())  # Switch turn
        black_moves = len(list(board.legal_moves))
        board.pop()
        score += mobility_bonus * (white_moves - black_moves)

        # Pawn structure and king safety
        score += pawn_structure(board, chess.WHITE) - pawn_structure(board, chess.BLACK)
        score += king_safety(board, chess.WHITE) - king_safety(board, chess.BLACK)

        return score

    # Valeurs des pièces
    def piece_value(self, piece_type):
        values = {chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330, chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 20000}
        return values.get(piece_type, 0)

    # Mobilité
    def mobility(self, board, color):
        return sum(1 for move in board.legal_moves if board.color_at(move.from_square) == color)

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

        return sorted(board.legal_moves, key=score_move, reverse=True)
        # moves = list(board.legal_moves)
        # np.random.shuffle(moves)
        # return moves

    def negamax(self, board: chess.Board, depth, alpha, beta, color):
        """
        Negamax function with Alpha-Beta Pruning and Transposition Table.
        """
        
        # print(f"{board}\nnegamax: remaining depth {depth}\n{[move.uci() for move in board.move_stack]}\nalphabetas: {alpha}, {beta}\n")
        # print(f"negamax: depth {depth}")
        board_key = board.fen()  # Unique board position
        if board_key in self.transposition_table:
            # print(f"negamax: transposition table hit")
            return self.transposition_table[board_key]

        if depth == 0 or board.is_game_over():
            eval_score = color * self.evaluate_board(board)
            self.transposition_table[board_key] = eval_score
            return eval_score
        
        best_value = float('-inf')
        # moves = list(board.legal_moves)
        moves = self.move_ordering(board)
        # print(f"negamax: possible moves {[move.uci() for move in moves]}")
        for move in moves:  # Use sorted moves
            board.push(move)
            value = -self.negamax(board, depth - 1, -beta, -alpha, -color)
            board.pop()
            best_value = max(best_value, value)
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        
        self.transposition_table[board_key] = best_value  # Cache result
        return best_value


    def evaluate_move(self, board: chess.Board, move, depth, color):
        # print(f"evaluate_move: move {move}, depth {depth}")
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
        # moves = list(board.legal_moves)
        best_move = moves[0]
        for move in moves:
            board.push(move)
            score = -self.negamax(board, depth - 1, float('-inf'), float('inf'), 1 if board.turn == chess.WHITE else -1)
            board.pop()
            # print(f"find_best_move: best_score {best_score}, score {score}")
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
        # print(f"best_move_threaded_2: depth {depth}")
        ordered_moves = self.move_ordering(board)

        # print([move.uci() for move in legal_moves])
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(lambda move: self.evaluate_move(board.copy(), move, depth, 1 if board.turn == chess.WHITE else -1), ordered_moves))
        
        best_move = max(results, key=lambda x: x[1])[0]
        # print(f"best_move_threaded_2: {best_move}")
        # print(f"best_move_threaded_2: best move {best_move}")
        return best_move

    def best_move_iterative_1(self, board, max_depth, time_limit=3.0):
        global transposition_table
        start_time = time.time()
        best_move = None
        depth_reached = 0

        for depth in range(1, max_depth + 1):
            transposition_table = {}  # Clear transposition table for each depth
            depth_reached = depth
            if time.time() - start_time > time_limit:
                # print(f"Out of time at depth {depth}")
                break  # Stop searching if out of time
            move = self.best_move_threaded_2(board, depth)
            # print(f"best_move_iterative: depth: {depth}, best move: {best_move}, move: {move}")

            if move:
                best_move = move

        print(f"best_move_iterative: depth reached: {depth_reached}")
        return best_move if best_move else self.best_move_threaded_2(board, 1)  # Return best move found

    def best_move_iterative_2(self, board, max_depth, time_limit=3.0):
        """
        Performs Iterative Deepening Search with move ordering and time control.
        Ensures deeper searches use results from previous depths.
        """
        global transposition_table
        start_time = time.time()
        best_move = None
        depth_reached = 0
        move_order = list(board.legal_moves)  # Start with basic move order

        for depth in range(1, max_depth + 1):
            depth_reached = depth
            transposition_table = {}  # Clear transposition table for each depth
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
        move = self.best_move_threaded_2(board, self.max_depth)
        # move = find_best_move(board, 3)
        # move = best_move_iterative_1(board, max_depth=100, time_limit=10)
        return move.uci()