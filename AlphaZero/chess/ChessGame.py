from __future__ import print_function
import sys
sys.path.append('..')
from Game import Game
from .ChessLogic import Board
import numpy as np
import copy
#import decode

class ChessGame(Game):
    square_content = {
        "K": "♔", "Q": "♕", "R": "♖", "B": "♗", "N": "♘", "P": "♙",
        "k": "♚", "q": "♛", "r": "♜", "b": "♝", "n": "♞", "p": "♟",
        "-": "."
    }

    @staticmethod
    def getSquarePiece(piece):
        return ChessGame.square_content[piece]

    def __init__(self, n):
        self.n = n

    def getInitBoard(self):
        # return initial board (numpy board)
        b = Board(self.n)
        return b.pieces, b.has_moved

    def getBoardSize(self):
        # (a,b) tuple
        return (self.n, self.n)

    def getActionSize(self):
        # return number of actions
        return self.n*self.n*73
    
    def verify_valid_moves(self, board, a, z, difference):
        b = Board(self.n)
        b.pieces = copy.deepcopy(board)
        print("count", [b.from_action_to_move(((a//73), a%73)) for a in a])
        print("valid", [b.from_action_to_move(((z//73), z%73)) for z in z])
        #print("diff", difference)
        #print("board", np.array(board))

    def getNextState(self, board, player, action, has_moved):
        # if player takes action on board, return next (board,player)
        # action must be a valid move
        b = Board(self.n)
        b.pieces = copy.deepcopy(board)
        b.has_moved = copy.deepcopy(has_moved)
        b.execute_move(action)
        return (b.pieces, b.has_moved, -player)

    def getValidMoves(self, board, player, has_moved):
        # return a fixed size binary vector
        valids = [0]*self.getActionSize()
        b = Board(self.n)
        b.pieces = copy.deepcopy(board)
        b.has_moved = copy.deepcopy(has_moved)
        legalMoves =  b.get_legal_moves(player)
        #print(np.array(b.pieces))
        #print(legalMoves)
        for i in range(len(legalMoves)):
            action = b.from_move_to_action(legalMoves[i][0], legalMoves[i][1])
            if type(action) == int:
                #print("dans getValidmove", (action//73, action%73))
                valids[action] = 1
            else:
                for ac in action:
                    valids[ac] = 1
        return np.array(valids)

    def only_king(self, board):
        for i in range(8):
            for j in range(8):
                if board[i][j] != '-' and board[i][j] != 'k' and board[i][j] != 'K':
                    return False
        return True
    
    def getGameEnded(self, board, player, has_moved):
        # return 0 if not ended, "tie" if tie, -1 if player 1 lost
        b = Board(self.n)
        b.pieces = copy.deepcopy(board)
        b.has_moved = copy.deepcopy(has_moved)
        is_Check = b.isCheck(player, b.pieces)
        has_legal_moves = b.has_legal_moves(player)
        only_king_on_board = self.only_king(b.pieces)
        if not has_legal_moves and is_Check:
            return -1
        elif (not has_legal_moves and not is_Check) or only_king_on_board:
            return "tie"
        else:
            return 0

    def getCanonicalForm(self, board, has_moved, player):
        # return state if player==1, else return -state if player==-1
        if player == -1:

            has_moved = {
                (k[0], 3 if k[1] == 4 else 4 if k[1] == 3 else k[1]): v for k, v in has_moved.items()
            }
            keys = list(has_moved.keys()) 
            values = list(has_moved.values()) 
            has_moved[keys[0]], has_moved[keys[1]], has_moved[keys[2]], has_moved[keys[3]], has_moved[keys[4]], has_moved[keys[5]] = values[5], values[4], values[3], values[2], values[1], values[0]
            
            board = np.rot90(np.rot90(board))
            board = [[cell.swapcase() for cell in row] for row in board]
            return board, has_moved
        return board, has_moved

    def board_to_nn_input(self, board):
        """Convertit un échiquier en une représentation 12x8x8 sous forme de tenseur numpy."""
        
        piece_to_index = {
            "p": 0, "r": 1, "n": 2, "b": 3, "q": 4, "k": 5,
            "P": 6, "R": 7, "N": 8, "B": 9, "Q": 10, "K": 11
        }
        
        nn_board = np.zeros((12, 8, 8), dtype=np.int8)

        for row in range(8):
            for col in range(8):
                piece = board[row][col]
                if piece == "-":
                    continue
                else:
                    nn_board[piece_to_index[piece], row, col] = 1
        
        return nn_board
    
    def compare_echiquiers(self, echiquier_avant, echiquier_apres):
        echiquier_avant_np = np.array(echiquier_avant)
        echiquier_apres_np = np.array(echiquier_apres)

        nb_pieces_avant = np.count_nonzero(echiquier_avant_np != '-')
        nb_pieces_apres = np.count_nonzero(echiquier_apres_np != '-')

        nombre_pieces_change = nb_pieces_avant != nb_pieces_apres

        positions_pions_avant = np.where(echiquier_avant_np == 'p')

        for i, j in zip(positions_pions_avant[0], positions_pions_avant[1]):
            if echiquier_apres_np[i, j] != 'p':
                return True

        return nombre_pieces_change

    def getSymmetries(self, board, pi):
        
        #assert board.ndim == 3 and pi.shape == (self.n * self.n * 73,)

        pi_board = np.reshape(pi, (self.n, self.n, 73))  # Reshape en 3D
        l = []

        for i in [0, 2]:  # Rotations 180°, 360°
            for j in [True, False]:  # Miroir ou non
                newB = []
                for b in board:
                    newB.append(np.rot90(b, i, axes=(0, 1)))  # Rotation du plateau (toutes les couches)
                newPi = np.rot90(pi_board, i, axes=(0, 1))  # Rotation des probabilités

                if j:
                    for i in range(len(newB)):
                        newB[i] = np.fliplr(newB[i])  # Miroir gauche-droite
                    newPi = np.fliplr(newPi)  # Miroir des probabilités

                l.append((newB, newPi.ravel()))  # Remet pi en 1D

        return l

    def stringRepresentation(self, board):
        return np.copy(board).tobytes()

    def stringRepresentationReadable(self, board):
        board_s = "".join(self.square_content[square] for row in board for square in row)
        return board_s

    @staticmethod
    def display(board):
        n = board.shape[0]
        print("   ", end="")
        for y in range(n):
            print(y, end=" ")
        print("")
        print("-----------------------")
        for y in range(n):
            print(y, "|", end="")    # print the row #
            for x in range(n):
                piece = board[y][x]    # get the piece to print
                print(ChessGame.square_content[piece], end=" ")
            print("|")

        print("-----------------------")
