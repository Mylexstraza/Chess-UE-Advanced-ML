'''
Author: Eric P. Nichols
Date: Feb 8, 2008.
Board class.
Board data:
  1=white, -1=black, 0=empty
  first dim is column , 2nd is row:
     pieces[1][7] is the square in column 2,
     at the opposite end of the board in row 8.
Squares are stored and manipulated as (x,y) tuples.
x is the column, y is the row.
'''
import copy
import numpy as np
class Board():

    # list of all 8 directions on the board, as (x,y) offsets
    __directions = [(1,1),(1,0),(1,-1),(0,-1),(-1,-1),(-1,0),(-1,1),(0,1)]

    def __init__(self, n):
        "Set up initial board configuration."

        self.antiboucle = False

        self.n = n

        self.pieces = [["-" for _ in range(self.n)] for _ in range(self.n)]  # Échiquier vide

        self.pieces[0] = ["R", "N", "B", "Q", "K", "B", "N", "R"]
        self.pieces[1] = ["P"] * self.n
        self.pieces[6] = ["p"] * self.n
        self.pieces[7] = ["r", "n", "b", "q", "k", "b", "n", "r"]

        self.has_moved = {(0,0): False, (0,4): False, (0,7): False, (7,0): False, (7,4): False, (7,7): False}

    # add [][] indexer syntax to the Board
    def __getitem__(self, index): 
        return self.pieces[index]

    def get_legal_moves(self, color):
        """Retourne tous les coups légaux possibles pour un joueur donné."""

        moves = []  # stores the legal moves.

        # Get all the squares with pieces of the given color.
        for y in range(self.n):
            for x in range(self.n):
                if (self.pieces[x][y].islower() and color==1) or (self.pieces[x][y].isupper() and color==-1):
                    newmoves = self.get_moves((x,y), self.pieces)
                    copymoves = copy.copy(newmoves)
                    for move in copymoves:
                        p = self.test_move((x,y), move[1], copy.deepcopy(self.pieces))
                        if self.isCheck(color, p):
                            newmoves.remove(move)
                    moves.extend(newmoves)
                    #print(moves)
                    
        return list(moves)

    def has_legal_moves(self, color):

        for y in range(self.n):
            for x in range(self.n):
                if (self.pieces[x][y].islower() and color==1) or (self.pieces[x][y].isupper() and color==-1):
                    newmoves = self.get_moves((x,y), self.pieces)
                    copymoves = copy.copy(newmoves)
                    for move in copymoves:
                        p = self.test_move((x,y), move[1], copy.deepcopy(self.pieces))
                        if self.isCheck(color, p):
                            newmoves.remove(move)
                    if len(newmoves)>0:
                        return True
        return False

    def get_moves(self, square, pieces, check_check=False):
        """Retourne tous les coups légaux possibles pour une pièce à la position donnée."""
        #print("get_moves", self.antiboucle)

        x, y = square
        piece = pieces[x][y]  # Récupère la pièce sur cette case

        if piece == "-":  # Si la case est vide, pas de coups possibles
            return []

        directions = []  # Stocke les directions possibles pour la pièce

        is_white = piece.islower()  # True si la pièce est blanche
        piece_type = piece.lower()  # Convertit en minuscule pour simplifier la comparaison

        # Définition des mouvements possibles en fonction du type de pièce
        if piece_type == "p":  # Pions
            return self._get_pawn_moves(x, y, piece, pieces, is_white, check_check=check_check)

        elif piece_type == "r":  # Tour : se déplace en lignes droites (orthogonalement)
            directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
            return self._add_sliding_moves(x, y, directions, is_white, pieces)

        elif piece_type == "b":  # Fou : se déplace en diagonale
            directions = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
            return self._add_sliding_moves(x, y, directions, is_white, pieces)

        elif piece_type == "q":  # Dame : combine Tour + Fou
            directions = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
            a = self._add_sliding_moves(x, y, directions, is_white, pieces)
            directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
            b = self._add_sliding_moves(x, y, directions, is_white, pieces)
            return a+b

        elif piece_type == "k":  # Roi : une case dans n'importe quelle direction
            directions = self.__directions
            return self._add_king_moves(x, y, directions, is_white, pieces)

        elif piece_type == "n":  # Cavalier : mouvement en "L"
            directions = [(2, 1), (2, -1), (-2, 1), (-2, -1), (1, 2), (1, -2), (-1, 2), (-1, -2)]
            return self._add_knight_moves(x, y, directions, is_white, pieces)

    def test_move(self, init, move, pieces):
        """Execute a move from init to move."""
        piece = pieces[init[0]][init[1]]
        
        """print("piece", piece)
        print("move", init, move)
        print(np.array(pieces))
        print(self.isCheck(1, pieces))
        print(self.has_moved)"""

        #pieces = pieces.tolist()
        pieces[move[0]][move[1]] = piece
        pieces[init[0]][init[1]] = "-"

        if abs(move[1]-init[1]) == 2 and piece.lower() == "k":
            x, y = init
            if piece == "k":
                if move == (0, y-2):
                    pieces[x][y-1] = "r"
                    pieces[x][0] = "-"
                else:
                    pieces[x][y+1] = "r"
                    pieces[x][7] = "-"
            else:
                if move == (0, y-2):
                    pieces[x][y-1] = "R"
                    pieces[x][0] = "-"
                else:
                    pieces[x][y+1] = "R"
                    pieces[x][7] = "-"
        
        if pieces[move[0]][move[1]].lower() == "p" and (move[0] == 0 or move[0] == 7):
            pieces[move[0]][move[1]] = "q" if pieces[move[0]][move[1]].islower() else "Q"

        return pieces

    def execute_move(self, action):
        """Execute a move from init to move."""
        underpromotion = None
        #print(move)
        init, move, underpromotion = self.from_action_to_move(action)
        #print(np.array(self.pieces))
        #print(init, move)
        piece = self.pieces[init[0]][init[1]]
        #print("piece", piece)
        #if piece == "-" or piece.isupper() or 0 > move[0] or move[0] > 7 or 0 > move[1] or move[1] > 7:
            #print(np.array(self.pieces))
            #print(init, move)
            #print("piece", piece)
            #raise ValueError("Mouvement non valide")
        
        self.pieces[move[0]][move[1]] = piece
        self.pieces[init[0]][init[1]] = "-"

        if not underpromotion:
            if init in self.has_moved.keys():
                self.has_moved[init] = True

            if abs(move[1]-init[1]) == 2 and piece.lower() == "k":
                x, y = init
                if piece == "k":
                    if move == (0, y-2):
                        self.pieces[x][y-1] = "r"
                        self.pieces[x][0] = "-"
                        self.has_moved[(x, 0)] = True
                    else:
                        self.pieces[x][y+1] = "r"
                        self.pieces[x][7] = "-"
                        self.has_moved[(x, 7)] = True
                else:
                    if move == (0, y-2):
                        self.pieces[x][y-1] = "R"
                        self.pieces[x][0] = "-"
                        self.has_moved[(x, 0)] = True
                    else:
                        self.pieces[x][y+1] = "R"
                        self.pieces[x][7] = "-"
                        self.has_moved[(x, 7)] = True
            
            if self.pieces[move[0]][move[1]].lower() == "p" and (move[0] == 0 or move[0] == 7):
                self.pieces[move[0]][move[1]] = "q" if self.pieces[move[0]][move[1]].islower() else "Q"
        else:
            if self.pieces[move[0]][move[1]].lower() == "p" and (move[0] == 0 or move[0] == 7):
                self.pieces[move[0]][move[1]] = underpromotion.lower() if self.pieces[move[0]][move[1]].islower() else underpromotion.upper()

    def from_action_to_move(self, action):
        action = ((action//73), action%73)
        init = ((action[0]//self.n), action[0]%self.n)

        index = action[1]

        if 0 <= index < 56:
            queen_moves = []
            for dx, dy in self.__directions :
                for i in range(1, 8):
                    queen_moves.append((dx * i, dy * i))
            return init, (init[0] + queen_moves[index][0], init[1] + queen_moves[index][1]), None
        elif 56 <= index < 64:
            knight_moves = [(2, 1), (2, -1), (-2, 1), (-2, -1),
                        (1, 2), (1, -2), (-1, 2), (-1, -2)]
            return init, (init[0] + knight_moves[index-56][0], init[1] + knight_moves[index-56][1]), None
        else:
            if init[0] == 1: 
                pawn_moves = [(-1, -1), (-1, 0), (-1, 1)]
            else:
                pawn_moves = [(1, -1), (1, 0), (1, 1)]
            
            # knight, bishop, rook
            if 64 <= index < 67:
                return init, (init[0] + pawn_moves[index-64][0], init[1] + pawn_moves[index-64][1]), "n"
            elif 67 <= index < 70:
                return init, (init[0] + pawn_moves[index-67][0], init[1] + pawn_moves[index-67][1]), "b"
            else:
                return init, (init[0] + pawn_moves[index-70][0], init[1] + pawn_moves[index-70][1]), "r"
    
    def from_move_to_action(self, init, move):

        # Convertir init en index de case
        init_index = init[0] * self.n + init[1]
        
        # Calculer le déplacement
        dx, dy = move[0] - init[0], move[1] - init[1]

        # Rechercher l'index du déplacement dans les coups possibles
        if init[0] == 1: 
            pawn_moves = [(-1, -1), (-1, 0), (-1, 1)]
        else:
            pawn_moves = [(1, -1), (1, 0), (1, 1)]

        piece = self.pieces[init[0]][init[1]]
        if piece.lower() == "p" and (init[0] == 1 or init[0] == 6) and (dx, dy) in pawn_moves:
            base_index = 64
            move_index1 = base_index + pawn_moves.index((dx, dy))
            base_index = 67
            move_index2 = base_index + pawn_moves.index((dx, dy))
            base_index = 70
            move_index3 = base_index + pawn_moves.index((dx, dy))
            return [init_index * 73 + move_index1, init_index * 73 + move_index2, init_index * 73 + move_index3]
        
        # Déplacements de la reine (56 premiers indices)
        queen_moves = []
        for direction in self.__directions:
            for i in range(1, 8):
                queen_moves.append((direction[0] * i, direction[1] * i))

        if (dx, dy) in queen_moves:
            move_index = queen_moves.index((dx, dy))
            return init_index * 73 + move_index

        # Déplacements du cavalier (indices 56 à 63)
        knight_moves = [(2, 1), (2, -1), (-2, 1), (-2, -1),
                        (1, 2), (1, -2), (-1, 2), (-1, -2)]
        
        if (dx, dy) in knight_moves:
            move_index = 56 + knight_moves.index((dx, dy))
            return init_index * 73 + move_index

        # Déplacements du pion (indices 64 à 72)
        
    def _get_pawn_moves(self, x, y, piece, pieces, is_white, check_check=False):
        """Retourne les mouvements valides pour un pion."""
        moves = []
        direction = 1 if not is_white else -1  # Blancs montent, Noirs descendent

        if not check_check:
            # Avancer d'une case
            if pieces[x + direction][y] == "-":
                moves.append(((x, y), (x + direction, y)))

                # Si le pion est sur sa rangée de départ, il peut avancer de 2 cases
                if (is_white and x == 6) or (not is_white and x == 1):
                    if pieces[x + 2 * direction][y] == "-":
                        moves.append(((x, y), (x + 2 * direction, y)))

        # Captures diagonales
        for dy in [-1, 1]:  
            if 0 <= y + dy < self.n:
                target = pieces[x + direction][y + dy]
                if target != "-" and target.islower() != is_white:
                    moves.append(((x, y), (x + direction, y + dy)))

        return moves

    def _add_sliding_moves(self, x, y, directions, is_white, pieces):
        """Ajoute les mouvements pour les pièces glissant sur plusieurs cases (Tour, Fou, Dame)."""
        moves = []
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            while 0 <= nx < self.n and 0 <= ny < self.n:
                target = pieces[nx][ny]

                if target == "-":  # Case vide → déplacement possible
                    moves.append(((x, y), (nx, ny)))
                else:
                    if target.islower() != is_white:  # Pièce adverse → capture possible
                        moves.append(((x, y), (nx, ny)))
                    break  # Bloqué par une pièce

                nx += dx
                ny += dy

        return moves

    def _add_king_moves(self, x, y, directions, is_white, pieces):
        """Ajoute les mouvements possibles pour le Roi."""
        moves = []
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.n and 0 <= ny < self.n:
                target = pieces[nx][ny]
                if target == "-" or target.islower() != is_white:  # Vide ou capture possible
                    moves.append(((x, y), (nx, ny)))
        
        if is_white:
            color = 1
        else:
            color = -1
        if not self.antiboucle:
            can_right_castle, can_left_castle = self.can_castle(color, pieces)
            if can_right_castle:
                moves.append(((x, y), (x, y + 2)))
            if can_left_castle:    
                moves.append(((x, y), (x, y - 2)))
        else:
            self.antiboucle = False
        return moves
    
    def _add_knight_moves(self, x, y, directions, is_white, pieces):
        """Ajoute les mouvements possibles pour un Cavalier."""
        moves = []

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.n and 0 <= ny < self.n:
                target = pieces[nx][ny]
                if target == "-" or target.islower() != is_white:  # Vide ou adversaire
                    moves.append(((x, y), (nx, ny)))
        return moves
    
    def is_under_attack(self, x, y, color, pieces):
        """Retourne True si la case (x, y) est attaquée par une pièce de l'autre couleur."""
        #print("is_under_attack", self.antiboucle)
        #print(np.array(self.pieces))

        for row in range(self.n):
            for col in range(self.n):
                piece = pieces[row][col]
                
                if piece == "-":
                    continue

                # Vérifie si la pièce est de la couleur adverse
                if (piece.isupper() and color == 1) or (piece.islower() and color == -1):
                    #print(piece)
                    # Obtenir les mouvements possibles de cette pièce
                    moves = self.get_moves((row, col), pieces, check_check=True)
                    for move in moves:
                        #print(move[1])
                        # Si l'adversaire peut atteindre (x, y)
                        if move[1] == (x, y):
                            return True
        return False

    def isCheck(self, color, pieces):
        # Trouver la position du roi du joueur
        #print(self.antiboucle)

        king_position = None
        self.antiboucle = True
        found = False
        for row in range(self.n):
            for col in range(self.n):
                if (color == 1 and pieces[row][col] == "k") or (color == -1 and pieces[row][col] == "K"):
                    king_position = (row, col)
                    found = True
                    break
            if found:
                break

        if king_position is None:
            import numpy as np
            print(np.array(pieces))
            raise ValueError("Le roi n'a pas été trouvé")
        
        # Vérifier si le roi est en échec
        if self.is_under_attack(king_position[0], king_position[1], color, pieces):
            return True

        return False

    def can_castle(self, color, pieces):
        """Détermine si le joueur peut effectuer un grand ou petit roque."""
        
        # Trouver les positions du roi et des tours du joueur
        if color == -1:
            i=0
        else:
            i=7

        if (i, 4) in self.has_moved.keys():    
            ind = True            
            king_position = (i, 4)
        else:
            ind = False
            king_position = (i, 3)
        rook_positions = [(i, 0), (i, 7)]

        
        # Le roi ne doit pas avoir bougé
        isckeck = self.isCheck(color, pieces)
        if self.has_moved[king_position] or isckeck:
            return False, False

        
        can_large_castle = False
        if not self.has_moved[rook_positions[0]]:
            if all(pieces[king_position[0]][i] == "-" for i in range(rook_positions[0][1] + 1, king_position[1])):
                can_large_castle = True

        can_small_castle = False
        if not self.has_moved[rook_positions[1]]:
            if all(pieces[king_position[0]][i] == "-" for i in range(king_position[1] + 1, rook_positions[1][1])):
                can_small_castle = True

        if not ind:
            return can_large_castle, can_small_castle
        return can_small_castle, can_large_castle
