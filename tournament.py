import os

import engines.cfish
import engines.minimax_5
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Disable oneDNN for tensorflow
# from kaggle_environments import make, Environment
import chess
import numpy as np
import pandas as pd
import time
import engines
import engines.chess_engine
import engines.deep_nn_3
import engines.minimax_1
import engines.minimax_2
import engines.minimax_3
import engines.minimax_4
import engines.stockfish


class Player():
    '''
    Player class to keep track of player ratings and K factors
    '''
    def __init__(self, name, chess_engine: engines.chess_engine, rating, num_games, num_win, num_draw, num_loss):
        self.name = name
        self.chess_engine: engines.chess_engine = chess_engine
        self.rating = rating
        self.num_games = num_games
        self.num_win = num_win
        self.num_draw = num_draw
        self.num_loss = num_loss

    def adjust_rating(self, opponent_rating: int, score: int):
        estimated_score = 1 / (1 + 10 ** ((opponent_rating - self.rating) / 400))
        self.rating = self.rating + self.K_factor() * (score - estimated_score)
        self.rating = max(100, self.rating)
        self.adjust_num_games(score)
    
    def K_factor(self):
        if self.num_games < 30:
            return 40
        elif self.rating < 2400:
            return 20
        else:
            return 10

    def adjust_num_games(self, score):
        self.num_games += 1
        if score == 1:
            self.num_win += 1
        elif score == 0.5:
            self.num_draw += 1
        else:
            self.num_loss += 1

def play(white: Player, black: Player):
    white.chess_engine.reset()
    black.chess_engine.reset()

    iter = 0
    board = chess.Board()
    moves = []
    while not board.is_game_over() and iter < 200:
        # print(f"play: iter {iter}")
        if iter % 2 == 0:
            player = white
        else:
            player = black
        
        start = time.time()
        move = player.chess_engine.next_move(board)
        turn_duration = time.time() - start
        board.push(chess.Move.from_uci(move))
        moves.append((move, turn_duration))
        iter += 1

    result = board.result() # 1-0, 0-1, 1/2-1/2, * (* if not finished)

    # Update player ratings
    white_rating = white.rating
    black_rating = black.rating
    white_result = 0.5 if (result.split("-")[0] == "1/2" or result.split("-")[0]=="*") else int(result.split("-")[0])
    black_result = 1 - white_result
    white.adjust_rating(black_rating, white_result)
    black.adjust_rating(white_rating, black_result)

    print(f"play: iter {iter} moves {[moves[i][0] for i in range(len(moves))]}")
    return moves, result
    

def tournament(players, num_games):
    '''
    Play a tournament between players
    '''
    games = [] # List of games played
    for i in range(num_games):
        [white, black] = np.random.choice(players, 2, replace=False)
        current_time = time.strftime("%H:%M:%S", time.localtime())
        print(f"\n{current_time} / Game {i} / {white.name} vs {black.name}")
        moves, result = play(white, black)
        games.append({"white": white.name, "black": black.name, "moves": moves, "result": result, "new_ratings": [white.rating, black.rating]})
        print(f"Result: {result}")
        print_ratings(players)
    return games

def print_ratings(players):
    ratings = pd.DataFrame([[player.name, player.rating, player.num_games, player.num_win, player.num_draw, player.num_loss] for player in players], columns=["Player", "Rating", "Number of Games", "Number of Win", "Number of Draw", "Number of Lose"]).sort_values("Rating", ascending=False).reset_index(drop=True)
    ratings['Win Rate'] = ratings['Number of Win'] / ratings['Number of Games']
    print("\n", ratings, "\n")

def write_tournament(games):
    text = ""
    for i, game in enumerate(games):
        text += f"Game {i}\n"
        text += f"White vs Black: {game['white']}, {game['black']}\n"
        text += f"Result: {game['result']}\n"
        text += f"New Ratings: {game['new_ratings']}\n"

        moves, duration = zip(*game['moves'])
        text += f"Moves: {', '.join(moves)}\n"
        text += f"Duration: {', '.join([f'{d:.2f}' for d in duration])}\n\n"
    
    if not os.path.exists("tournaments"):
        os.makedirs("tournaments")
    tournament_num = len(os.listdir("tournaments"))
    with open(f"tournaments/tournament_{tournament_num}.txt", "w") as f:
        f.write(text)

def main():
    players = [
        Player("minimax_5_d2", engines.minimax_5.Minimax5(2), 1200, 0, 0, 0, 0),
        # Player("minimax_5_d3", engines.minimax_5.Minimax5(3), 1200, 0, 0, 0, 0),
        # Player("minimax_5_d5", engines.minimax_5.Minimax5(5), 1200, 0, 0, 0, 0),
        # Player("minimax_1_d2", engines.minimax_1.Minimax1(2), 1200, 0, 0, 0, 0),
        # Player("minimax_2_d3", engines.minimax_2.Minimax2(3), 1200, 0, 0, 0, 0),
        # Player("minimax_3_d4", engines.minimax_3.Minimax3(4), 1200, 0, 0, 0, 0),
        # Player("minimax_4_d4", engines.minimax_4.Minimax4(4), 1200, 0, 0, 0, 0),
        # Player("deep_nn", engines.deep_nn_3.DeepNN3(), 1200, 0, 0, 0, 0),
        Player("stockfish", engines.stockfish.StockfishEngine(), 1200, 0, 0, 0, 0),
        # Player("cfish", engines.cfish.CfishEngine(), 1200, 0, 0, 0, 0),
    ]

    num_games = 100

    games = tournament(players, num_games)
    write_tournament(games)
    print_ratings(players)


if __name__ == "__main__":
    main()