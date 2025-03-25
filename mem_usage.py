# importing libraries
import os

import engines.deep_nn_3
import engines.minimax_5
import engines.random_engine
import engines.stockfish
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import psutil
from kaggle_environments import make
import numpy as np
import pandas as pd
from tournament import play, Player
import engines

# inner psutil function
def process_memory():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss

# decorator function
def profile(func):
    def wrapper(*args, **kwargs):

        mem_before = process_memory()
        result = func(*args, **kwargs)
        mem_after = process_memory()
        mb_diff = (mem_after - mem_before) / 1024 ** 2
        memory_usage.append(mb_diff)
        print("{}:consumed memory: {:,}".format(func.__name__, mb_diff))

        return result
    return wrapper

white = "random"
black = "./engines/minimax_5_kaggle.py"


memory_usage = []

# env = make("chess", debug=True)
# @profile
# def play(env, white, black, show_result=True):
#     result = env.run([white, black])

#     if show_result:
#         # look at the generated replay.json and print out the agent info
#         print("Agent exit status/reward/time left: ")
#         for agent in result[-1]:
#             print("\t", agent.status, "/", agent.reward, "/", agent.observation.remainingOverageTime)
#         print("\n")

# for _ in range(10):
#     play(env, white, black, show_result=True)


@profile
def play_game():
    black = Player("", engines.random_engine.RandomEngine(), 1200, 0, 0, 0, 0)
    white = Player("", engines.stockfish.StockfishEngine(), 1200, 0, 0, 0, 0)

    moves, result = play(white, black)
    print(f"Result: {result}")

for _ in range(10):
    play_game()

memory_usage_df = pd.DataFrame(memory_usage, columns=["memory_usage"])
print("Memory usage: \n\t{}\n\n{}".format(memory_usage, memory_usage_df.describe()))