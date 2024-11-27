"""
A runner module handling the game itself, so main.py remains clean
"""

import sys
import GameLogic
from GameLogic import InvalidPlayerParams

# what I want to do is literally just populating a lsit with all the players based on the input
# actual game implementation


def play_game() -> None:
    paramsNormalizedDecision = {"avg": int(0), "std": int(1), "max": int(60), "min": int(0)}

    print("Currently configured players: A, M, T, K")
    while True:
        try:
            bot_name = input("Plase choose a player you wish to play against:\n")
            bot = GameLogic.Player(paramsNormalizedDecision, bot_name)
            break
        except FileNotFoundError as err:
            print(err)
            continue
        except InvalidPlayerParams as err:
            print(err)
            continue
    # set the gamemode score (we are mostly playing classic, no double-out 301 games)
    initial_score = 301
    counter = 0
    print(f"current score is {initial_score}")

    while True:
        # main runner
        round_score = initial_score
        match input("Do you wish to generate (A)nother turn, or (E)nd the game?\n").strip().upper():
            case "A":
                # current_round = tuple([throws], bool(did it overshoot?))
                current_round = bot.run(100, round_score)
                round_score = current_round[1]
                print(f"After round {counter} the score is: {round_score}, bot threw {sum(current_round[0])}")

                # checks for win
                if current_round[3]:
                    print(f"the bot has kicked your ass after {counter} runs")
                    sys.exit()

                # if the bot didn't win outright, checks for possible overshoots
                match current_round[2]:
                    case True:
                        initial_score = initial_score
                        print(f"\033[31mAn overshoot occured, score unchanged: {initial_score}\033[0m")
                    case False:
                        initial_score = round_score
                print(current_round)
            case "E":
                sys.exit()
            case _:
                print("Please enter a valid option")
        counter += 1
        if counter > 30:
            # noone wants to play darts for more than 30 rounds, no matter how stubborn. And if I don't
            # pull the plug personally after 20, the bot will just do it after 30.
            print("\033[31mAfter extreme unluck, the bot decided to just win\033[0m")
            sys.exit()
