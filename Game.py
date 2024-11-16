import sys
import GameLogic


# what I want to do is literally just populating a lsit with all the players based on the input
# actual game implementation
def play_game() -> None:
    print("Currently configured players: A, M, T, K")
    bot_name = input("Plase choose a player you wish to play against:")

    # set the gamemode score (we are mostly playing classic, no double-out 301 games)
    initial_score = 301
    paramsNormalizedDecision = {"avg": int(0), "std": int(1), "max": int(60), "min": int(0)}
    bot = GameLogic.Player(paramsNormalizedDecision, bot_name)
    counter = 0
    while True:
        # okay, now it actually updates the score, I need to find a way for it to "wait"
        # before each simulation run
        # set the initial state - 301 points to finish
        round_score = initial_score

        match input("Do you wish to generate (A)nother turn, or (E)nd the game?").strip().upper():
            case "A":
                # current_round = tuple([throws], bool(score, did it overshoot?))
                current_round = bot.run(100, round_score)
                round_score = current_round[1]
                # checks for win flag
                if current_round[3]:
                    print(f"the bot has kicked your ass after {counter} runs")
                    sys.exit()
                # if the bot didn't win outright, checks for possible overshoots
                match current_round[2]:
                    case True:
                        initial_score = initial_score
                        print(f"An overshoot occured, score unchanged: {initial_score}")
                    case False:
                        initial_score = round_score
                print(current_round)
            case "E":
                sys.exit()
        counter += 1
