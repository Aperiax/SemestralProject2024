import pandas as pd
import os
import json
import GameLogic
import CurvesAndStats
import Game
if __name__ == "__main__":
    # currently dummy parameters
    params = {
            "avg": int(30),
            "std": int(50),
            "max": int(60),
            "min": int(0)
            }

    paramsNormalizedDecision = {
                        "avg": int(0),
                        "std": int(1),
                        "max": int(60),
                        "min": int(0)
                               }

    params_uniform = {
        "max": int(3),
        "min": int(0)
    }

    Game.play_game()
