import GameLogic
import matplotlib.pyplot as plt
import os
import numpy as np
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

    Player1 = GameLogic.Player(params, params_uniform, "A")
    # for i in range(50):
    #     print(Player1.run(50, 301))
    print(Player1.biasing(301))
    print(Player1.biasing(1))
