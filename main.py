import GameLogic
if __name__ == "__main__":
    # currently dummy parameters
    params = {
            "avg": int(35),
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

    Player1.run(50)
