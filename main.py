from GameLogic import *

if __name__ == "__main__":

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
        "max": int(10),
        "min": int(0)
    }

    Player1 = Player(params, paramsNormalizedDecision, "A")
    print(Player1)
    print(Player1.player_name)

    Player1.generate_fx(player_name=Player1.player_name)
    Player1.get_candidate(Player1.parameters)

