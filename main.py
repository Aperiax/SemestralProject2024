import GameLogic
import numpy as np
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
        "max": int(1),
        "min": int(0)
    }

    Player1 = GameLogic.Player(params, params_uniform, "A")

    print(Player1.parameters)
    Player1.get_candidate(params)
    print(Player1.parameters)
    listofalphas = []
    listofus = []
    for i in range(2000):
        tupleToUnpack = Player1.run()
        listofalphas.append(tupleToUnpack[0])
        listofus.append(tupleToUnpack[1])
    print(f"average alpha: {np.average(listofalphas)}")
    print(f"average u: {np.average(listofus)}")
