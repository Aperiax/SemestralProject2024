import pandas as pd
import os
import json
import GameLogic
import CurvesAndStats
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

    Player1 = GameLogic.Player(params_uniform, "A")
    for i in range(20):
        print(Player1.run(200, 301))
    data = pd.read_csv(GameLogic.LOADPATH).fillna(0).to_numpy()
    # analysis = [CurvesAndStats.Statistics(data, i) for i in ["A", "M", "T", "K"]]
    # # as is now, the CurvesAndStats implementation is cleaner, more general, and working
    # # analysis.generate_fx()
    # # for i in analysis:
    # #     i.savePlayerAsJson()
    # unpacked_json = None
    # with open(f"{os.getcwd()}/PlayerParams/A_parameters.json", "r") as g:
    #     unpacked_json = json.load(g)
    #
    # print(unpacked_json[0], unpacked_json[1])
    analysis = CurvesAndStats.Statistics(data, "A")
    analysis.generate_fx()
