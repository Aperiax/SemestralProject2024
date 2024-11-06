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

    something = Player(params, paramsNormalizedDecision)

    print(UniformDistribution(params_uniform).get_u(25))

    # okay, this works, im passing player into MH class and getting initial condition
    MetropolisHastings.get_candidate(params, 25)
    MetropolisHastings.get_initial_state(parameters=params,
                                         number_of_simulations=25)
    MetropolisHastings.generate_fx("A")
