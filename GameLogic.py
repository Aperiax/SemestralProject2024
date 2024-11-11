# TODO: constructors for::
#       [] figure out biasing
#       [] treat the possibility of overshooting
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt
import copy
import CurvesAndStats

# load real world data for pre-analysis
CWD = os.getcwd()

SAVENAME = "dfThrowing.csv"
LOADPATH = f"{CWD}/{SAVENAME}"


class Distribution(object):
    """
    constructor for a distribution-type object
    """

    def __init__(self, parameters: dict) -> None:
        # error checks
        if not parameters:
            raise ValueError("Parameters for distribution are required")
        if not isinstance(parameters, dict):
            raise TypeError("Parameters have to be of type dict")
        self._params = parameters


class UniformDistribution(Distribution):
    """
    a uniform distribution entity
    """

    @property
    def max(self):
        """get the maximum"""
        maximum = self._params[NormalDistribution.MAXIMUM]
        return maximum

    @property
    def min(self):
        """get the minimum"""
        minimum = self._params[NormalDistribution.MINIMUM]
        return minimum

    def get_u(self, simulations: int) -> float:
        """
        get the uniform number for MC calculation
        """
        sample = np.random.uniform(self.max, self.min, simulations)
        return float(np.average(sample))


# just rewrite this shit and get rid of the error checks, I'm not so dumb as to
# pass missing data into my own program

class NormalDistribution(Distribution):
    AVERAGE = "avg"
    STANDARDDEVIATION = "std"
    MINIMUM = "min"
    MAXIMUM = "max"

    @property
    def avg(self) -> float:
        """get the average"""
        if self.AVERAGE not in self._params:
            raise KeyError("AVG parameter missing")
        avg = self._params[self.AVERAGE]
        return avg

    @property
    def std(self) -> float:
        """get the standard deviation"""
        if self.STANDARDDEVIATION not in self._params:
            raise KeyError("STD parameter is missing")
        std = self._params[self.STANDARDDEVIATION]
        return std

    @property
    def min(self) -> float:
        """get the min value"""
        if self.MINIMUM not in self._params:
            raise KeyError("MIN parameter is missing")
        minimum = self._params[self.MINIMUM]
        return minimum

    @property
    def max(self) -> float:
        """get the maximum value"""
        if self.MAXIMUM not in self._params:
            raise KeyError("MIN parameter is missing")
        maximum = self._params[self.MAXIMUM]
        return maximum

    def get_xt(self, simulations: int) -> float:
        """gets the initial x for simulation"""
        sample = np.random.normal(self.avg, self.std, simulations)
        return min(float(self.max), float(np.average(sample)))

    # I will just update the parameters passed into the getCandidate after each run, to have
    # it centered around the previous candidate

    def candidate_stddist(self, simulations: int) -> float:
        sample = np.random.normal(self.avg, self.std, simulations)
        return min(float(self.max), float(np.average(sample)))


class MetropolisHastings(NormalDistribution, UniformDistribution):

    """


    Metropolis-Hastings algorithm implemetnation


    """

    # this one is probably require a rewrite
    def get_candidate(self, parameters: dict, *args) -> int:
        """
        calls normal distribution class getXt method and checks it against LEGALTHROWS
        pass a normal distribution-type parameters into this
        """
        parameter = parameters
        x_prime = 0
        for arg in args:
            x_prime = arg
        # this just updates the avg part of params, this is gonna get thrown around a *lot*
        i = 0
        while True:
            x_prime = abs(int(NormalDistribution(parameter).get_xt(10)))
            i += 1
            if x_prime in Player.LEGALTHROWS:
                parameter.update({"avg": int(x_prime)})
                return x_prime
            else:
                continue

    # currently working on remaking this from static method

    # do I really need to make this method static?

    def generate_fx(self, player_name: str) -> list:
        """
        Gets the data from .csv z throwSaver.py, načte to do dict a pak s tím dál pracuje
        """

        # unpacks the player data
        data = pd.read_csv(LOADPATH).fillna(0).to_numpy()

        transformed_data = np.delete(data, 0, 1)
        upper = list(transformed_data.shape)[1]

        unpacked_data = {}

        for (initial, points) in zip(["A", "M", "T", "K"], range(upper)):
            print(transformed_data[:, points])
            unpacked_data.update(
                {initial: transformed_data[:, points].tolist()})

        a = transformed_data[:, 0]
        hist, edges = np.histogram(a, 61, (a.min(), 60), True)
        plt.plot(edges[1:], hist)
        edge_centers = []

        for i in range(1, len(edges.tolist())):
            edge_center = (edges[i] + edges[i - 1]) / 2
            edge_centers.append(edge_center)

        xdata = np.array(edge_centers)
        ydata = np.array(hist)

        initial_guess = [a.mean(), a.std()]
        result = opt.curve_fit(
            CurvesAndStats.Gaussian.normal_curve, xdata, ydata, p0=initial_guess)
        # to_plot je v podstate list tech idealnich parametru, tady to tedy bude \mu a \sigma
        to_plot = result[0].tolist()

        plt.plot(xdata, CurvesAndStats.Gaussian.normal_curve(
            xdata, to_plot[0], to_plot[1]), color="red")
        plt.savefig("retardedData.jpg")

        return to_plot

    #  remove parameters later and add back f_x_params
    def calculate_alpha(self, parameters: dict, initial_x: int, candidate: int) -> float:
        """
        a function to generate the acceptance coefficient \alpha = f(x)/f(x')
        """

        initial_state = initial_x

        # mu, sigma = f_x_params[0], f_x_params[1]
        # FOR TESTING AND BUILDING ONLY, REPLACE WITH REAL DATA LATER
        mu, sigma = parameters.get("avg"), parameters.get("std")
        # respektive proposal funkce by měla bejt gaussovka
        f_x = CurvesAndStats.Gaussian.normal_curve(initial_state, mu, sigma)
        f_x_prime = CurvesAndStats.Gaussian.normal_curve(candidate, mu, sigma)
        alpha = f_x_prime / f_x

        return alpha

    def reject_or_accept(self, alpha: float, parameters_uniform: dict) -> bool:
        """
        a logical check dependent on \alpha
        """
        u = UniformDistribution(parameters_uniform)
        decision_u = u.get_u(50)
        if alpha <= decision_u:
            return True
        else:
            return False

    def biasing(self, game_score: int, current_candidate: int):
        """
        A weighing function to make the bot "aim" more accurately the closer he gets to
        0 points
        """
        # I somehow need to add a check do decide how to manipulate the centering
        # of the actual g(x|x')
        candidate_to_be_assesed: int = current_candidate
        current_score = game_score

        pass


class Player(MetropolisHastings, Distribution):

    def __init__(self, parameters: dict, params_uniform: dict, initial: str) -> None:
        super().__init__(parameters)
        self.parameters = parameters
        self.decision_uniform = params_uniform
        self.player_name = initial
        # slap a curve fit equation here, late
        self._fx = self.generate_fx(self.player_name)
        self._normsdist = NormalDistribution(parameters)

    def __str__(self):
        return (f"Player entity {self.player_name},\n"
                f"parameters: {self.parameters},\n"
                f"decision parameters: {self.decision_uniform}")

    @staticmethod
    def make_a_lookup() -> list:
        lookup_list = list()
        for i in range(0, 21):
            lookup_list.append(i)
        lookup_list.append(25)
        list_triples = list(map(lambda x: x*3, lookup_list))
        list_doubles = list(map(lambda x: x*2, lookup_list))
        lookup = sorted(lookup_list + list_triples + list_doubles)
        lookup.pop()
        return lookup

    LEGALTHROWS = make_a_lookup()

    @staticmethod
    def is_a_valid_throw(legal_throws: list, throw_candidate: int) -> bool:
        """
        checks whether a throw is a legal one
        """
        if throw_candidate in legal_throws:
            return True
        else:
            return False

    def get_initial_state(self, number_of_simulations: int) -> int:
        """
        generate the initial x for simulating by drawing random samples and comparing them to
        a lookup table, the rest is relegated to MH class' get_candidate function, which updates the parameters
        dict so as to modulate g(x|x')
        """
        parameters = self._params
        while True:
            initial_throw_xt = abs(int(NormalDistribution(
                parameters).get_xt(number_of_simulations)))
            if Player.is_a_valid_throw(Player.LEGALTHROWS, initial_throw_xt):
                return initial_throw_xt
            else:
                continue

    def run(self, max_iterations: int, max_score: int) -> None:
        """
        Runs the monte carlo simulation to draw a throw from
        a distribution modelled after each player
        """
        initial_state = self.get_initial_state(number_of_simulations=25)
        counter = 0
        signal = True
        bin = []
        legal = copy.copy(Player.LEGALTHROWS)
        score = max_score
        while signal:
            # print(f"Entering iteration: {counter}")
            if counter != max_iterations:
                signal = True
            else:
                signal = False
            candidate = self.get_candidate(self.parameters)
            # print(f"iteration {counter} candidate: {candidate}\n")
            alpha = self.calculate_alpha(
                self.parameters, initial_state, candidate)
            # print(f"calculated alpha: {alpha}")
            reject_or_accept = self.reject_or_accept(
                alpha, self.decision_uniform)

            if reject_or_accept:
                # print(f"Candidate accepted, new intial state: {initial_state}")
                initial_state = candidate
                bin.append(candidate)
                counter += 1

            else:
                # print(f"Candidate rejected, intial state unchanged ({initial_state})")
                initial_state = initial_state
                counter += 1
        # tady to pravděpodobně budu muset zase narhadit z return value na jenom apend do listu
        # potřebuju callnout biasing a passnout mu aktuální score
        average: int = int(np.average(bin))
        if average in legal:
            print(average)
            score -= average
        else:
            legal.append(average)
            newlist = sorted(legal)
            return_val = newlist[newlist.index(average)-1]
            print(return_val)
            score -= return_val
