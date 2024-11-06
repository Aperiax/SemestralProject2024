# TODO: constructors for::
#       [] distribution
#       [] features
#       [] attributes
#       [] implement M-H, flowchart is done
#       [] figure out biasing
#       [] refactor the player and metropolis hastings classes, I want the player to
#          inherit the MH methods, so I can pass self as first argument, that way I can just
#          construct the player with distribution parameters and work more simply with the rest
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as scp
import scipy.optimize as opt
import CurvesAndStats

# load real world data for pre-analysis
CWD = os.getcwd()

SAVENAME = "dfThrowing.csv"
LOADPATH = f"{CWD}\\{SAVENAME}"


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
        return int(min(self.max, np.average(sample)))


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
        sample = np.random.normal(self.avg, self.std, simulations)
        return min(self.max, np.average(sample))

    # I will just update the parameters passed into the getCandidate after each run, to have
    # it centered around the previous candidate

    def candidate_stddist(self, simulations: int) -> float:
        sample = np.random.normal(self.avg, self.std, simulations)
        return min(self.max, np.average(sample))


class Player:

    def __init__(self, parameters: dict, params_decision: dict) -> None:
        self._fx = None  # slap a curve fit equation here, later
        self._gxyDistrib = NormalDistribution(params_decision)
        self._normsdist = NormalDistribution(parameters)

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
    def is_a_valid_throw(legal_throws, throw_candidate) -> bool:
        """
        checks whether a throw is a legal one
        """
        if throw_candidate in legal_throws:
            return True
        else:
            return False

    @staticmethod
    def get_initial_state(parameters, number_of_simulations):
        while True:
            initial_throw_xt = abs(int(NormalDistribution(parameters).get_xt(number_of_simulations)))
            if Player.is_a_valid_throw(Player.LEGALTHROWS, initial_throw_xt):
                print(f"Fetching initial state for simulation...\nstate acquired: {initial_throw_xt}")
                return initial_throw_xt
            else:
                continue


class MetropolisHastings(Player, NormalDistribution, UniformDistribution):

    @staticmethod
    def get_candidate(parameters: dict, *args) -> int:
        """
        calls normal distribution class getXt method and checks it against LEGALTHROWS
        pass a normal distribution-type parameters into this
        """
        parameter = parameters
        x_prime = 0
        for arg in args: 
            x_prime = arg
        # this just updates the avg part of params, this is gonna get thrown around a *lot*
        parameter.update({"avg": int(x_prime)})
        i = 0
        while True: 
            x_t = abs(int(NormalDistribution(parameter).get_xt(10)))
            i += 1
            if x_t in Player.LEGALTHROWS:
                print(f"Accepted candidate: {x_t}\nexiting loop after {i} iterations...")
                return x_t
            else: 
                print(f"{x_t}, candidate was rejected")
                continue

    @staticmethod
    def generate_fx(player_initial: str) -> list:
        """
        Gets the data from .csv z throwSaver.py, načte to do dict a pak s tím dál pracuje
        """

        # unpacks the player data
        player = player_initial
        data = pd.read_csv(LOADPATH).fillna(0).to_numpy()

        transformed_data = np.delete(data, 0, 1)
        upper = list(transformed_data.shape)[1]

        unpacked_data = {}

        for (initial, points) in zip(["A", "M", "T", "K"], range(upper)):
            print(transformed_data[:, points])
            print(initial)
            unpacked_data.update({initial: transformed_data[:, points].tolist()})

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

        # why does this only work with covariance stated as well?
        # parameters now hold mu and sigma parameters
        result = opt.curve_fit(CurvesAndStats.Gaussian.normal_curve, xdata, ydata, p0=initial_guess)
        to_plot = result[0].tolist()

        plt.plot(xdata, CurvesAndStats.Gaussian.normal_curve(xdata, to_plot[0], to_plot[1]), color="red")
        plt.show()

        return to_plot

    def calculate_alpha(self, initial_x: int, candidate: int) -> float:

        # gets alpha by slamming candidate and
        # initial state into the curve f(x)
        # alpha = f(x)/f(x') => i.e. potřebuju tu křivku

        pass

    def reject_or_accept(self) -> bool:
        # rejects or accepts based on $\alpha$
        pass

    def biasing(self):
        pass
