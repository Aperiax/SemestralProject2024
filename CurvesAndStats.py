"""A set of static method for scipy curve fittings as well
as a statistical suite for data analysis and saving the fitted
parameters in JSON format
"""

# TODO: actually finish the statistics part
import numpy as np
import math
import json
import scipy.optimize as opt


class Gaussian:

    @staticmethod
    def normal_curve(x, mu, sigma):
        f_x = 1 / (np.sqrt(2 * np.pi * (sigma ** 2))) * \
            (np.exp(-((x - mu) ** 2 / (2 * sigma ** 2))))
        return f_x

    @staticmethod
    def erf_with_random_factor(x):
        erf_x = (math.sqrt(x) / 10) + np.random.choice((-1, 1), 1) * \
            np.random.choice(np.linspace(0.25, 0)) * math.erf(x / 100)
        return erf_x / 20
# in case I'd need a polynomial fit


class Polynomial:
    """
    f(x) = ax^5 + bx^4 + cx^3 + dx^2 + ex
    """
    @staticmethod
    def fifth_degree(x, a, b, c, d, e, f):

        f_x = a * x ** 5 + b * x ** 4 + c * x ** 3 + d * x ** 2 + e * x + f
        return f_x

    @staticmethod
    def fourth_degree(x, a, b, c, d, e):

        f_x = (a * x ** 4 + b * x ** 3 + c * x ** 2 + d * x + e)
        return f_x

    @staticmethod
    def cubic(x, a, b, c, d):
        f_x = a * x ** 3 + b * x ** 2 + c * x + d
        return f_x

    @staticmethod
    def quadratic(x, a, b, c):
        f_x = a * x ** 2 + b * x + c
        return f_x


# TODO: potřebuju přesunout generate_fx do tohohle modulu
class Statistics:
    """
    class containing statistical methods used to process
    the input real-world data
    """
    def __init__(self, player_data: list, playerInitial: str) -> None:
        self.player_data_raw = player_data
        self.playerName = playerInitial

    def calculate_data(self) -> dict:
        """
        initial data dict constructor
        """
        data = {
            "avg": None,
            "std": None,
            "min": int(0),
            "max": int(60),
        }
        data.update({"avg": np.avg(self.player_data_raw)})
        data.update({"std": np.std(self.player_data_raw)})

        return data
    
    def normality_test(self):
        """
        function to return a bool by testing whether the data
        comes from normal distribution
        """
        raw_data = self.player_data_raw
        # implementovat to z příští chemometriky
        pass

    def generate_fx(self):
        """
        Gets the data from .csv z throwSaver.py,
        načte to do dict a pak s tím dál pracuje
        """

        # unpacks the player data
        data = self.player_data_raw

        match self.playerInitial:
            case "A": id = 0
            case "M": id = 1
            case "T": id = 2
            case "K": id = 3

        a = data[:, id]

        hist, edges = np.histogram(a, 61, (0, 60), True)

        edge_centers = []

        for i in range(1, len(edges.tolist())):
            edge_center = (edges[i] + edges[i - 1]) / 2
            edge_centers.append(edge_center)

        xdata = np.array(edge_centers)
        ydata = np.array(hist)

        initial_guess = [a.mean(), a.std()]
        # here comes the catch -> I can't expeect the distribution to be
        # normal, i.e. I need to check for normality, then default to
        # polynomials
        result = opt.curve_fit(
            Gaussian.normal_curve, xdata, ydata, p0=initial_guess)
        to_plot = result[0].tolist()

        return to_plot

    def savePlayerAsJson(self, complete_data: dict) -> None:
        # this is a placeholder, the json in and of itself will probably have
        # to have more fields
        with open(f"{self.playerName}_parameters.json", "w") as g:
            json.dump(complete_data, g)
