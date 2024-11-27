"""
A set of static method for scipy curve fittings as well
as a statistical suite for data analysis and saving the fitted
parameters in JSON format
"""
import numpy as np
import math
import json
import scipy.optimize as opt
import os
import matplotlib.pyplot as plt
import normality_tests


class Gaussian:
    """
    class containing exponential curves used in this project for fitting the
    real world data
    """
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
    to various degrees
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


class Statistics:
    """
    Class containing statistical methods used to process
    the input real-world data
    """

    def __init__(self, player_data: list, playerInitial: str) -> None:
        """
        A constructor
        """
        self.player_data_raw: list = player_data
        self.player_name: str = playerInitial
        match self.player_name:
            case "A": self.id = 0
            case "M": self.id = 1
            case "T": self.id = 2
            case "K": self.id = 3

    def calculate_data(self) -> dict:
        """
        initial data dictionary builder
        """
        parameters = {
            "avg": None,
            "std": None,
            "min": int(0),
            "max": int(60),
        }

        parameters.update({"avg": np.average(self.player_data_raw[:, self.id])})
        parameters.update({"std": np.std(self.player_data_raw[:, self.id])})

        return parameters

    def test_normality(self) -> bool:
        if normality_tests.tests.testnormal(self.player_data_raw, self.id) or\
           normality_tests.tests.testlognormal(self.player_data_raw, self.id):
            return True
        else:
            return False

    def generate_fx(self) -> list:
        """
        Turns the input data into a (np.ndarray resulting from pd.readcsv()) histogram,
        calculates bin centers and fits a select curve through them
        :returns list(to_plot) -> curve coefficients
        """

        # unpacks the player data
        data = self.player_data_raw
        # slicing according to player
        a = data[:, self.id]

        # generate histogram
        nbins = int(a.max() - a.min())
        hist, edges = np.histogram(a, bins=nbins, range=(a.min(), a.max()), density=True)

        print(f"len edges {len(edges.tolist())}")

        histNew, edgesNew = np.histogram(a, int((a.max() - a.min())), (a.min(), a.max()))
        edge_centers = []
        plt.plot(edges[1:], hist, color="blue")
        for i in range(1, len(edges.tolist())):
            edge_center = (edges[i] + edges[i - 1]) / 2
            edge_centers.append(edge_center)

        print(f"len centres {len(edge_centers)}")

        xdata = np.array(edge_centers)
        ydata = np.array(hist)

        initial_guess = [a.mean(), a.std()]

        # if the distribution isn't by any means normal, it defaults to polynomial curve for fit
        # i.e. if the skewness/kurtosis is out of whack completely on the dataset, it defaults to
        # less specialised curve

        if Statistics.test_normality(self):
            result = opt.curve_fit(Gaussian.normal_curve, xdata, ydata, p0=initial_guess)
            to_plot = result[0].tolist()
        else:
            result = opt.curve_fit(Polynomial.cubic, xdata, ydata, p0=initial_guess)
        return to_plot

    def savePlayerAsJson(self) -> None:
        """
        Serializes the currently configured player params dict into a json to be loaded by Game.py
        and used by MH algorithm
        """
        parameters = self.calculate_data()
        to_plot = self.generate_fx()
        with open(f"{os.getcwd()}/PlayerParams/{self.player_name}_parameters.json", "w") as g:

            json.dump([parameters, to_plot], g)
