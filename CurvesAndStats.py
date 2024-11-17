"""A set of static method for scipy curve fittings as well
as a statistical suite for data analysis and saving the fitted
parameters in JSON format
"""

# TODO: actually finish the statistics part

import numpy as np
import math
import json
import scipy.optimize as opt
import os
import matplotlib.pyplot as plt


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
# TODO: add a method to configure a new player
class Statistics:
    """
    class containing statistical methods used to process
    the input real-world data
    """

    def __init__(self, player_data: list, playerInitial: str) -> None:
        self.player_data_raw = player_data
        self.player_name = playerInitial
        match self.player_name:
            case "A": self.id = 0
            case "M": self.id = 1
            case "T": self.id = 2
            case "K": self.id = 3

    def calculate_data(self) -> dict:
        """
        initial data dict constructor
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

    def normality_test(self) -> bool:
        """
        function to return a bool by testing whether the data
        comes from normal distribution
        """
        raw_data = self.player_data_raw
        # implementovat to z příští chemometriky
        # placeholder
        return True

    def generate_fx(self) -> list:
        """
        Gets the data from .csv z throwSaver.py,
        načte to do dict a pak s tím dál pracuje
        :returns list(to_plot) -> plotting coefficients
        """

        # unpacks the player data
        data = self.player_data_raw

        a = data[:, self.id]

        # the way I'm currently generating hte histogram is all wrong
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

        # debug plots
        # popt, pcov = opt.curve_fit(Polynomial.fifth_degree, xdata, ydata, p0=[1, 1, 1, 1, 1, 1])
        # plt.plot(xdata, Polynomial.fifth_degree(xdata, popt[0], popt[1], popt[2], popt[3], popt[4], popt[5]), label="quintic")
        # print(np.linalg.cond(pcov))
        # popt, pcov = opt.curve_fit(Polynomial.cubic, xdata, ydata, p0=[1,1,1,1])
        # plt.plot(xdata, Polynomial.cubic(xdata, popt[0], popt[1], popt[2], popt[3]),label="cubic")
        # print(np.linalg.cond(pcov))
        # popt, pcov = opt.curve_fit(Polynomial.quadratic, xdata, ydata, p0=[1, 1, 1])
        # plt.plot(xdata, Polynomial.quadratic(xdata, popt[0], popt[1], popt[2]), color="orange", label="quadratic")
        # print(np.linalg.cond(pcov))
        # popt, pcov = opt.curve_fit(Gaussian.normal_curve, xdata, ydata, p0=initial_guess)
        # plt.plot(xdata, Gaussian.normal_curve(xdata, popt[0], popt[1]), color="red", label="gaussian")
        # print(np.linalg.cond(pcov))
        # plt.legend()
        # plt.savefig(f"{os.getcwd()}/fitting_example.png")

        if self.normality_test():
            result = opt.curve_fit(
                Gaussian.normal_curve, xdata, ydata, p0=initial_guess)
            to_plot = result[0].tolist()
        else:
            result = opt.curve_fit(
                Polynomial.quadratic, xdata, ydata, p0=initial_guess)
            to_plot = result[0].tolist()
        return to_plot

    def savePlayerAsJson(self) -> None:
        # this is a placeholder, the json in and of itself will probably have
        # to have more fields
        parameters = self.calculate_data()
        to_plot = self.generate_fx()
        with open(f"{os.getcwd()}/PlayerParams/{self.player_name}_parameters.json", "w") as g:

            json.dump([parameters, to_plot], g)
