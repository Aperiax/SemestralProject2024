"""A set of static method for scipy curve fittings"""

import numpy as np
import math


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
