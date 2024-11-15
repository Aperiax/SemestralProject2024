# TODO: constructors for::
#       [] add a check for maximal possible throw without overhsooting
#       [] implement updating the ubiform distribution according to biasing
#           and max throw checks
#       [] zaimplementovat nějakej check na nejoptimálnější hod
import os
import numpy as np
import pandas as pd
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
        avg = self._params[self.AVERAGE]
        return avg

    @property
    def std(self) -> float:
        """get the standard deviation"""
        std = self._params[self.STANDARDDEVIATION]
        return std

    @property
    def min(self) -> float:
        """get the min value"""
        minimum = self._params[self.MINIMUM]
        return minimum

    @property
    def max(self) -> float:
        """get the maximum value"""
        maximum = self._params[self.MAXIMUM]
        return maximum

    def get_xt(self, simulations: int) -> float:
        """gets the initial x for simulation"""
        sample = np.random.normal(self.avg, self.std, simulations)
        return min(float(self.max), float(np.average(sample)))

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

        upper = list(data.shape)[1]
        print(upper)
        unpacked_data = {}

        for (initial, points) in zip(["A", "M", "T", "K"], range(upper)):
            print(f"points {points}, unpack {unpacked_data}")
            unpacked_data.update(
                {initial: data[:, points].tolist()})
        print(unpacked_data)
        a = data[:, 0]
        hist, edges = np.histogram(a, 61, (a.min(), 60), True)
        # plt.plot(edges[1:], hist)
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

        # plt.plot(xdata, CurvesAndStats.Gaussian.normal_curve(
        #     xdata, to_plot[0], to_plot[1]), color="red")

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

    def biasing(self, current_score: int, legal: list) -> tuple:
        """
        A weighing function to make the bot "aim" more accurately the closer he
        gets to 0 points, as it stands now, it works by literally just
        calculating the factor basend on score and returning said factor
        to be used by Player.run()
        :returns: tuple (factor, optimal_throw_iteration)
        """
        factor = CurvesAndStats.Gaussian.erf_with_random_factor(current_score)
        optimal_throw_iteration = 0
        lookup = copy.copy(legal)
        match current_score:
            case current_score if current_score >= 180:
                optimal_throw_iteration = 60
            case current_score if current_score < 180:
                if current_score in lookup:
                    optimal_throw_iteration = current_score
                else:
                    lookup.append(current_score)
                    newlist = sorted(lookup)
                    optimal_throw_iteration = newlist[newlist.index(current_score)-1]
        return factor, optimal_throw_iteration


class Player(MetropolisHastings, NormalDistribution, UniformDistribution, Distribution):

    # this is probably gonna need a rewrite, i will get the parameters as distinct parts of the json
    def __init__(self, parameters: dict, params_uniform: dict, initial: str) -> None:
        super().__init__(parameters)
        self.parameters = parameters
        self.decision_uniform = params_uniform
        self.player_name = initial
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
        return list(set(lookup))
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
        generate the initial x for simulating by drawing random samples
        and comparing them to a lookup table, the rest is relegated
        to MH class' get_candidate
        function, which updates the parameters
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

    def run(self, max_iterations: int, score: int) -> tuple:
        """
        Runs the monte carlo simulation to draw a sequence of three throws
        a distribution modelled after each player, introduces extra "randomness
        factor" to counteract getting way too accurate throws
        """
        initial_state = self.get_initial_state(number_of_simulations=25)
        legal = copy.copy(Player.LEGALTHROWS)
        return_throws = []
        score_inner = score
        u = UniformDistribution({"max": 0.5, "min": 0})
        TOUGH_LUCK = u.get_u(50)
        for _ in range(3):
            counter = 0
            signal = True
            bin = []
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
                    initial_state = candidate
                    bin.append(candidate)
                    counter += 1

                else:
                    initial_state = initial_state
                    counter += 1
            # tady to pravděpodobně budu muset zase narhadit z return value na jenom apend do listu
            # potřebuju callnout biasing a passnout mu aktuální score
            average: int = int(np.average(bin) + np.random.choice((-1, 1), 1) * np.average(bin) * TOUGH_LUCK)
            if average in Player.LEGALTHROWS:
                # print(f"{average} is in legal throws")
                return_throws.append(average)
                score_inner -= average
            else:
                legal.append(average)
                newlist = sorted(legal)
                return_val = newlist[newlist.index(average)-1]
                # print(
                #     f"{average} was not in legal throws, appending next one: {return_val}")
                return_throws.append(return_val)
                score_inner -= return_val

            # update parameters:
            bias_tuple = Player.biasing(self, score_inner, Player.LEGALTHROWS)
            self.parameters.update({"avg": bias_tuple[0] * bias_tuple[1]})
            # hopefully it is biased after eeach and eevery iteration 
            # note to self: budu potřebovat updatenout main, nebo Game.py tak,
            # aby si to pamatovalo score


        return (return_throws, score_inner)
