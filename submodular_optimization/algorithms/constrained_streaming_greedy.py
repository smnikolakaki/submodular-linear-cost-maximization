"""
This class implements constrained streaming greedy algorithm
1/2(3 - sqrt(5)) approximation
"""
import logging
import numpy as np


class ConstrainedStreamingGreedy(object):
    """
    Constrained Streaming Greedy algorithm implementation
    """
    def __init__(self, config, submodular_func, cost_func, E, k):
        """
        Constructor
        :param config:
        :param submodular_func:
        :param cost_func:
        :param E -- a python set:
        :return:
        """
        self.config = config
        self.logger = logging.getLogger("so_logger")
        self.submodular_func = submodular_func
        self.cost_func = cost_func
        self.E = E
        self.k = k

    def calc_marginal_gain(self, sol, e):
        """
        Calculates the marginal gain for adding element e to the current solution sol
        :param sol:
        :param e:
        :return marginal_gain:
        """
        prev_val = self.submodular_func(sol)
        sol.add(e)
        new_val = self.submodular_func(sol)
        marginal_gain = new_val - prev_val
        return marginal_gain

    def greedy_criterion(self, sol, e):
        """
        Calculates the contribution of element e to greedy solution
        :param sol:
        :param e:
        :return greedy_contrib:
        """
        # Weight scaling is constant c
        c = (1/2)*(3 + math.sqrt(5))
        marginal_gain = self.calc_marginal_gain(sol, e)
        weighted_cost = c * self.cost_func(set({e}))
        greedy_contrib = marginal_gain - weighted_cost
        return greedy_contrib

    def run(self):
        """
        Execute algorithm
        :param:
        :return:
        """
        curr_sol = set([])
        tau = (1/self.k)*(1/2*(3 - math.sqrt(5)))

        # while the stream is not empty
        for i in range(0, len(self.E)):
            greedy_element = i
            if self.greedy_criterion(curr_sol.copy(), greedy_element) >= tau and len(curr_sol) < self.k:
                
                curr_sol.add(greedy_element)

        curr_val = self.submodular_func(curr_sol) - self.cost_func(curr_sol)
        self.logger.info("Best solution: {}\nBest value: {}".format(curr_sol, curr_val))

        return curr_sol
