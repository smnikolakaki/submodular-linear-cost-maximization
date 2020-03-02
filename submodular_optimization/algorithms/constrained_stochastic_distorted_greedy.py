"""
This class implements constrained stochastic distorted greedy algorithm
(1 - 1/e - epsilon) approximation in expectation
"""
import logging
import numpy as np


class ConstrainedStochasticDistortedGreedy(object):
    """
    ConstrainedStochasticDistored Greedy algorithm implementation
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
        self.epsilon = self.config['algorithms']['constrained_stochastic_distorted_greedy_config']['epsilon']
        self.k = k

    def calc_sample_size(self, k):
        """
        Calculates sample size for stochastic distorted greedy
        :param k:
        :return s:
        """
        s = np.ceil(len(self.E) / k * np.log(1 / self.epsilon))
        return int(s)

    def calc_marginal_gain(self, sol, e):
        """
        Calculates the marginal gain for adding element e to the current solution
        :param sol:
        :param e:
        :return marginal_gain:
        """
        prev_val = self.submodular_func(sol)
        sol.add(e)
        new_val = self.submodular_func(sol)
        marginal_gain = new_val - prev_val
        return marginal_gain

    def greedy_criterion(self, sol, e, k, i, gamma=1):
        """
        Calculates the contribution of element e to greedy solution
        :param sol:
        :param e:
        :param k:
        :param i:
        :param gamma:
        :return greedy_contrib:
        """
        rho = (k - gamma) / k
        marginal_gain = self.calc_marginal_gain(sol, e)
        weighted_gain = rho**(k - i - 1) * marginal_gain
        cost = self.cost_func(set({e}))
        greedy_contrib = weighted_gain - cost
        return greedy_contrib

    def find_greedy_element(self, E, sol, k, i):
        """
        Finds the greedy element e to add to the current solution sol
        :param E:
        :param sol:
        :param k:
        :param i:
        """
        greedy_element = max(E, key=lambda x: self.greedy_criterion(sol.copy(), x, k, i))
        return greedy_element

    def run(self):
        """
        Execute algorithm
        :param:
        :return best_sol:
        """
        curr_sol = set([])

        for i in range(0, self.k):
            s = self.calc_sample_size(self.k)
            B = set(np.random.choice(list(self.E), size=s))
            greedy_element = self.find_greedy_element(B, curr_sol, self.k, i)
            if self.greedy_criterion(curr_sol.copy(), greedy_element, self.k, i) > 0:
                curr_sol.add(greedy_element)

        curr_val = self.submodular_func(curr_sol) - self.cost_func(curr_sol)
        self.logger.info("Best solution: {}\nBest value: {}".format(curr_sol, curr_val))

        return curr_sol