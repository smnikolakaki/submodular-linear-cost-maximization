"""
This class implements 2 * cost scaled greedy algorithm
1/2 approximation
"""
import logging


class CostScaledGreedy(object):
    """
    2 * cost scaled greedy algorithm implementation
    """
    def __init__(self, config, submodular_func, cost_func, E):
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

    def greedy_criterion(self, sol, e, k):
        """
        Calculates the contribution of element e to greedy solution
        :param sol:
        :param e:
        :param k:
        :return greedy_contrib:
        """
        # Weight scaling is constant
        rho = 2
        marginal_gain = self.calc_marginal_gain(sol, e)
        weighted_cost = rho * self.cost_func(set({e}))
        greedy_contrib = marginal_gain - weighted_cost
        return greedy_contrib

    def find_greedy_element(self, E, sol, k):
        """
        Finds the greedy element e to add to the current solution sol
        :param E:
        :param sol:
        :param k:
        :return e:
        """
        greedy_element = max(E, key=lambda x: self.greedy_criterion(sol.copy(), x, k))
        return greedy_element

    def run(self):
        """
        Execute algorithm
        :param:
        :return:
        """
        # We set k = n
        k = len(self.E)
        curr_sol = set([])

        for i in range(0, k):
            greedy_element = self.find_greedy_element(self.E, curr_sol, k)
            if self.greedy_criterion(curr_sol.copy(), greedy_element, k) > 0:
                curr_sol.add(greedy_element)

        curr_val = self.submodular_func(curr_sol) - self.cost_func(curr_sol)
        self.logger.info("Best solution: {}\nBest value: {}".format(curr_sol, curr_val))

        return curr_sol
