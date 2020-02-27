"""
This class implements greedy algorithm - (1 - 1/e) approximation
"""
import logging


class SetCoverGreedy(object):
    """
    Set Cover Greedy Algorithm implementation
    """
    def __init__(self, config, submodular_func, E):
        """
        Constructor
        :param config:
        :param submodular_func:
        :param E -- a python set:
        :return:
        """
        self.config = config
        self.logger = logging.getLogger("so_logger")
        self.submodular_func = submodular_func
        self.scaling_factor = 1
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

    def greedy_criterion(self, sol, e):
        """
        Calculates the contribution of element e to greedy solution
        :param sol:
        :param e:
        :return greedy_contrib:
        """
        marginal_gain = self.calc_marginal_gain(sol, e)
        greedy_contrib = marginal_gain
        return greedy_contrib

    def find_greedy_element(self, E, sol):
        """
        Finds the greedy element e to add to the current solution sol
        :param E:
        :param sol:
        :return e:
        """
        greedy_element = max(E, key=lambda x: self.greedy_criterion(sol.copy(), x))
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
            greedy_element = self.find_greedy_element(self.E, curr_sol)
            if self.greedy_criterion(curr_sol.copy(), greedy_element) > 0:
                curr_sol.add(greedy_element)

        curr_val = self.submodular_func(curr_sol)
        self.logger.info("Best solution: {}\nBest value: {}".format(curr_sol, curr_val))

        return curr_sol
