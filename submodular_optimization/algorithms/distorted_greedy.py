"""
This class implements distorted greedy algorithm
(1 - 1/e) approximation in expectation
"""
import logging


class DistortedGreedy(object):
    """
    Distored Greedy algorithm implementation
    """
    def __init__(self, config, submodular_func, cost_func, E, k):
        """
        Constructor
        :param config:
        :param submodular_func:
        :param cost_func:
        :param E -- a python set:
        :param k:
        :return:
        """
        self.config = config
        self.logger = logging.getLogger("so_logger")
        self.submodular_func = submodular_func
        self.cost_func = cost_func
        self.E = E

        if k == None:
            self.k = len(self.E)
        else:
            self.k = k

    def calc_marginal_gain(self, sol, e):
        """
        Calculates the marginal gain for adding element e to the current solution
        :param sol:
        :param e:
        :return marginal_gain:
        """
        prev_val = self.submodular_func(sol)
        sol.append(e)
        new_val = self.submodular_func(sol)
        marginal_gain = new_val - prev_val
        return marginal_gain

    def distorted_greedy_criterion(self, sol, e, k, i, gamma=1):
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
        cost = self.cost_func([e])
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
        greedy_element = max(E, key=lambda x: self.distorted_greedy_criterion(sol.copy(), x, k, i))
        return greedy_element

    def run(self):
        """
        Execute algorithm
        :param:
        :return best_sol:
        """
        curr_sol = []

        for i in range(0, self.k):
            # Greedy element decided wrt distorted objective
            greedy_element = self.find_greedy_element(self.E, curr_sol, self.k, i)
            # Element is added to the solution wrt distorted objective
            if self.distorted_greedy_criterion(curr_sol.copy(), greedy_element, self.k, i) > 0:
                curr_sol.append(greedy_element)

        # Computing the original objective value for current solution
        curr_val = self.submodular_func(curr_sol) - self.cost_func(curr_sol)
        self.logger.info("Best solution: {}\nBest value: {}".format(curr_sol, curr_val))

        return curr_sol
