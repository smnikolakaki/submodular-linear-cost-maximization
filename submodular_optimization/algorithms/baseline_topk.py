"""
Baseline top-k algorithm
No approximation
"""
import logging
import numpy as np

class BaselineTopk(object):
    """
    Baseline implementation
    """
    def __init__(self, config, init_submodular_func_coverage, submodular_func, cost_func, E, k):
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
        self.init_submodular_func_coverage = init_submodular_func_coverage
        self.E = E
        if k == None:
            self.k = len(self.E)
        else:
            self.k = k

    def calc_marginal_gain(self, skills_covered, e):
        """
        Calculates the marginal gain for adding element e to the current solution sol
        :param sol:
        :param e:
        :return marginal_gain:
        """
        prev_val, skills_covered = self.submodular_func(skills_covered, [])
        new_val, skills_covered = self.submodular_func(skills_covered, [e])
        marginal_gain = new_val - prev_val

        return marginal_gain

    def original_greedy_criterion(self, skills_covered, e):
        """
        Calculates the contribution of element e to greedy solution
        :param sol:
        :param e:
        :return greedy_contrib:
        """
        # No weight scaling
        rho = 1
        marginal_gain = self.calc_marginal_gain(skills_covered, e)
        weighted_cost = rho * self.cost_func([e])
        greedy_contrib = marginal_gain - weighted_cost

        return greedy_contrib

    def sort_greedy_elements(self, E, skills_covered):
        """
        Finds the greedy element e to add to the current solution sol
        :param E:
        :param sol:
        :param k:
        :return e:
        """

        greedy_elements = sorted(E, key=lambda x: self.original_greedy_criterion(skills_covered, x), reverse = True)

        return greedy_elements

    def run(self):
        """
        Execute algorithm
        :param:
        :return best_sol:
        """
        # Keep track of current solution for a given value of k
        curr_sol = []
        # Keep track of the submodular value
        curr_val = 0    

        # Initialize the submodular function coverage skills
        self.skills_covered = self.init_submodular_func_coverage()
        # Sort greedy elements based on  objective
        sorted_greedy_elements = self.sort_greedy_elements(self.E, self.skills_covered)
 
        for i in range(0, self.k):
            greedy_element = sorted_greedy_elements[i]         
            # Element is added to the solution wrt the original objective
            # For a cardinality constraint do not exceed k elements
            if self.original_greedy_criterion(self.skills_covered, greedy_element) >= 0 and len(curr_sol) < self.k:
                curr_sol.append(greedy_element)
                submodular_gain, self.skills_covered = self.submodular_func(self.skills_covered, [greedy_element])
                curr_val += submodular_gain

        # Computing the original objective value for current solution
        curr_val = curr_val - self.cost_func(curr_sol)
        self.logger.info("Best solution: {}\nBest value: {}".format(curr_sol, curr_val))

        return curr_sol
