"""
This class implements the baseline top k matroid algorithm
No approximation guarantee
"""
import logging
import numpy as np
import sys

class BaselineTopkMatroid(object):
    """
    Baseline top k matroid implementation for the partition matroid constraint
    """
    def __init__(self, config, init_submodular_func_coverage, submodular_func, cost_func, E, partitions):
        """
        Constructor
        :param config:
        :param submodular_func:
        :param cost_func:
        :param E -- a python set:
        :param partitions:
        :return:
        """
        self.config = config
        self.logger = logging.getLogger("so_logger")
        self.submodular_func = submodular_func
        self.cost_func = cost_func
        self.init_submodular_func_coverage = init_submodular_func_coverage
        self.E = E
        self.partitions = partitions
        self.inverse_partition = self.inverse_index()

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

    def scaled_greedy_criterion(self, skills_covered, e):
        """
        Calculates the contribution of element e to greedy solution
        :param sol:
        :param e:
        :return greedy_contrib:
        """
        # Weight scaling is constant
        p_id = self.inverse_partition[e]
        if self.partitions[p_id]['k'] == 0:
            return -float("inf")

        rho = 2
        marginal_gain = self.calc_marginal_gain(skills_covered, e)
        weighted_cost = rho * self.cost_func([e])
        greedy_contrib = marginal_gain - weighted_cost
        return greedy_contrib

    def original_greedy_criterion(self, skills_covered, e):
        """
        Calculates the contribution of element e to greedy solution
        :param sol:
        :param e:
        :return greedy_contrib:
        """
        # No weight scaling
        p_id = self.inverse_partition[e]
        if self.partitions[p_id]['k'] == 0:
            return -float("inf")
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

    def inverse_index(self):
        """
        Creates an inverse index where the key is the user and the value 
        is the partition id the user belongs to
        :param :
        """
        inverse_partition = {}
        for p_id, inner_dict in self.partitions.items():
            for user in inner_dict['users']:
                inverse_partition[user] = p_id

        return inverse_partition

    def update_valid_elements(self, greedy_element):
        """
        Updates the set of valid elements for selection
        :param greedy_element:
        """
        p_id = self.inverse_partition[greedy_element]
        self.partitions[p_id]['k'] -= 1

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
        # Initialize valid elements 
        self.N = self.E.copy()
        # Initialize the submodular function coverage skills
        self.skills_covered = self.init_submodular_func_coverage()
        # Sort greedy elements based on  objective
        sorted_greedy_elements = self.sort_greedy_elements(self.E, self.skills_covered)

        for i in range(0, len(self.E)):
            greedy_element = sorted_greedy_elements[i]
            p_id = self.inverse_partition[greedy_element]
            if not self.N:
                break

            # Element is added to the solution wrt the original objective
            if self.original_greedy_criterion(self.skills_covered, greedy_element) >= 0 and self.partitions[p_id]['k'] > 0:
                curr_sol.append(greedy_element)
                submodular_gain, self.skills_covered = self.submodular_func(self.skills_covered, [greedy_element])
                curr_val += submodular_gain
                self.update_valid_elements(greedy_element)
            else:
                self.N.remove(greedy_element)

        # Computing the original objective value for current solution
        curr_val = curr_val - self.cost_func(curr_sol)
        self.logger.info("Best solution: {}\nBest value: {}\ni: {}".format(curr_sol, curr_val, i))

        return curr_sol