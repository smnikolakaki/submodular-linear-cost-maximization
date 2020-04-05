"""
This class implements 2 * cost scaled greedy algorithm
1/2 approximation
"""
import logging
import numpy as np

class CostScaledGreedy(object):
    """
    2 * cost scaled greedy algorithm implementation
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
        # print('Previous value:',prev_val)
        new_val, skills_covered = self.submodular_func(skills_covered, [e])
        # print('New value:',new_val)
        marginal_gain = new_val - prev_val
        # print('Marginal gain:',marginal_gain)
        return marginal_gain

    def scaled_greedy_criterion(self, skills_covered, e):
        """
        Calculates the contribution of element e to greedy solution
        :param sol:
        :param e:
        :return greedy_contrib:
        """
        # Weight scaling is constant
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
        rho = 1
        marginal_gain = self.calc_marginal_gain(skills_covered, e)
        weighted_cost = rho * self.cost_func([e])
        greedy_contrib = marginal_gain - weighted_cost
        return greedy_contrib

    def find_greedy_element(self, E, skills_covered):
        """
        Finds the greedy element e to add to the current solution sol
        :param E:
        :param sol:
        :param k:
        :return e:
        """

        greedy_element = max(E, key=lambda x: self.scaled_greedy_criterion(skills_covered, x))
    
        return greedy_element

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

        for i in range(0, self.k):
            # Greedy element decided wrt scaled objective
            greedy_element = self.find_greedy_element(self.E, self.skills_covered)
    
            # Element is added to the solution wrt the original objective
            if self.scaled_greedy_criterion(self.skills_covered, greedy_element) >= 0:
                # print('Appending to solution:',curr_sol,'element:',greedy_element)
                curr_sol.append(greedy_element)
                submodular_gain, self.skills_covered = self.submodular_func(self.skills_covered, [greedy_element])
                curr_val += submodular_gain
                # print('Current submodular gain:',submodular_gain,'Indices with elements equal to zero:',np.where(self.skills_covered == 0)[0],'Number of indices:',len(np.where(self.skills_covered == 0)[0]))
                # print()

        # Computing the original objective value for current solution
        curr_val = curr_val - self.cost_func(curr_sol)
        self.logger.info("Best solution: {}\nBest value: {}".format(curr_sol, curr_val))

        return curr_sol
