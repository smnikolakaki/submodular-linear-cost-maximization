"""
This class implements stochastic distorted greedy algorithm
(1 - 1/e - epsilon) approximation in expectation
"""
import logging
import numpy as np


class StochasticDistortedGreedy(object):
    """
    StochasticDistored Greedy algorithm implementation
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
        self.epsilon = self.config['algorithms']['stochastic_distorted_greedy_config']['epsilon']
        if k == None:
            self.k = len(self.E)
        else:
            self.k = k

    def calc_sample_size(self, k):
        """
        Calculates sample size for stochastic distorted greedy
        :param k:
        :return s:
        """
        s = np.ceil((len(self.E) / k) * np.log(1 / self.epsilon))
        return int(s)

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

    def distorted_greedy_criterion(self, skills_covered, e, k, i, gamma=1):
        """
        Calculates the contribution of element e to greedy solution
        :param sol:
        :param e:
        :param k:
        :param i:
        :param gamma:
        :return greedy_contrib:
        """
        # Weight scaling is distorted
        rho = (k - gamma) / k
        marginal_gain = self.calc_marginal_gain(skills_covered, e)
        weighted_gain = rho**(k - i - 1) * marginal_gain
        cost = self.cost_func([e])
        greedy_contrib = weighted_gain - cost
        return greedy_contrib

    def find_greedy_element(self, E, skills_covered, k, i):
        """
        Finds the greedy element e to add to the current solution sol
        :param E:
        :param sol:
        :param k:
        :param i:
        """
        greedy_element = max(E, key=lambda x: self.distorted_greedy_criterion(skills_covered, x, k, i))
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
        # print('1 Indices with elements equal to zero:',np.where(self.skills_covered == 0)[0],'Number of indices:',len(np.where(self.skills_covered == 0)[0]))

        for i in range(0, self.k):
            s = self.calc_sample_size(self.k)
            B = set(np.random.choice(list(self.E), size=s))
            # Greedy element decided wrt distorted objective
            greedy_element = self.find_greedy_element(B, self.skills_covered, self.k, i)
            # Element is added to the solution wrt distorted objective
            if self.distorted_greedy_criterion(self.skills_covered, greedy_element, self.k, i) > 0:
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
