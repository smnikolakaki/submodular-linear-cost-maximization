"""
This class implements 2 * cost scaled greedy algorithm with lazy exact evaluation
1/2 * approximation
"""
import logging
import numpy as np
import sys
from heapq import heappush
from heapq import heappop


class CostScaledLazyExactGreedy(object):
    """
    2 * cost scaled greedy algorithm implementation using lazy evaluation
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

    def initialize_max_heap(self):
        """
        Initializes the max heap with elements e with key their score contribution to the empty set
        :return H:
        """
        self.H = []

        rho = 2

        for idx in self.E:
            submodular_score, skills_covered = self.submodular_func(self.skills_covered, [idx])
            # print('Idx:',idx,'Submodular score:',submodular_score,'skills covered:',self.skills_covered)
            # print()
            new_gain = submodular_score - rho * self.cost_func([idx])
            # Do not add an element into the heap if its marginal value is negative
            if new_gain < 0:
                continue

            # Multiplying inserted element with -1 to convert to min heap to max
            heappush(self.H, (-1 * new_gain, idx))

    def find_lazy_exact_greedy_eval_element(self, skills_covered, k):
        """
        Finds the greedy element e to add to the current solution sol
        :param sol:
        :return e:
        """

        rho = 2

        # Perform lazy evaluation of elements in heap H
        while self.H:
            # Retrieving top element in the heap and computing its updated gain
            (prev_gain, idx) = heappop(self.H)              

            # Multiplying popped element with -1 to convert to its original gain
            prev_gain = -1 * prev_gain

            # For k == 1 return the top element of the heap with the largest gain if that is positive
            # else return None
            if k == 1:
                if prev_gain > 0:
                    return idx
                else:
                    return None

            marginal_gain = self.calc_marginal_gain(skills_covered, idx)
            new_gain = marginal_gain - rho * self.cost_func([idx])

            # If there is no element left in the heap
            if not self.H:
                # Return the popped element if the new gain is positive
                if new_gain > 0:
                    return idx
                else:
                    return None
           
            # Retrieving the outdated gain of the next element in the heap
            (next_element_gain, next_element_idx) = self.H[0]
            # Multiplying popped element with -1 to convert to its original gain
            next_element_gain = -1 * next_element_gain

            # For k != 1 do lazy exact greedy evaluation
            if new_gain >= next_element_gain:
                return idx
            elif new_gain >= 0:
                # Multiplying inserted element with -1 to convert to min heap to max
                heappush(self.H, (-1 * new_gain, idx))
            else:
                # Removing the element from the heap
                continue

        # If heap empties and there is no element satisfying the conditions return None
        return None

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
        # Initialize the max heap for a given value of ks
        self.initialize_max_heap()

        for i in range(1, self.k + 1):
            # Greedy element decided wrt scaled objective
            greedy_element = self.find_lazy_exact_greedy_eval_element(self.skills_covered, i)
            # print('Greedy element:',greedy_element)
            # If an element is returned it is added to the solution wrt the original objective
            if greedy_element and self.original_greedy_criterion(self.skills_covered, greedy_element) >= 0:
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
