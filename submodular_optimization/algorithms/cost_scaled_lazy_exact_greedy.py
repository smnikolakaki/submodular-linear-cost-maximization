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

    def initialize_max_heap(self):
        """
        Initializes the max heap with elements e with key their score contribution to the empty set
        :return H:
        """
        self.H = []

        rho = 2

        for idx in self.E:
            submodular_score = self.submodular_func({idx})
            new_gain = submodular_score - rho * self.cost_func({idx})
            # Do not add an element into the heap if its marginal value is negative
            if new_gain < 0:
                continue

            # Multiplying inserted element with -1 to convert to min heap to max
            heappush(self.H, (-1 * new_gain, idx))

    def find_lazy_exact_greedy_eval_element(self, sol, k):
        """
        Finds the greedy element e to add to the current solution sol
        :param sol:
        :param k:c
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

            marginal_gain = self.calc_marginal_gain(sol.copy(), idx)
            new_gain = marginal_gain - rho * self.cost_func({idx})

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
            elif new_gain > 0:
                # Multiplying inserted element with -1 to convert to min heap to max
                heappush(self.H, (-1 * new_gain, idx))
            else:
                # Removing the element from the heap
                continue

        # If heap empties and there is no element satisfying the conditions return None
        return None

    def run(self):
        """
        Execute algorithm
        :param:
        :return best_sol:
        """

        # Keep track of best solution of any value of k
        best_sol = set([])
        best_val = -1 * np.inf

        # Keep track of current solution for a given value of k
        curr_sol = set([])
        curr_val = 0

        # Initialize the max heap for a given value of ks
        self.initialize_max_heap()

        for k in range(1, len(self.E)):
            lazy_greedy_element = self.find_lazy_exact_greedy_eval_element(curr_sol,k)

            if lazy_greedy_element:
                curr_sol.add(lazy_greedy_element)

        # If the solution for current value of k is better than previous best solution
        # update the overall best solution
        curr_val = self.submodular_func(curr_sol) - self.cost_func(curr_sol)

        if curr_val > best_val:
            best_sol = curr_sol.copy()
            best_val = curr_val

        self.logger.info("Best solution: {}\nBest value: {}".format(best_sol, best_val))

        return best_sol
