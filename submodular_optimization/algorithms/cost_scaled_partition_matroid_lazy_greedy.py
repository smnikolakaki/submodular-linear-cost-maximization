"""
This class implements 2 * cost scaled matroid constraint greedy algorithm with lazy exact evaluation
1/2 * approximation
"""
import logging
import numpy as np
import sys
from heapq import heappush
from heapq import heappop

class CostScaledPartitionMatroidLazyGreedy(object):
    """
    2 * cost scaled matroid constraint greedy algorithm implementation using exact lazy evaluation
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

    def initialize_max_heap(self):
        """
        Initializes the max heap with elements e with key their score contribution to the empty set
        :return H:
        """
        self.H = []

        rho = 2

        for idx in self.E:
            submodular_score, skills_covered = self.submodular_func(self.skills_covered, [idx])
            new_scaled_gain = submodular_score - rho * self.cost_func([idx])
            new_original_gain = submodular_score - self.cost_func([idx])
            # Do not add an element into the heap if its marginal value is negative
            # Remove this element from the valid elements
            if new_original_gain < 0:
                p_id = self.inverse_partition[idx]
                self.partitions[p_id]['users'].remove(idx)
                self.N.remove(idx)
                continue

            # Multiplying inserted element with -1 to convert to min heap to max
            heappush(self.H, (-1 * new_scaled_gain, idx))

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
        p_id = self.inverse_partition[e]
        if self.partitions[p_id]['k'] == 0:
            return -float("inf")
        # No weight scaling
        rho = 1
        marginal_gain = self.calc_marginal_gain(skills_covered, e)
        weighted_cost = rho * self.cost_func([e])
        greedy_contrib = marginal_gain - weighted_cost
        return greedy_contrib

    def find_lazy_exact_greedy_eval_element(self, skills_covered, k):
        """
        Finds the greedy element e to add to the current solution sol
        :param sol:
        :return e:
        """

        rho = 2

        if not self.H:
            return None

        heap_size = len(self.H)
        # Perform lazy evaluation of elements in heap H
        for i in range(heap_size):
            # Retrieving top element in the heap and computing its updated gain
            (prev_gain, idx) = heappop(self.H)              
            # Multiplying popped element with -1 to convert to its original gain
            prev_gain = -1 * prev_gain

            marginal_gain = self.calc_marginal_gain(skills_covered, idx)
            new_scaled_gain = marginal_gain - rho * self.cost_func([idx])
            new_original_gain = marginal_gain - self.cost_func([idx])
            # For k == 1 return the top element of the heap with the largest gain if that is positive
            # else return None
            if k == 1:
                if new_original_gain > 0:
                    return idx
                else:
                    return None

            # If there is no element left in the heap
            if not self.H:
                # Return the popped element if the new gain is positive
                if new_original_gain > 0:
                    return idx
                else:
                    if idx in self.N:
                        self.N.remove(idx)
                    return None
            
            # Retrieving the outdated gain of the next element in the heap
            (next_element_scaled_gain, next_element_idx) = self.H[0]
            # Multiplying popped element with -1 to convert to its original gain
            next_element_scaled_gain = -1 * next_element_scaled_gain

            # For k != 1 do lazy exact greedy evaluation
            if new_scaled_gain >= next_element_scaled_gain:
                return idx
            elif new_original_gain >= 0:
                # Multiplying inserted element with -1 to convert to min heap to max
                heappush(self.H, (-1 * new_scaled_gain, idx))
            else:
                # Removing the element from the heap and the set of valid elements
                if idx in self.N:
                    self.N.remove(idx)
                continue

        return None

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
        # Initialize the max heap for a given value of ks
        self.initialize_max_heap()

        for i in range(1, len(self.E) + 1):
            if not self.N:
                break
            # Greedy element decided wrt scaled objective
            greedy_element = self.find_lazy_exact_greedy_eval_element(self.skills_covered, i)
            # If the element is not valid anymore continuec
            if greedy_element not in self.N:
                continue
            # If an element is returned it is added to the solution wrt the original objective
            if greedy_element and self.scaled_greedy_criterion(self.skills_covered, greedy_element) >= 0:
                curr_sol.append(greedy_element)
                submodular_gain, self.skills_covered = self.submodular_func(self.skills_covered, [greedy_element])
                curr_val += submodular_gain
                self.update_valid_elements(greedy_element)
            elif greedy_element:
                self.N.remove(greedy_element)

        # Computing the original objective value for current solution
        curr_val = curr_val - self.cost_func(curr_sol)
        self.logger.info("Best solution: {}\nBest value: {}".format(curr_sol, curr_val))

        return curr_sol
