"""
This class implements the scaled single-threshold Greedy algorithm
1/2(3 - sqrt(5)) approximation
"""
import logging
import numpy as np
import collections
import operator
import sys

class ScaledSingleThresholdGreedy(object):
    """
    Scaled single-threshold Greedy algorithm implementation
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
        self.k = k
        self.epsilon = self.config['algorithms']['scaled_single_threshold_greedy_config']['epsilon']

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

    def calc_scaled_objective(self, skills_covered, user_id, sol, val):
        """
        Calculates the scaled objective
        :param sol:
        :return obj_val:
        """
        # Weight scaling is constant c
        c = (1/2)*(3 + np.sqrt(5))
        submodular_gain, skills_covered = self.submodular_func(skills_covered, user_id)
        val = val + submodular_gain
        weighted_cost = c * self.cost_func(sol)
        obj_val = val - weighted_cost
        return obj_val

    def get_set_of_thresholds(self, m):
        """
        Returns the set of thresholds
        :param m:
        :return O:
        """
        O = []
        lb = m
        ub = 2 * self.k * m

        if m == 0 or self.k == 0:
            li = 0
            ui = 1
        else:
            li = np.log(m) / np.log(1 + self.epsilon)
            ui = np.log(2 * self.k * m) / np.log(1 + self.epsilon) + 1

        li = int(np.ceil(li)) # smallest integer greater than li
        ui = int(np.floor(ui)) # largest integer not greater than ui

        for i in range(li,ui):
            v = np.power((1+self.epsilon), i)
            if lb <= v and v <= ub:
                O.append(v)
            if v > ub:
                break
        return O

    def update_set_keys(self, S,O):
        """
        Updates the sets of the thresholds
        :param S:
        :param O:
        :return S:
        """
        # Create empty Sv for v in Oi that are new
        for v in O:
            if v not in S:
                S[v] = {}
                S[v]['solution'] = list()
                S[v]['skills_covered'] = self.init_submodular_func_coverage()
                S[v]['value'] = 0

        # Delete sets Sv for v that do not exist in Oi
        S_vs = set(S.keys())
        O_set = set(O)
        remove_vs = S_vs - O_set
        for v in remove_vs:
            del S[v]

        return S

    def scaled_greedy_criterion(self, skills_covered, e):
        """
        Calculates the contribution of element e to greedy solution
        :param sol:
        :param e:
        :return greedy_contrib:
        """
        # Weight scaling is constant c
        c = (1/2)*(3 + np.sqrt(5))
        marginal_gain = self.calc_marginal_gain(skills_covered, e)
        weighted_cost = c * self.cost_func([e])
        greedy_contrib = marginal_gain - weighted_cost
        return greedy_contrib

    def update_sets_new_element(self, v, e, S):
        """
        Updates the sets with the new element
        :param v:
        :param e:
        :param S:
        :return :
        """
        Sv_solution = S[v]['solution']
        Sv_skills_covered = S[v]['skills_covered']
        Sv_value = S[v]['value']

        if self.k == 0:
            return S

        # Threshold tau wrt the value of the scaled objective - from original paper
        denominator = self.k - len(Sv_solution)
        if denominator == 0:
            return S
        nominator = (v/2) - self.calc_scaled_objective(Sv_skills_covered, [], Sv_solution, Sv_value)
        tau = nominator / denominator

        # tau = (1/self.k)*((1/2)*(3 - np.sqrt(5))*Sv_value - self.cost_func(Sv_solution))

        # Marginal gain wrt scaled objective
        marg_gain = self.scaled_greedy_criterion(Sv_skills_covered, e)
        
        if tau < 0 :
            tau = 0
        if marg_gain >= tau and len(Sv_solution) < self.k:
            S[v]['solution'].append(e)
            submodular_gain, skills_covered = self.submodular_func(Sv_skills_covered, [e])
            S[v]['skills_covered'] = skills_covered
            S[v]['value'] = Sv_value + submodular_gain

        return S

    def find_max(self, S):
        max_solution = []
        max_value = -float("inf")

        for v, nested_dict in S.items():
            # print('Nested dictionary:',nested_dict['solution'])
            submodular_value = nested_dict['value']; solution = nested_dict['solution']
            value = submodular_value - self.cost_func(solution)
            if max_value < value:
                max_value = value
                max_solution = solution

        return max_solution, max_value

    def run(self):
        """
        Execute algorithm
        :param:
        :return:
        """
        print(self.epsilon)
        curr_sol = []
        curr_val = 0
        S = collections.defaultdict(list)
        m = 0

        # Initialize the submodular function coverage skills
        self.skills_covered = self.init_submodular_func_coverage()

        for e_i in self.E:
            # Thresholds defined over the scaled objective value
            m = max(m, self.calc_scaled_objective(self.skills_covered,[e_i],[],0))   
            # Creating set of thresholds
            Oi = self.get_set_of_thresholds(m)
            # Update the set Sv keys
            S = self.update_set_keys(S,Oi)
            # Update the sets Sv with new element in parallel
            for v in Oi:
                S = self.update_sets_new_element(v, e_i, S)
        if S:
            # Return the solution that maximizes original objective value
            curr_sol, curr_val = self.find_max(S)
            # print(max(S, key=lambda sol: S[sol]['value'] - self.cost_func(S[sol]['solution'])))

        self.logger.info("Best solution: {}\nBest value: {}".format(curr_sol, curr_val))

        return curr_sol 
