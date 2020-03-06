"""
This class implements the scaled single-threshold Greedy algorithm
1/2(3 - sqrt(5)) approximation
"""
import logging
import numpy as np
import collections
import operator
import sys
# import multiprocessing as mp

# from pathos.pools import ProcessPool

class ScaledSingleThresholdGreedy(object):
    """
    Scaled single-threshold Greedy algorithm implementation
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
        self.k = k
        self.epsilon = self.config['algorithms']['scaled_single_threshold_greedy_config']['epsilon']

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

    def calc_scaled_objective(self, sol):
        """
        Calculates the scaled objective
        :param sol:
        :return obj_val:
        """
        # Weight scaling is constant c
        c = (1/2)*(3 + np.sqrt(5))

        val = self.submodular_func(sol)
        weighted_cost = c * self.cost_func(sol)
        obj_val = val - weighted_cost
        return obj_val

    def get_set_of_thresholds(self, m):
        """
        Returns the set of thresholds
        :param m:
        :return O:
        """
        O = set([])
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
                O.add(v)
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
                S[v] = set()

        # Delete sets Sv for v that do not exist in Oi
        S_vs = set(S.keys())
        remove_vs = S_vs - O
        for v in remove_vs:
            del S[v]

        return S

    def greedy_criterion(self, sol, e):
        """
        Calculates the contribution of element e to greedy solution
        :param sol:
        :param e:
        :return greedy_contrib:
        """
        # Weight scaling is constant c
        c = (1/2)*(3 + np.sqrt(5))
        marginal_gain = self.calc_marginal_gain(sol, e)
        weighted_cost = c * self.cost_func(set({e}))
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
        Sv = S[v].copy()
        denominator = self.k - len(Sv)
        if denominator == 0:
            return S

        marg_gain = self.greedy_criterion(Sv.copy(), e)
        nominator = (v/2) - self.calc_scaled_objective(Sv)

        tau = nominator / denominator

        if tau < 0 :
            tau = 0
        if marg_gain >= tau and len(Sv) < self.k:
            S[v].add(e)
        return S

    def get_arguments(self, S, e_i):
        """
        Creates the arguments for parallel code
        :param S:
        :param e_i:
        :return args:
        """
        args = []
        for v, sol in S.items():
            args.append((v, e_i, S[v].copy(), self.submodular_func, self.cost_func, self.k))
        return args 

    def combine_result(self, result):
        """
        Combining results from parallelization
        :param result:
        :return S:
        """
        S = collections.defaultdict(set)
        for v,sol in result:
           S[v] = sol 
        return S

    @staticmethod
    def run_algorithm(arg):
        v = arg[0]; e_i = arg[1]; Sv = arg[2]; submodular_func = arg[3]; cost_func = arg[4]; k = arg[5]

        def calc_marginal_gain(sol):
            """
            Calculates the marginal gain for adding element e to the current solution sol
            :param sol:
            :param e:
            :return marginal_gain:
            """
            prev_val = submodular_func(sol)
            sol.add(e_i)
            new_val = submodular_func(sol)
            marginal_gain = new_val - prev_val
            return marginal_gain

        def calc_scaled_objective(sol):
            """
            Calculates the scaled objective
            :param sol:
            :return obj_val:
            """
            # Weight scaling is constant c
            c = (1/2)*(3 + np.sqrt(5))

            val = submodular_func(sol)
            weighted_cost = c * cost_func(sol)
            obj_val = val - weighted_cost
            return obj_val

        def greedy_criterion(sol):
            """
            Calculates the contribution of element e to greedy solution
            :param sol:
            :param e:
            :return greedy_contrib:
            """
            # Weight scaling is constant c
            c = (1/2)*(3 + np.sqrt(5))
            marginal_gain = calc_marginal_gain(sol)
            weighted_cost = c * cost_func(set({e_i}))
            greedy_contrib = marginal_gain - weighted_cost
            return greedy_contrib

        marg_gain = greedy_criterion(Sv.copy())
        nominator = (v/2) - calc_scaled_objective(Sv)
        denominator = k - len(Sv)
        # print("Marg gain:",marg_gain,"Nominator",nominator,"Denominator:",denominator)
        tau = nominator / denominator
        if tau < 0 :
            tau = 0
        if marg_gain >= tau and len(Sv) < k:
            Sv.add(e_i)
        return (v,Sv)

    def run(self):
        """
        Execute algorithm
        :param:
        :return:
        """

        # Non-parallel version
        curr_sol = set([])
        S = collections.defaultdict(set)
        m = 0

        for i in range(0, len(self.E)):
            e_i = i
            # self.logger.info("Outer ei: {}".format(e_i))
            m = max(m, self.calc_scaled_objective(set({e_i})))       
            # Creating set of thresholds
            Oi = self.get_set_of_thresholds(m)
            # Update the set Sv keys
            S = self.update_set_keys(S,Oi)
            # Update the sets Sv with new element in parallel
            for v in Oi:
                self.update_sets_new_element(v, e_i, S)
        
        if S:
            curr_sol = max(S.items(), key=lambda sol: self.submodular_func(sol[1]) - self.cost_func(sol[1]))[1]
        curr_val = self.submodular_func(curr_sol) - self.cost_func(curr_sol)
        self.logger.info("Best solution: {}\nBest value: {}".format(curr_sol, curr_val))

        # # Parallel version
        # curr_sol = set([])
        # S = collections.defaultdict(set)
        # m = 0

        # num_processes = mp.cpu_count()
        # self.logger.info("Processes: {}".format(num_processes))
        # # self.logger.info("Args: {}\n".format(args))
        # pool = ProcessPool(nodes=1)
        # print('Num elements:',len(self.E))
        # for i in range(0, len(self.E)):
        #     e_i = i

        #     self.logger.info("Outer ei: {}".format(e_i))
        #     m = max(m, self.calc_scaled_objective(set({e_i})))       
        #     # Creating set of thresholds
        #     Oi = self.get_set_of_thresholds(m)
        #     # Update the set Sv keys
        #     S = self.update_set_keys(S,Oi)
        #     args = self.get_arguments(S,e_i)
        #     # print('m:',m,'Oi:',Oi,'S:',S,'e_i:',e_i)
        #     # Update the sets Sv with new element in parallel
        #     # Create a pool of processes
        #     results = pool.amap(self.run_algorithm, args).get()
        #     # print('results:',results)
        #     S = self.combine_result(results)
        #     # print('Here S:',S)

        # pool.terminate()
        # pool.join()
        # pool.clear()
        # curr_sol = max(S.items(), key=lambda sol: self.submodular_func(sol[1]) - self.cost_func(sol[1]))[1]
        # curr_val = self.submodular_func(curr_sol) - self.cost_func(curr_sol)
        # self.logger.info("Best solution: {}\nBest value: {}".format(curr_sol, curr_val))

        return curr_sol 