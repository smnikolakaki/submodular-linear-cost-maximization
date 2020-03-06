"""
This class implements ExperimentTest - used for debugging purposes

"""

import logging
import pandas as pd
import numpy as np
from data.data_provider import DataProvider
from data.data_exporter import DataExporter
from algorithms.algorithm_driver import AlgorithmDriver
from algorithms.set_cover_greedy import SetCoverGreedy

class ExperimentTest(object):
    """
    ExperimentTest class
    """

    def __init__(self, config):
        """
        Constructor

        :param config:
        :return:
        """
        self.config = config
        self.logger = logging.getLogger("so_logger")
        self.data_provider = DataProvider(self.config)
        self.data_exporter = DataExporter(self.config)

    @staticmethod
    def run_algorithm(args):
        # Run algorithm
        alg = AlgorithmDriver()
        data = alg.run(*args)
        return data

    def set_scaling_factor(self,data):

        # Find appropriate scaling factor
        alg = SetCoverGreedy(self.config, data.submodular_func, data.E)
        sol = alg.run()
        submodular_val = data.submodular_func(sol)
        cost = data.cost_func(sol)
        scaling_factor = data.scaling_func(submodular_val, cost)

        # Update scaling factor
        data.scaling_factor = scaling_factor

    def run(self):
        """
        Run experiment
        :param:
        :return:
        """
        self.logger.info("Starting experiment test")

        self.expt_config = self.config['experiment_configs']['experiment_02']
        popular_threshold = self.expt_config['popular_threshold']
        rare_threshold = self.expt_config['rare_threshold']

        # user_sample_ratios = [0.005,0.1,0.2,0.3,0.4,0.5]
        # seeds = [i for i in range(6,11)]
        # ks = [5,10,15,20,25,30,35,40,45,50]

        user_sample_ratios = [0.4]
        seeds = [2]
        ks = [15]

        sampling_epsilon_values = [0.01,0.3]

        num_sampled_skills = 50
        rare_sample_fraction = 0.1
        popular_sample_fraction = 0.1
        scaling_factor = 800

        alg = AlgorithmDriver()
        results = []
        for seed in seeds:
            for user_sample_ratio in user_sample_ratios:
                self.logger.info("Experiment for user sample ratio: {} and scaling factor: {}".format(user_sample_ratio,scaling_factor))

                # Load dataset
                data = self.data_provider.read_guru_data_obj()


                self.logger.info("Scaling factor for submodular function is: {}".format(scaling_factor))

                # Distorted Greedy
                config = self.config.copy()
                for k in ks:
                    result = alg.run(self.config, data, "distorted_greedy",
                         None, None, scaling_factor, num_sampled_skills,
                         rare_sample_fraction, popular_sample_fraction, rare_threshold, popular_threshold,
                         user_sample_ratio, seed, k)
                    results.append(result)

                # Cost Scaled Greedy
                print("In cost scaled greedy")
                config = self.config.copy()
                result = alg.run(self.config, data, "cost_scaled_lazy_exact_greedy",
                     None, None, scaling_factor, num_sampled_skills,
                     rare_sample_fraction, popular_sample_fraction, rare_threshold, popular_threshold,
                     user_sample_ratio, seed, None)
                # For cardinality constraints of size k we find the prefix of size k
                # and compute the score, cost, submodular val  of that solution and update result 
                for k in ks:
                    result_k = result.copy()
                    if k < len(result['sol']):
                        sol_k = set(list(result['sol'])[:k])
                        submodular_val = data.submodular_func(sol_k)
                        cost = data.cost_func(sol_k)
                        val = submodular_val - cost
                        result_k['sol'] = sol_k; result_k['val'] = val; result_k['submodular_val'] = submodular_val;
                        result_k['cost'] = cost; result_k['k'] = k;
                    else:
                        sol_k = result['sol']
                        val = result['val']

                    result_k['k'] = k;
                    self.logger.info("Best solution cost scaled: {}\nBest value: {}".format(sol_k, val))
                    results.append(result_k)

                # Scaled Single Threshold Greedy
                config = self.config.copy()
                for k in ks:
                    for sample_epsilon in sampling_epsilon_values:
                        result = alg.run(self.config, data, "scaled_single_threshold_greedy",
                             sample_epsilon, None, scaling_factor, num_sampled_skills,
                             rare_sample_fraction, popular_sample_fraction, rare_threshold, popular_threshold,
                             user_sample_ratio, seed, k)
                        results.append(result)

                # Scaled Single Threshold Greedy Max Val
                config = self.config.copy()
                for k in ks:
                    for sample_epsilon in sampling_epsilon_values:
                        result = alg.run(self.config, data, "scaled_single_threshold_max_val_greedy",
                             sample_epsilon, None, scaling_factor, num_sampled_skills,
                             rare_sample_fraction, popular_sample_fraction, rare_threshold, popular_threshold,
                             user_sample_ratio, seed, k)
                        results.append(result)


                self.logger.info("\n")

        self.logger.info("Finished experiment test")
