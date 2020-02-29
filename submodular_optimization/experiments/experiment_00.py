"""
This class implements experiment_00

"""

import logging
import pandas as pd
import numpy as np
from data.data_provider import DataProvider
from data.data_exporter import DataExporter
from algorithms.algorithm_driver import AlgorithmDriver
from algorithms.set_cover_greedy import SetCoverGreedy

class Experiment00(object):
    """
    Experiment00 class
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
        self.logger.info("Starting experiment 00")

        self.expt_config = self.config['experiment_configs']['experiment_00']
        popular_threshold = self.expt_config['popular_threshold']
        rare_threshold = self.expt_config['rare_threshold']

        user_sample_ratios = [0.005,0.1,0.2,0.3,0.4]
        seeds = [i for i in range(6,11)]
        # lazy_eval_epsilon_values = [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]
        sampling_epsilon_values = [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]

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

                # # Create controlled samples dataset
                # data.sample_skills_to_be_covered_controlled(num_sampled_skills, rare_sample_fraction,
                #                                         popular_sample_fraction, rare_threshold,
                #                                         popular_threshold, user_sample_ratio)

                # # Setting scaling factor of coverage as coverage(S)/cost(S) for set cover solution S
                # self.set_scaling_factor(data)


                self.logger.info("Scaling factor for submodular function is: {}".format(scaling_factor))

                # Dstorted greedy - ICML
                config = self.config.copy()
                result = alg.run(self.config, data, "distorted_greedy",
                     None, None, scaling_factor, num_sampled_skills,
                     rare_sample_fraction, popular_sample_fraction, rare_threshold, popular_threshold,
                     user_sample_ratio,seed)
                results.append(result)

                # Cost scaled greedy
                config = self.config.copy()
                result = alg.run(self.config, data, "cost_scaled_greedy",
                     None, None, scaling_factor, num_sampled_skills,
                     rare_sample_fraction, popular_sample_fraction, rare_threshold, popular_threshold,
                     user_sample_ratio,seed)
                results.append(result)

                # Cost scaled lazy exact greedy
                config = self.config.copy()
                result = alg.run(config, data, "cost_scaled_lazy_exact_greedy",
                     None, None, scaling_factor, num_sampled_skills,
                     rare_sample_fraction, popular_sample_fraction, rare_threshold, popular_threshold,
                     user_sample_ratio,seed)
                results.append(result)

                # Unconstrained Linear 
                config = self.config.copy()
                result = alg.run(config, data, "unconstrained_linear",
                     None, None, scaling_factor, num_sampled_skills,
                     rare_sample_fraction, popular_sample_fraction, rare_threshold, popular_threshold,
                     user_sample_ratio,seed)
                results.append(result)

                # Unconstrained distorted greedy
                config = self.config.copy()
                for i in range(100):
                    result = alg.run(config, data, "unconstrained_distorted_greedy",
                     None, None, scaling_factor, num_sampled_skills,
                     rare_sample_fraction, popular_sample_fraction, rare_threshold, popular_threshold,
                     user_sample_ratio,seed)
                    results.append(result)

                # Stochastic distorted greedy
                config = self.config.copy()
                for sample_epsilon in sampling_epsilon_values:
                    for i in range(50):
                        config = self.config.copy()
                        config['algorithms']['stochastic_distorted_greedy_config']['epsilon'] = sample_epsilon
                        result = alg.run(config, data, "stochastic_distorted_greedy",
                             sample_epsilon, None, scaling_factor, num_sampled_skills,
                             rare_sample_fraction, popular_sample_fraction, rare_threshold, popular_threshold,
                             user_sample_ratio,seed)
                        results.append(result)

                self.logger.info("\n")

        self.logger.info("Finished experiment 00")

        # Export results
        df = pd.DataFrame(results)
        self.data_exporter.export_csv_file(df, "experiment_00.csv")
        self.logger.info("Exported experiment_00 results")
