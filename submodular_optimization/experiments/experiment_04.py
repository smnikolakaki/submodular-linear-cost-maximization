"""
This class implements experiment_04

"""

import logging
import pandas as pd
import numpy as np
import sys
from data.data_provider import DataProvider
from data.data_exporter import DataExporter
from algorithms.algorithm_driver import AlgorithmDriver
from algorithms.set_cover_greedy import SetCoverGreedy
from timeit import default_timer as timer

class Experiment04(object):
    """
    Experiment04 class
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

        self.expt_config = self.config['experiment_configs']['experiment_04']
        popular_threshold = self.expt_config['popular_threshold']
        rare_threshold = self.expt_config['rare_threshold']
        num_of_partitions = self.expt_config['num_of_partitions']
        partition_type = self.expt_config['partition_type']

        user_sample_ratios = [0.001,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
        seeds = [i for i in range(6,11)]

        num_sampled_skills = 50
        rare_sample_fraction = 0.1
        popular_sample_fraction = 0.1
        scaling_factor = 800

        alg = AlgorithmDriver()
        results = []
        for seed in seeds:
            for user_sample_ratio in user_sample_ratios:
                self.logger.info("Experiment for user sample ratio: {} and scaling factor: {} and seed: {}".format(user_sample_ratio,scaling_factor,seed))

                # Load dataset
                data = self.data_provider.read_guru_data_obj()
                config = self.config.copy()
                # Creating the ground set of users
                alg.create_sample(config, data, num_sampled_skills, rare_sample_fraction, popular_sample_fraction, 
                                    rare_threshold,popular_threshold, user_sample_ratio, seed)
                # Assigning users to partitions uniformly at random
                alg.create_partitions(data, num_of_partitions, partition_type)

                self.logger.info("Scaling factor for submodular function is: {}".format(scaling_factor))
                
                # Partition matroid greedy
                start = timer()
                result = alg.run(config, data, "partition_matroid_greedy",
                     None, None, scaling_factor, num_sampled_skills,
                     rare_sample_fraction, popular_sample_fraction, rare_threshold, popular_threshold,
                     user_sample_ratio, seed, None)
                end = timer()
                result['runtime'] = end - start
                self.logger.info("Algorithm: {} and k: {} and runtime: {}".format("partition_matroid_greedy",None,end - start))
                results.append(result)
                
                self.logger.info("\n")

                # Cost scaled partition matroid greedy
                start = timer()
                result = alg.run(config, data, "cost_scaled_partition_matroid_greedy",
                     None, None, scaling_factor, num_sampled_skills,
                     rare_sample_fraction, popular_sample_fraction, rare_threshold, popular_threshold,
                     user_sample_ratio, seed, None)
                end = timer()
                result['runtime'] = end - start
                self.logger.info("Algorithm: {} and k: {} and runtime: {}".format("cost_scaled_partition_matroid_greedy",None,end - start))
                results.append(result)

                self.logger.info("\n")

                # Cost scaled partition matroid lazy exact greedy
                start = timer()
                result = alg.run(config, data, "cost_scaled_partition_matroid_lazy_greedy",
                     None, None, scaling_factor, num_sampled_skills,
                     rare_sample_fraction, popular_sample_fraction, rare_threshold, popular_threshold,
                     user_sample_ratio, seed, None)
                end = timer()
                result['runtime'] = end - start
                self.logger.info("Algorithm: {} and k: {} and runtime: {}".format("cost_scaled_partition_matroid_lazy_greedy",None,end - start))
                results.append(result)
                
                self.logger.info("\n")

                # Partition matroid greedy
                start = timer()
                result = alg.run(config, data, "cost_scaled_partition_matroid_scaled_lazy_greedy",
                     None, None, scaling_factor, num_sampled_skills,
                     rare_sample_fraction, popular_sample_fraction, rare_threshold, popular_threshold,
                     user_sample_ratio, seed, None)
                end = timer()
                result['runtime'] = end - start
                self.logger.info("Algorithm: {} and k: {} and runtime: {}".format("cost_scaled_partition_matroid_scaled_lazy_greedy",None,end - start))
                results.append(result)

                self.logger.info("\n")

        self.logger.info("Finished experiment 04")

        # Export results
        df = pd.DataFrame(results)
        self.data_exporter.export_csv_file(df, "experiment_04_salary.csv")
        self.logger.info("Exported experiment 04 results")
