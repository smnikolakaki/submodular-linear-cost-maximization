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
        self.logger.info("Starting experiment 04")

        self.expt_config = self.config['experiment_configs']['experiment_04']
        popular_threshold = self.expt_config['popular_threshold']
        rare_threshold = self.expt_config['rare_threshold']
        num_of_partitions = self.expt_config['num_of_partitions']
        partition_type = self.expt_config['partition_type']
        cardinality_constraint = self.expt_config['cardinality_constraint']

        user_sample_ratios = [1]
        seeds = [i for i in range(6,10)]
        cardinality_constraints = [i for i in range(1,11)]
        num_of_partitions = [i for i in range(1,6)]

        num_sampled_skills = 50
        rare_sample_fraction = 0.1
        popular_sample_fraction = 0.1
        scaling_factor = 800

        alg = AlgorithmDriver()
        results = []
        for seed in seeds:
            for user_sample_ratio in user_sample_ratios:
                for cardinality_constraint in cardinality_constraints:
                    for num_of_partition in num_of_partitions:
                        self.logger.info("Experiment for user sample ratio: {} and scaling factor: {} and seed: {} and cardinality constraint:{} and num of partitions:{} ".format(user_sample_ratio,scaling_factor,seed,cardinality_constraint,num_of_partition))

                        # Load dataset
                        data = self.data_provider.read_freelancer_data_obj()
                        config = self.config.copy()
                        # Creating the ground set of users
                        alg.create_sample(config, data, num_sampled_skills, rare_sample_fraction, popular_sample_fraction, 
                                            rare_threshold,popular_threshold, user_sample_ratio, seed)
                        # Assigning users to partitions uniformly at random
                        alg.create_partitions(data, num_of_partition, partition_type, cardinality_constraint)

                        self.logger.info("Scaling factor for submodular function is: {}".format(scaling_factor))
                        
                        # # Partition matroid greedy
                        # start = timer()
                        # result = alg.run(config, data, "partition_matroid_greedy",
                        #      None, None, scaling_factor, num_sampled_skills,
                        #      rare_sample_fraction, popular_sample_fraction, rare_threshold, popular_threshold,
                        #      user_sample_ratio, seed, None)
                        # end = timer()
                        # result['runtime'] = end - start
                        # result['cardinality_constraint'] = cardinality_constraint
                        # result['num_of_partitions'] = num_of_partition
                        # self.logger.info("Algorithm: {} and k: {} and runtime: {}".format("partition_matroid_greedy",None,end - start))
                        # results.append(result)

                        # self.logger.info("\n")

                        # Cost scaled partition matroid greedy
                        start = timer()
                        result = alg.run(config, data, "cost_scaled_partition_matroid_greedy",
                             None, None, scaling_factor, num_sampled_skills,
                             rare_sample_fraction, popular_sample_fraction, rare_threshold, popular_threshold,
                             user_sample_ratio, seed, None)
                        end = timer()
                        result['runtime'] = end - start
                        result['cardinality_constraint'] = cardinality_constraint
                        result['num_of_partitions'] = num_of_partition
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
                        result['cardinality_constraint'] = cardinality_constraint
                        result['num_of_partitions'] = num_of_partition
                        self.logger.info("Algorithm: {} and k: {} and runtime: {}".format("cost_scaled_partition_matroid_lazy_greedy",None,end - start))
                        results.append(result)

                        self.logger.info("\n")


                        # # Baseline Top k
                        # start = timer()
                        # result = alg.run(config, data, "baseline_topk_matroid",
                        #      None, None, scaling_factor, num_sampled_skills,
                        #      rare_sample_fraction, popular_sample_fraction, rare_threshold, popular_threshold,
                        #      user_sample_ratio, seed, None)
                        # end = timer()
                        # result['runtime'] = end - start
                        # result['cardinality_constraint'] = cardinality_constraint
                        # result['num_of_partitions'] = num_of_partition
                        # self.logger.info("Algorithm: {} and k: {} and runtime: {}".format("baseline_topk_matroid",None,end - start))
                        # results.append(result)

                        # self.logger.info("\n")


        self.logger.info("Finished experiment 04")

        # Export results
        df = pd.DataFrame(results)
        self.data_exporter.export_csv_file(df, "experiment_04_freelancer_salary_pop01_rare01_cost_scaled.csv")
        self.logger.info("Exported experiment 04 results")
