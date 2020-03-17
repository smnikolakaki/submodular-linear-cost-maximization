"""
This class implements experiment_02

"""

import logging
import pandas as pd
import numpy as np
from data.data_provider import DataProvider
from data.data_exporter import DataExporter
from algorithms.algorithm_driver import AlgorithmDriver
from algorithms.set_cover_greedy import SetCoverGreedy
from timeit import default_timer as timer

class Experiment02(object):
    """
    Experiment02 class
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
        self.logger.info("Starting experiment 02")

        self.expt_config = self.config['experiment_configs']['experiment_02']
        popular_threshold = self.expt_config['popular_threshold']
        rare_threshold = self.expt_config['rare_threshold']

        user_sample_ratios = [0.4,1]
        seeds = [i for i in range(6,11)]
        ks = [1,5,10,15,20,25,30,35,40,45,50]
   
        sampling_epsilon_values_stochastic = [0.01]
        error_epsilon_values_scaled_threshold = [0.1]

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
                data = self.data_provider.read_freelancer_data_obj()
                config = self.config.copy()
                alg.create_sample(config, data, num_sampled_skills, rare_sample_fraction, popular_sample_fraction, 
                                    rare_threshold,popular_threshold, user_sample_ratio, seed)


                self.logger.info("Scaling factor for submodular function is: {}".format(scaling_factor))

                # Distorted Greedy
                for k in ks:
                    # Run algorithm
                    start = timer()
                    result = alg.run(self.config, data, "distorted_greedy",
                         None, None, scaling_factor, num_sampled_skills,
                         rare_sample_fraction, popular_sample_fraction, rare_threshold, popular_threshold,
                         user_sample_ratio, seed, k)
                    end = timer()
                    result['runtime'] = end - start
                    results.append(result)
                    self.logger.info("Algorithm: {} and k: {} and runtime: {}".format("distorted_greedy",k,end - start))

                self.logger.info("\n")

                # Stochastic Distorted Greedy
                for k in ks:
                    for sample_epsilon in sampling_epsilon_values_stochastic:
                        # Run algorithm
                        start = timer()
                        result = alg.run(config, data, "stochastic_distorted_greedy",
                             sample_epsilon, None, scaling_factor, num_sampled_skills,
                             rare_sample_fraction, popular_sample_fraction, rare_threshold, popular_threshold,
                             user_sample_ratio, seed, k) 
                        end = timer()
                        result['runtime'] = end - start
                        results.append(result)
                        self.logger.info("Algorithm: {} and k: {} and runtime: {}".format("stochastic_distorted_greedy",k,end - start))

                self.logger.info("\n")

                # Cost Scaled Greedy              
                for k in ks:
                    # Run algorithm
                    start = timer()
                    result = alg.run(self.config, data, "cost_scaled_greedy",
                         None, None, scaling_factor, num_sampled_skills,
                         rare_sample_fraction, popular_sample_fraction, rare_threshold, popular_threshold,
                         user_sample_ratio, seed, k)
                    end = timer()
                    result['runtime'] = end - start
                    results.append(result)
                    self.logger.info("Algorithm: {} and k: {} and runtime: {}".format("cost_scaled_greedy",k,end - start))

                self.logger.info("\n")

                # Cost scaled lazy exact greedy
                for k in ks:
                    # Run algorithm
                    start = timer()
                    result = alg.run(self.config, data, "cost_scaled_lazy_exact_greedy",
                         None, None, scaling_factor, num_sampled_skills,
                         rare_sample_fraction, popular_sample_fraction, rare_threshold, popular_threshold,
                         user_sample_ratio, seed, k)
                    end = timer()
                    result['runtime'] = end - start
                    results.append(result)
                    self.logger.info("Algorithm: {} and k: {} and runtime: {}".format("cost_scaled_lazy_exact_greedy",k,end - start))

                self.logger.info("\n")

                # Scaled Single Threshold Greedy
                for k in ks:
                    for error_epsilon in error_epsilon_values_scaled_threshold:
                        # Run algorithm
                        start = timer()
                        result = alg.run(self.config, data, "scaled_single_threshold_greedy",
                             None, error_epsilon, scaling_factor, num_sampled_skills,
                             rare_sample_fraction, popular_sample_fraction, rare_threshold, popular_threshold,
                             user_sample_ratio, seed, k)
                        end = timer()
                        result['runtime'] = end - start
                        results.append(result)
                        self.logger.info("Algorithm: {} and k: {} and runtime: {}".format("scaled_single_threshold_greedy",k,end - start))

                self.logger.info("\n")

        self.logger.info("Finished experiment 02")

        # Export results
        df = pd.DataFrame(results)
        self.data_exporter.export_csv_file(df, "experiment_02_freelancer.csv")
        self.logger.info("Exported experiment_02 results")
