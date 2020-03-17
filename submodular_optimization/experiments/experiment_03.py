"""
This class implements experiment_03

"""

import logging
import pandas as pd
import numpy as np
from data.data_provider import DataProvider
from data.data_exporter import DataExporter
from algorithms.algorithm_driver import AlgorithmDriver
from algorithms.set_cover_greedy import SetCoverGreedy
from timeit import default_timer as timer

class Experiment03(object):
    """
    Experiment03 class
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
        self.logger.info("Starting experiment 03")

        self.expt_config = self.config['experiment_configs']['experiment_03']
        popular_threshold = self.expt_config['popular_threshold']
        rare_threshold = self.expt_config['rare_threshold']

        user_sample_ratios = [0.05,0.1]
        seeds = [i for i in range(6,11)]

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

                # Load dataset
                data = self.data_provider.read_freelancer_data_obj()
                config = self.config.copy()
                alg.create_sample(config, data, num_sampled_skills, rare_sample_fraction, popular_sample_fraction, 
                                    rare_threshold,popular_threshold, user_sample_ratio, seed)

                self.logger.info("Experiment for user sample ratio: {} and scaling factor: {} and seed: {} and number of elements: {}".format(user_sample_ratio,scaling_factor,seed,len(data.E)))
                self.logger.info("Scaling factor for submodular function is: {}".format(scaling_factor))

                # Total number of elements
                n = len(data.E)

                # Distorted Greedy
                total_runtime = 0
                for k in range(1,n+1):
                    # Run algorithm
                    start = timer()
                    result = alg.run(self.config, data, "distorted_greedy",
                         None, None, scaling_factor, num_sampled_skills,
                         rare_sample_fraction, popular_sample_fraction, rare_threshold, popular_threshold,
                         user_sample_ratio, seed, k)
                    end = timer()
                    print('Previous runtime:',total_runtime,'new runtime:',end - start)
                    total_runtime += end - start
                    result['runtime'] = total_runtime
                    results.append(result)
                    self.logger.info("Algorithm: {} and k: {} and runtime: {}".format("distorted_greedy",k,total_runtime))

                self.logger.info("\n")

                # Stochastic Distorted Greedy
                total_runtime = 0
                for k in range(1,n+1):
                    for sample_epsilon in sampling_epsilon_values_stochastic:
                        # Run algorithm
                        start = timer()
                        result = alg.run(config, data, "stochastic_distorted_greedy",
                             sample_epsilon, None, scaling_factor, num_sampled_skills,
                             rare_sample_fraction, popular_sample_fraction, rare_threshold, popular_threshold,
                             user_sample_ratio, seed, k) 
                        end = timer()
                        total_runtime += end - start
                        result['runtime'] = total_runtime
                        results.append(result)
                        self.logger.info("Algorithm: {} and k: {} and runtime: {}".format("stochastic_distorted_greedy",k,total_runtime))

                self.logger.info("\n")

                # Cost Scaled Greedy   
                # Run algorithm that creates greedy ordering
                start = timer()
                result = alg.run(self.config, data, "cost_scaled_greedy",
                     None, None, scaling_factor, num_sampled_skills,
                     rare_sample_fraction, popular_sample_fraction, rare_threshold, popular_threshold,
                     user_sample_ratio, seed, n)
                end = timer()
                result['runtime'] = end - start
                # For each individual k we find the prefix of size k and find the corresponding solution
                for k in range(1,n+1):
                    result_k = result.copy()
                    if k < len(result['sol']):
                        sol_k = set(list(result['sol'])[:k])
                        submodular_val_k = data.submodular_func(sol_k)
                        cost_k = data.cost_func(sol_k)
                        val_k = submodular_val_k - cost_k
                        result_k['sol'] = sol_k; result_k['val'] = val_k; result_k['submodular_val'] = submodular_val_k;
                        result_k['cost'] = cost_k;
                    else:
                        sol_k = result['sol']
                        val_k = result['val']
                    result_k['k'] = k;
                    results.append(result_k)
                    self.logger.info("Best solution: {}\nBest value: {}".format(sol_k, val_k))
                    self.logger.info("Algorithm: {} and k: {} and runtime: {}".format("cost_scaled_greedy",k,end - start))

                self.logger.info("\n")

                # Cost scaled lazy exact greedy
                # Run algorithm that creates greedy ordering
                start = timer()
                result = alg.run(self.config, data, "cost_scaled_lazy_exact_greedy",
                     None, None, scaling_factor, num_sampled_skills,
                     rare_sample_fraction, popular_sample_fraction, rare_threshold, popular_threshold,
                     user_sample_ratio, seed, n)
                end = timer()
                result['runtime'] = end - start
                # For each individual k we find the prefix of size k and find the corresponding solution
                for k in range(1,n+1):
                    result_k = result.copy()
                    if k < len(result['sol']):
                        sol_k = set(list(result['sol'])[:k])
                        submodular_val_k = data.submodular_func(sol_k)
                        cost_k = data.cost_func(sol_k)
                        val_k = submodular_val_k - cost_k
                        result_k['sol'] = sol_k; result_k['val'] = val_k; result_k['submodular_val'] = submodular_val_k;
                        result_k['cost'] = cost_k;
                    else:
                        sol_k = result['sol']
                        val_k = result['val']
                    result_k['k'] = k;
                    results.append(result_k)
                    self.logger.info("Best solution: {}\nBest value: {}".format(sol_k, val_k))
                    self.logger.info("Algorithm: {} and k: {} and runtime: {}".format("cost_scaled_lazy_exact_greedy",k,end - start))

                self.logger.info("\n")

                # Scaled Single Threshold Greedy
                total_runtime = 0
                for k in range(1,n+1):
                    for error_epsilon in error_epsilon_values_scaled_threshold:
                        # Run algorithm
                        start = timer()
                        result = alg.run(self.config, data, "scaled_single_threshold_greedy",
                             None, error_epsilon, scaling_factor, num_sampled_skills,
                             rare_sample_fraction, popular_sample_fraction, rare_threshold, popular_threshold,
                             user_sample_ratio, seed, k)
                        end = timer()
                        total_runtime += end - start
                        result['runtime'] = total_runtime
                        results.append(result)
                        self.logger.info("Algorithm: {} and k: {} and runtime: {}".format("scaled_single_threshold_greedy",k,total_runtime))

                self.logger.info("\n")

        self.logger.info("Finished experiment 03")

        # Export results
        df = pd.DataFrame(results)
        self.data_exporter.export_csv_file(df, "experiment_03_freelancer.csv")
        self.logger.info("Exported experiment_03 results")
