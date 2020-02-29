"""
This class implements experiment_00_parallel

Experiment 00 involves testing all algorithms parallely for various values of algorithm specific parameters
"""

import logging
import numpy as np
import pandas as pd
from pathos.pools import ProcessPool
import multiprocessing as mp
from data.data_provider import DataProvider
from data.data_exporter import DataExporter
from algorithms.algorithm_driver import AlgorithmDriver
from algorithms.set_cover_greedy import SetCoverGreedy


class Experiment00P(object):
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
    def run_algorithm(arg):
        # Run algorithm
        alg = AlgorithmDriver()
        data = alg.run(*arg)
        return data

    def run(self):
        """
        Run experiment
        :param:
        :return:
        """
        self.logger.info("Starting experiment 00 Parallel")

        self.expt_config = self.config['experiment_configs']['experiment_00']
        num_sampled_skills = self.expt_config['num_sampled_skills']
        popular_threshold = self.expt_config['popular_threshold']
        rare_threshold = self.expt_config['rare_threshold']

        user_sample_ratios = [0.01, 0.1, 0.3]
        rare_sample_fraction = 0.1
        popular_sample_fraction = 0.1
        lazy_eval_epsilon_values = [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]
        sampling_epsilon_values = [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]

        # Load dataset
        data = self.data_provider.read_guru_data_obj()

        # Create different configurations for algorithms
        args = []
        for user_sample_ratio in user_sample_ratios:
            np.random.seed(seed=0)

            # Create controlled samples dataset
            data.sample_skills_to_be_covered_controlled(num_sampled_skills, rare_sample_fraction,
                                                        popular_sample_fraction, rare_threshold,
                                                        popular_threshold, user_sample_ratio)

            # Find appropriate scaling factor
            alg = SetCoverGreedy(self.config, data.submodular_func, data.E)
            sol = alg.run()
            submodular_val = data.submodular_func(sol)
            cost = data.cost_func(sol)
            scaling_factor = data.scaling_func(submodular_val, cost)

            # Update scaling factor
            data.scaling_factor = scaling_factor
            self.logger.info("Scaling factor for submodular function is: {}".format(data.scaling_factor))

            # Cost distorted greedy
            args.append((self.config, data, "cost_distorted_greedy",
                         None, None, scaling_factor, num_sampled_skills,
                         rare_sample_fraction, popular_sample_fraction, rare_threshold, popular_threshold,
                         user_sample_ratio))

            # Cost scaled greedy
            args.append((self.config, data, "cost_scaled_greedy",
                         None, None, scaling_factor, num_sampled_skills,
                         rare_sample_fraction, popular_sample_fraction, rare_threshold, popular_threshold,
                         user_sample_ratio))

            # Distorted greedy
            args.append((self.config, data, "distorted_greedy",
                         None, None, scaling_factor, num_sampled_skills,
                         rare_sample_fraction, popular_sample_fraction, rare_threshold, popular_threshold,
                         user_sample_ratio))

            # Unconstrained distorted greedy
            for i in range(100):
                args.append((self.config, data, "unconstrained_distorted_greedy",
                             None, None, scaling_factor, num_sampled_skills,
                             rare_sample_fraction, popular_sample_fraction, rare_threshold, popular_threshold,
                             user_sample_ratio))

            # LAZY EVALUATION ALGORITHMS

            # Cost scaled lazy greedy and Distorted lazy greedy
            for lazy_epsilon in lazy_eval_epsilon_values:
                config = self.config.copy()
                config['algorithms']['cost_scaled_lazy_greedy_config']['epsilon'] = lazy_epsilon
                args.append(
                    (self.config, data, "cost_scaled_lazy_greedy",
                     None, lazy_epsilon, scaling_factor, num_sampled_skills,
                     rare_sample_fraction, popular_sample_fraction, rare_threshold, popular_threshold,
                     user_sample_ratio)
                )

                config = self.config.copy()
                config['algorithms']['distorted_lazy_greedy_config']['epsilon'] = lazy_epsilon
                args.append(
                    (self.config, data, "distorted_lazy_greedy",
                     None, lazy_epsilon, scaling_factor, num_sampled_skills,
                     rare_sample_fraction, popular_sample_fraction, rare_threshold, popular_threshold,
                     user_sample_ratio)
                )

            # SAMPLING BASED ALGORITHMS

            # Stochastic distorted greedy
            for sample_epsilon in sampling_epsilon_values:
                for i in range(50):
                    config = self.config.copy()
                    config['algorithms']['stochastic_distorted_greedy_config']['epsilon'] = sample_epsilon
                    args.append(
                        (self.config, data, "stochastic_distorted_greedy",
                         sample_epsilon, None, scaling_factor, num_sampled_skills,
                         rare_sample_fraction, popular_sample_fraction, rare_threshold, popular_threshold,
                         user_sample_ratio)
                    )

        # Create a pool of processes
        num_processes = mp.cpu_count()
        self.logger.info("Processes: {}".format(num_processes))
        pool = ProcessPool(nodes=num_processes)

        # Run the algorithms
        results = pool.amap(self.run_algorithm, args).get()
        pool.terminate()
        pool.join()
        pool.clear()

        self.logger.info("Finished experiment 00 Parallel")

        # Export results
        df = pd.DataFrame(results)
        self.data_exporter.export_csv_file(df, "experiment_00_parallel.csv")
        self.logger.info("Exported experiment_00_parallel results")
