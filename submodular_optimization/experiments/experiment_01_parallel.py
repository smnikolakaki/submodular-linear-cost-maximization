"""
This class implements experiment_01_parallel

"""

import logging
import pandas as pd
from pathos.pools import ProcessPool
import multiprocessing as mp
from data.data_provider import DataProvider
from data.data_exporter import DataExporter
from algorithms.algorithm_driver import AlgorithmDriver


class Experiment01P(object):
    """
    Experiment01 class
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
        self.logger.info("Starting experiment 01 Parallel")

        self.expt_config = self.config['experiment_configs']['experiment_01']
        popular_threshold = self.expt_config['popular_threshold']
        rare_threshold = self.expt_config['rare_threshold']

        user_sample_ratios = [0.3]
        num_sampled_skills = 50
        rare_sample_fraction = 0.1
        popular_sample_fraction = 0.1
        scaling_factors = [1, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
        lazy_eval_epsilon_values = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]

        # Load dataset
        data = self.data_provider.read_guru_data_obj()
        args = []
        for user_sample_ratio in user_sample_ratios:
            for scaling_factor in scaling_factors:
                for lazy_epsilon in lazy_eval_epsilon_values:
                    # BASELINE - distorted_greedy
                    args.append(
                        (self.config, data, "distorted_greedy",
                         None, lazy_epsilon, scaling_factor, num_sampled_skills,
                         rare_sample_fraction, popular_sample_fraction, rare_threshold, popular_threshold,
                         user_sample_ratio)
                    )

                    # LAZY EVALUATION ALGORITHMS
                    config = self.config.copy()
                    config['algorithms']['cost_scaled_lazy_greedy_config']['epsilon'] = lazy_epsilon
                    args.append(
                        (config, data, "cost_scaled_lazy_greedy",
                         None, lazy_epsilon, scaling_factor, num_sampled_skills,
                         rare_sample_fraction, popular_sample_fraction, rare_threshold, popular_threshold,
                         user_sample_ratio)
                    )

        self.logger.info("Finished creating combinations")

        # Create a pool of processes
        num_processes = mp.cpu_count()
        self.logger.info("Processes: {}".format(num_processes))
        pool = ProcessPool(nodes=num_processes)

        # Run the algorithms
        results = pool.amap(self.run_algorithm, args).get()
        pool.terminate()
        pool.join()
        pool.clear()

        self.logger.info("Finished experiment 01 Parallel")

        # Export results
        df = pd.DataFrame(results)
        self.data_exporter.export_csv_file(df, "experiment_01_parallel.csv")
        self.logger.info("Exported experiment_01_parallel results")
