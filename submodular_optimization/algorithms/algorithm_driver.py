"""
This class runs an algorithm with given config
"""
import logging
import numpy as np
from timeit import default_timer as timer
from algorithms.distorted_greedy import DistortedGreedy
from algorithms.cost_scaled_greedy import CostScaledGreedy
from algorithms.unconstrained_linear import UnconstrainedLinear
from algorithms.cost_scaled_lazy_exact_greedy import CostScaledLazyExactGreedy
from algorithms.stochastic_distorted_greedy import StochasticDistortedGreedy
from algorithms.unconstrained_distorted_greedy import UnconstrainedDistortedGreedy
from algorithms.scaled_single_threshold_greedy import ScaledSingleThresholdGreedy

class AlgorithmDriver(object):
    """
    Creates experiment driver
    """

    def __init__(self):
        """
        Constructor
        :param:
        :return:
        """
        self.logger = logging.getLogger("so_logger")

    def create_sample(self, config, data, num_sampled_skills, rare_sample_fraction, popular_sample_fraction, rare_threshold,
                        popular_threshold, user_sample_ratio, seed):
        """
        create the sample
        """
        np.random.seed(seed=seed)
        data.sample_skills_to_be_covered_controlled(num_sampled_skills, rare_sample_fraction,
                                                    popular_sample_fraction, rare_threshold,
                                                    popular_threshold, user_sample_ratio)

    def run(self, config, data, algorithm, sample_epsilon, error_epsilon, scaling_factor, num_sampled_skills,
            rare_sample_fraction, popular_sample_fraction, rare_threshold, popular_threshold, user_sample_ratio, seed, k):
        """run

        :param config:
        :param data:
        :param algorithm:
        :param sample_epsilon:
        :param lazy_epsilon:
        :param scaling_factor:
        :param num_sampled_skills:
        :param rare_sample_fraction:
        :param popular_sample_fraction:
        :param rare_threshold:
        :param popular_threshold:
        :param user_sample_ratio:
        :seed:
        :k:
        """
        data.scaling_factor = scaling_factor

        if algorithm == "distorted_greedy":
            alg = DistortedGreedy(config, data.init_submodular_func_coverage_caching, data.submodular_func_caching, data.cost_func, data.E, k)

        elif algorithm == "cost_scaled_greedy":
            alg = CostScaledGreedy(config, data.init_submodular_func_coverage_caching, data.submodular_func_caching, data.cost_func, data.E, k)

        elif algorithm == "unconstrained_linear":
            alg = UnconstrainedLinear(config, data.init_submodular_func_coverage_caching, data.submodular_func_caching, data.cost_func, data.E)

        elif algorithm == "cost_scaled_lazy_exact_greedy":
            alg = CostScaledLazyExactGreedy(config, data.init_submodular_func_coverage_caching, data.submodular_func_caching, data.cost_func, data.E, k)

        elif algorithm == "unconstrained_distorted_greedy":
            alg = UnconstrainedDistortedGreedy(config, data.init_submodular_func_coverage_caching, data.submodular_func_caching, data.cost_func, data.E)

        elif algorithm == "stochastic_distorted_greedy":
            config['algorithms']['stochastic_distorted_greedy_config']['epsilon'] = sample_epsilon
            alg = StochasticDistortedGreedy(config, data.init_submodular_func_coverage_caching, data.submodular_func_caching, data.cost_func, data.E, k)
            
        elif algorithm == "scaled_single_threshold_greedy":
            config['algorithms']['scaled_single_threshold_greedy_config']['epsilon'] = error_epsilon
            alg = ScaledSingleThresholdGreedy(config, data.init_submodular_func_coverage_caching, data.submodular_func_caching, data.cost_func, data.E, k)
            
        else:
            self.logger.info("Algorithm is not implemented")

        # Run algorithm
        sol = alg.run()

        submodular_val = data.submodular_func(sol)
        cost = data.cost_func(sol)
        val = submodular_val - cost
        result = {'alg': algorithm,
                  'sol': sol,
                  'val': val,
                  'submodular_val': submodular_val,
                  'cost': cost,
                  'runtime': None,
                  'error_epsilon': error_epsilon,
                  'sample_epsilon': sample_epsilon,
                  'user_sample_ratio': user_sample_ratio,
                  'scaling_factor': scaling_factor,
                  'num_rare_skills': data.num_rare_skills,
                  'num_common_skills': data.num_common_skills,
                  'num_popular_skills': data.num_popular_skills,
                  'num_sampled_skills': num_sampled_skills,
                  'seed': seed,
                  'k': k
                  }
        return result
