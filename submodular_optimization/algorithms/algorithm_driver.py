"""
This class runs an algorithm with given config
"""
import logging
import numpy as np
from timeit import default_timer as timer
from algorithms.cost_distorted_greedy import CostDistortedGreedy
from algorithms.distorted_greedy import DistortedGreedy
from algorithms.cost_scaled_greedy import CostScaledGreedy
from algorithms.unconstrained_linear import UnconstrainedLinear
from algorithms.cost_scaled_lazy_greedy import CostScaledLazyGreedy
from algorithms.cost_scaled_lazy_exact_greedy import CostScaledLazyExactGreedy
from algorithms.stochastic_distorted_greedy import StochasticDistortedGreedy
from algorithms.unconstrained_distorted_greedy import UnconstrainedDistortedGreedy
from algorithms.cost_distorted_lazy_greedy import CostDistortedLazyGreedy
from algorithms.distorted_lazy_greedy import DistortedLazyGreedy
from algorithms.scaled_single_threshold_greedy import ScaledSingleThresholdGreedy
from algorithms.scaled_single_threshold_max_val_greedy import ScaledSingleThresholdMaxValGreedy

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

    def run(self, config, data, algorithm, sample_epsilon, lazy_epsilon, scaling_factor, num_sampled_skills,
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
        """
        np.random.seed(seed=seed)
        data.sample_skills_to_be_covered_controlled(num_sampled_skills, rare_sample_fraction,
                                                    popular_sample_fraction, rare_threshold,
                                                    popular_threshold, user_sample_ratio)
        data.scaling_factor = scaling_factor

        if algorithm == "cost_distorted_greedy":
            alg = CostDistortedGreedy(config, data.submodular_func, data.cost_func, data.E)

        elif algorithm == "distorted_greedy":
            alg = DistortedGreedy(config, data.submodular_func, data.cost_func, data.E, k)

        elif algorithm == "cost_scaled_greedy":
            alg = CostScaledGreedy(config, data.submodular_func, data.cost_func, data.E)

        elif algorithm == "unconstrained_linear":
            alg = UnconstrainedLinear(config, data.submodular_func, data.cost_func, data.E)

        elif algorithm == "cost_scaled_lazy_greedy":
            config['algorithms']['cost_scaled_lazy_greedy_config']['epsilon'] = lazy_epsilon
            alg = CostScaledLazyGreedy(config, data.submodular_func, data.cost_func, data.E)

        elif algorithm == "cost_scaled_lazy_exact_greedy":
            alg = CostScaledLazyExactGreedy(config, data.submodular_func, data.cost_func, data.E)

        elif algorithm == "stochastic_distorted_greedy":
            config['algorithms']['stochastic_distorted_greedy_config']['epsilon'] = sample_epsilon
            alg = StochasticDistortedGreedy(config, data.submodular_func, data.cost_func, data.E, k)

        elif algorithm == "unconstrained_distorted_greedy":
            alg = UnconstrainedDistortedGreedy(config, data.submodular_func, data.cost_func, data.E)

        elif algorithm == "cost_distorted_lazy_greedy":
            config['algorithms']['cost_distorted_lazy_greedy_config']['epsilon'] = lazy_epsilon
            alg = CostDistortedLazyGreedy(config, data.submodular_func, data.cost_func, data.E)

        elif algorithm == "distorted_lazy_greedy":
            config['algorithms']['distorted_lazy_greedy_config']['epsilon'] = lazy_epsilon
            alg = DistortedLazyGreedy(config, data.submodular_func, data.cost_func, data.E)

        elif algorithm == "scaled_single_threshold_greedy":
            config['algorithms']['scaled_single_threshold_greedy_config']['epsilon'] = sample_epsilon
            alg = ScaledSingleThresholdGreedy(config, data.submodular_func, data.cost_func, data.E, k)

        elif algorithm == "scaled_single_threshold_max_val_greedy":
            config['algorithms']['scaled_single_threshold_max_val_greedy_config']['epsilon'] = sample_epsilon
            alg = ScaledSingleThresholdMaxValGreedy(config, data.submodular_func, data.cost_func, data.E, k)
            
        else:
            self.logger.info("Algorithm is not implemented")

        # Run algorithm
        start = timer()
        sol = alg.run()
        end = timer()

        submodular_val = data.submodular_func(sol)
        cost = data.cost_func(sol)
        val = submodular_val - cost
        result = {'alg': algorithm,
                  'sol': sol,
                  'val': val,
                  'submodular_val': submodular_val,
                  'cost': cost,
                  'runtime': end - start,
                  'lazy_epsilon': lazy_epsilon,
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
