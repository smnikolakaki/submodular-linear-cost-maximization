"""
This class contains methods related to the guru dataset
"""

import logging
import numpy as np
import pandas as pd
import sys
# from numba import jit
# from numba.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings

# Suppress Numba deprecation warnings
# warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
# warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)


class GuruData(object):
    """
    This class contains methods related to the guru dataset
    """
    def __init__(self, config, user_df, skill_df, users, scaling_factor):
        """
        Constructor
        :param config:
        :param user_df:
        :param skill_df:
        :param users:
        :return:
        """
        self.config = config
        self.logger = logging.getLogger("so_logger")
        self.num_skills = len(skill_df)
        self.num_users = len(user_df)
        self.skill_df = skill_df
        self.user_df = user_df
        self.users = users
        self.scaling_factor = scaling_factor

        # Create numba - useable data
        self.skills_matrix = np.array([x['skills_array'] for x in self.users])
        self.cost_vector = np.array([x['cost'] for x in self.users])

    def create_samples(self, skills_sample_fraction=1.0, users_sample_fraction=1.0):
        """
        Samples skills and users
        :param skills_sample_fraction:
        :param users_sample_fraction:
        :return:
        """
        # Sampling
        self.sample_skills_to_be_covered(skills_sample_fraction)
        self.sample_users(users_sample_fraction)

    def sample_skills_to_be_covered(self, fraction=1.0):
        """
        Samples a fraction of skills that need to be covered
        instead of all the skills based on the sampling scheme.
        Note: This is equivalent to marking the unsampled skills
        as covered
        :param fraction:
        :return:
        """
        self.skills_covered = np.zeros(self.num_skills)
        if fraction < 1.0:
            num_sampled_skills = int(fraction * self.num_skills)
            sampled_skills = np.random.choice(self.num_skills, size=num_sampled_skills, replace=False)

            for skill_id in range(self.num_skills):
                if skill_id not in sampled_skills:
                    self.skills_covered[skill_id] = 1  # Mark unsampled skills as already covered

        self.skills_covered = self.skills_covered.astype(bool)

    def sample_users(self, fraction=1.0):
        """
        Samples users instead of using all users to cover the skills
        :param fraction:
        :return:
        """
        if fraction < 1.0:
            num_sampled_users = int(fraction * self.num_users)
            sampled_users = np.random.choice(self.num_users, size=num_sampled_users, replace=False)
            self.E = set(sampled_users)
        else:
            self.E = set(np.arange(self.num_users))

    def categorize_skills(self, df_sampled_users, rare_threshold=0.33, popular_threshold=0.33):
        """
        Categorizes skills of sampled users into three categories based on frequency histogram
        1. rare skills (e.g., bottom 33% frequencies)
        2. common skills (rest of the skills)
        3. popular skills (e.g., top 33% frequencies)

        :param df_sampled_users:
        :param rare_upper_threshold:
        :param popular_lower_threshold:
        """
        # Get frequency of each skills
        skills_array = np.array(df_sampled_users['skills_array'].values)
        freq = np.sum(skills_array, axis=0)
        freq_skills_available = freq[freq > 0]
        num_skills_available = freq_skills_available.shape[0]

        # Get indices of ascending order sorted frequencies
        sorted_idx = np.argsort(freq_skills_available)

        rare_threshold_idx = int(num_skills_available * rare_threshold)
        popular_threshold_idx = int(num_skills_available * (1 - popular_threshold))

        # Split the sampled skills into categories using frequencies
        rare_skills = sorted_idx[:rare_threshold_idx]
        common_skills = sorted_idx[rare_threshold_idx: popular_threshold_idx]
        popular_skills = sorted_idx[popular_threshold_idx:]

        return (rare_skills, common_skills, popular_skills)

    def sample_skills_to_be_covered_controlled(self, num_sampled_skills=50, rare_sample_fraction=0.33,
                                               popular_sample_fraction=0.33, rare_threshold=0.33,
                                               popular_threshold=0.33, user_sample_fraction=1.0):
        """
        Creates a sample of skills of size 'num_skills'. In this sample, 'rare_sample_fraction' of them
        are from rare skills category, 'popular_sample_fraction' of them are from popular skills
        category.

        :param num_sampled_skills:
        :param rare_sample_fraction:
        :param popular_sample_fraction:
        :param rare_threshold:
        :param popular_threshold:
        :param user_sample_fraction:
        """

        self.sample_users(user_sample_fraction)

        df_users = pd.DataFrame(self.users)
        df_users_sampled = df_users[df_users['user_id'].isin(self.E)]

        # Get categorized skills
        r, c, p = self.categorize_skills(df_users_sampled, rare_threshold, popular_threshold)

        # Sample skills from each category
        num_rare_skills = int(num_sampled_skills * rare_sample_fraction)
        num_popular_skills = int(num_sampled_skills * popular_sample_fraction)
        num_common_skills = num_sampled_skills - num_rare_skills - num_popular_skills

        # Ensure that skills to be sampled in each category is >= number of skills in that category
        if num_rare_skills > len(r):
            num_rare_skills = len(r)
        if num_common_skills > len(c):
            num_common_skills = len(c)
        if num_common_skills < 0:
            num_common_skills = 0
        if num_popular_skills > len(p):
            num_popular_skills = len(p)

        sampled_rare_skills = np.random.choice(r, size=num_rare_skills, replace=False)
        sampled_common_skills = np.random.choice(c, size=num_common_skills, replace=False)
        sampled_popular_skills = np.random.choice(p, size=num_popular_skills, replace=False)

        # Merge indices of all sampled skills
        sampled_skills = np.concatenate((sampled_rare_skills, sampled_common_skills, sampled_popular_skills))

        # Create final skills sample
        self.skills_covered = np.zeros(self.num_skills)
        
        for skill_id in range(self.num_skills):
            if skill_id not in sampled_skills:
                self.skills_covered[skill_id] = 1  # Mark unsampled skills as already covered

        self.skills_covered = self.skills_covered.astype(bool)
        self.num_rare_skills = num_rare_skills
        self.num_common_skills = num_common_skills
        self.num_popular_skills = num_popular_skills

    @staticmethod
    # @jit(nopython=True)
    def submodular_func_jit(sol, skills_covered, skills_matrix):
        """
        Submodular function
        :param sol -- a pythons set of user_ids:
        :param skills_covered:
        :param skills_matrix:
        :return val -- number of covered skills:
        """
        skills_covered_during_sampling = len(np.nonzero(skills_covered)[0])
        for user_id in sol:
            skills_covered = np.logical_or(skills_covered, skills_matrix[user_id])
        val = len(np.nonzero(skills_covered)[0])
        if skills_covered_during_sampling > 0:
            val -= skills_covered_during_sampling
        return val

    def submodular_func(self, sol):
        """
        Submodular function
        :param sol -- a python set of user_ids:
        :param cache_val:
        :param use_cached_val:
        :return val -- number of covered skills:
        """
        if len(sol) == 0:
            return 0
        skills_covered = self.skills_covered
        val = self.submodular_func_jit(sol, skills_covered, self.skills_matrix)
        return val * self.scaling_factor

    def init_submodular_func_coverage_caching(self):
        skills_covered = self.skills_covered
        return skills_covered

    def submodular_func_caching_jit(self, skills_covered, user_id, skills_matrix):
        """
        Submodular function
        :param sol -- a pythons set of user_ids:
        :param skills_covered:
        :param skills_matrix:
        :return val -- number of covered skills:
        """
        skills_covered_during_sampling = len(np.nonzero(skills_covered)[0])
        if user_id:
            # print('Skills covered before:',skills_covered)
            skills_covered = np.logical_or(skills_covered, skills_matrix[user_id])
            # print('Skills covered are after:',skills_covered)
        val = len(np.nonzero(skills_covered)[0])
        # print('Skills coverd during sampling:',skills_covered_during_sampling,'val:',val,'skills covered:',skills_covered,'user id',user_id)
        if skills_covered_during_sampling > 0:
            val -= skills_covered_during_sampling

        return val, skills_covered

    def submodular_func_caching(self, skills_covered, user_id):
        """
        Submodular function
        :param sol -- a python set of user_ids:
        :param cache_val:
        :param use_cached_val:
        :return val -- number of covered skills:
        """
        val, skills_covered = self.submodular_func_caching_jit(skills_covered, user_id, self.skills_matrix)
        # print('Main Indices with elements equal to zero:',np.where(skills_covered == 0)[0],'Number of indices:',len(np.where(skills_covered == 0)[0]))
        return val * self.scaling_factor, skills_covered

    def cost_func(self, sol):
        """
        Cost function
        :param sol -- a python set of user_ids:
        :return cost -- cost in dollars:
        """
        cost = 0
        for user_id in sol:
            cost += self.cost_vector[user_id]
        return cost

    def scaling_func(self, sol_value, cost):
        """
        Scaling factor function
        :param sol_size:
        :param cost:
        :return scaling_factor:
        """
        scaling_factor = cost / sol_value
        if scaling_factor <= 0:
            scaling_factor = 1
        return scaling_factor
