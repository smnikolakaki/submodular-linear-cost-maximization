"""
This class contains methods related to the freelancer dataset
"""

import logging
import numpy as np
from numba import jit
from numba.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings
# Suppress Numba deprecation warnings
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)


class FreelancerData(object):
    """
    This class contains methods related to the freelancer dataset
    """
    def __init__(self, config, user_df, skill_df, users):
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
        instead of all the skills.
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

    @staticmethod
    @jit(nopython=True)
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
        return val * 1

    def submodular_func_old(self, sol):
        """
        Submodular function
        :param sol -- a python set of user_ids:
        :return val -- number of covered skills:
        """
        skills_covered = self.skills_covered

        for user_id in sol:
            skills_covered = np.logical_or(skills_covered, self.users[user_id]['skills_array'])

        val = len(np.nonzero(skills_covered)[0])

        # If we sampled skills to be covered, then subtract the skills that
        # we already marked covered during sampling
        skills_covered_during_sampling = len(np.nonzero(self.skills_covered)[0])
        if skills_covered_during_sampling > 0:
            val -= skills_covered_during_sampling

        return 1*val

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
