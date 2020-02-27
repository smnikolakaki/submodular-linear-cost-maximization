"""
This class implements data provider for reading various datasets
"""
import logging
import yaml
import os
import pandas as pd
import dill


class DataProvider(object):
    """
    This class implements the data provider for reading various data
    """

    def __init__(self, config):
        """
        Constructor
        :param config:
        :return:
        """
        self.config = config
        self.logger = logging.getLogger("so_logger")

    @classmethod
    def read_config(cls, config_yaml_path):
        """
        Reads config file
        :param config_yaml_path:
        :return config:
        """
        with open(config_yaml_path) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        return config

    def read_guru_user_data(self):
        """
        Reads user info from guru dataset
        :param:
        :return df:
        """
        DATA_DIR = self.config['project']['DATA_DIR']
        filename = os.path.join(
            DATA_DIR,
            "raw_data",
            "all",
            "Original_Data",
            "Guru",
            "guruUserInfo.txt")
        df = pd.read_csv(
            filename,
            header=None,
            index_col=False,
            sep="\t")
        return df

    def read_guru_skill_data(self):
        """
        Reads skills from guru dataset
        :param:
        :return df:
        """
        DATA_DIR = self.config['project']['DATA_DIR']
        filename = os.path.join(
            DATA_DIR,
            "raw_data",
            "all",
            "Processed_Data",
            "Guru",
            "Guru__skillId_skillName_numJobs_numWorkers.tsv")
        df = pd.read_csv(
            filename,
            header=None,
            index_col=False,
            sep="\t")
        return df

    def read_guru_data_obj(self):
        """
        Reads guru data
        :param:
        :return guru_obj:
        """
        DATA_DIR = self.config['project']['DATA_DIR']
        filepath = os.path.join(DATA_DIR, "guru", "guru_data.dill")
        with open(filepath, 'rb') as f:
            guru_obj = dill.load(f)
        # self.logger.info("Guru data object read")
        return guru_obj

    def read_freelancer_user_data(self):
        """
        Reads user info from freelancer dataset
        :param:
        :return df:
        """
        DATA_DIR = self.config['project']['DATA_DIR']
        filename = os.path.join(
            DATA_DIR,
            "raw_data",
            "all",
            "Original_Data",
            "Freelancer",
            "freelancer_profiles.txt")
        df = pd.read_csv(
            filename,
            header=None,
            index_col=False,
            sep="\t")
        return df

    def read_freelancer_skill_data(self):
        """
        Reads skills from freelancer dataset
        :param:
        :return df:
        """
        DATA_DIR = self.config['project']['DATA_DIR']
        filename = os.path.join(
            DATA_DIR,
            "raw_data",
            "all",
            "Processed_Data",
            "Freelancer",
            "Freelance__skillId_skillName_numJobs_numWorkers.tsv")
        df = pd.read_csv(
            filename,
            header=None,
            index_col=False,
            sep="\t")
        return df

    def read_freelancer_data_obj(self):
        """
        Reads freelancer data
        :param:
        :return freelancer_obj:
        """
        DATA_DIR = self.config['project']['DATA_DIR']
        filepath = os.path.join(DATA_DIR, "freelancer", "freelancer_data.dill")
        with open(filepath, 'rb') as f:
            freelancer_obj = dill.load(f)
        self.logger.info("Freelancer data object read")
        return freelancer_obj
