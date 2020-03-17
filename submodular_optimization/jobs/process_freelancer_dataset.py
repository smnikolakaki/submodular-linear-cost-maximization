"""
This job processes guru dataset and exports the processed data object
"""

import logging
import pandas as pd
from data.data_provider import DataProvider
from data.data_exporter import DataExporter
from sklearn.preprocessing import MultiLabelBinarizer
from freelancer.freelancer_dataset import FreelancerData


class FreelancerDataProcessor(object):
    """
    This job processes freelancer dataset and exports the processed data
    """
    def __init__(self, config):
        """
        Constructor
        :param config:
        :return:
        """
        self.config = config
        self.logger = logging.getLogger("so_logger")

    def run(self):
        """
        Execute the job
        :param:
        :return:
        """
        self.logger.info("Starting job: FreelancerDataProcessor\n")
        data_provider = DataProvider(self.config)
        data_exporter = DataExporter(self.config)

        # Read freelancer data
        df = data_provider.read_freelancer_user_data()
        df_cost = df[[1]]  # Salary/Hour
        df_skills = df[df.columns[4::2]]
        df_skills.replace(to_replace=["Other Skills"], value="", inplace=True)
        df_skills = (df_skills.iloc[:, 0].map(str)
                     + ',' + df_skills.iloc[:, 1].map(str)
                     + ',' + df_skills.iloc[:, 2].map(str)
                     + ',' + df_skills.iloc[:, 3].map(str)
                     + ',' + df_skills.iloc[:, 4].map(str)
                     + ',' + df_skills.iloc[:, 5].map(str)
                     )  # Skills

        user_df = pd.DataFrame()
        user_df['cost'] = df_cost.iloc[:, 0].tolist()
        # Converting all strings to lower case
        user_df['skills'] = df_skills.str.lower().tolist()

        user_df = user_df.reset_index(drop=True)
        user_df = user_df.assign(user_id=user_df.index.values)
        user_df = user_df.assign(skills=user_df.apply(lambda x: x['skills'][:-1].split(','), axis=1))

        # Convert cost to integers
        user_df.cost = user_df.cost.astype(int)

        # Read skills data
        df = data_provider.read_freelancer_skill_data()
        df = df[[1]]
        df.columns = ['skill']
        skill_df = df.assign(skill_id=df.index.values)

        # Create multilabel binarizer
        mlb = MultiLabelBinarizer(classes=skill_df.skill.values)

        # One hot encoding of user skills
        skills = mlb.fit_transform(user_df['skills'])

        # Create dataset
        users = user_df.to_dict('records')
        for i in range(len(users)):
            users[i]['skills_array'] = skills[i]

        # Export csv files
        data_exporter.export_csv_file(user_df, "freelancer/freelancer_user_df.csv")
        data_exporter.export_csv_file(skill_df, "freelancer/freelancer_skill_df.csv")

        # Scaling factor for submodular function
        scaling_factor = 1

        # Create and export data object to be used in experiments
        # containing all methods related to freelancer data
        freelancer = FreelancerData(self.config, user_df, skill_df, users, scaling_factor)
        data_exporter.export_dill_file(freelancer, "freelancer/freelancer_data.dill")

        self.logger.info("Finished job: FreelancerDataProcessor")
