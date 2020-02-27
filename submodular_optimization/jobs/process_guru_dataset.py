"""
This job processes guru dataset and exports the processed data object
"""

import logging
from data.data_provider import DataProvider
from data.data_exporter import DataExporter
from sklearn.preprocessing import MultiLabelBinarizer
from guru.guru_dataset import GuruData


class GuruDataProcessor(object):
    """
    This job processes guru dataset and exports the processed data
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
        self.logger.info("Starting job: GuruDataProcessor\n")
        data_provider = DataProvider(self.config)
        data_exporter = DataExporter(self.config)

        # Read guru data
        df = data_provider.read_guru_user_data()
        df = df[[3, 4]]  # Salary and skills columns
        df.columns = ['cost', 'skills']
        df = df[(df.cost != "$0") & (df.skills != "UNKNOWN")]
        df = df.reset_index(drop=True)
        df = df.assign(user_id=df.index.values)
        df = df.assign(skills=df.apply(lambda x: x['skills'][:-1].split(','), axis=1))

        # Convert cost to integers
        user_df = df.assign(cost=df.apply(lambda x: int(x['cost'][1:]), axis=1))

        # Read skills data
        df = data_provider.read_guru_skill_data()
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
        data_exporter.export_csv_file(user_df, "guru/guru_user_df.csv")
        data_exporter.export_csv_file(skill_df, "guru/guru_skill_df.csv")

        # Scaling factor for submodular function
        scaling_factor = 1

        # Create and export data object to be used in experiments
        # containing all methods related to guru data
        guru = GuruData(self.config, user_df, skill_df, users, scaling_factor)
        data_exporter.export_dill_file(guru, "guru/guru_data.dill")

        self.logger.info("Finished job: GuruDataProcessor")
