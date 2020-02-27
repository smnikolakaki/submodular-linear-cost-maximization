"""
This class implements data exporter
"""
import logging
import os
import dill


class DataExporter(object):
    """
    This class implements data exporter for storing various data
    """

    def __init__(self, config):
        """
        Constructor
        :param config:
        :return:
        """
        self.logger = logging.getLogger("so_logger")
        self.config = config

    def export_csv_file(self, df, filename):
        """
        Exports csv file containing the df
        :param df:
        :param filename:
        :return:
        """
        DATA_DIR = self.config['project']['DATA_DIR']
        filepath = os.path.join(DATA_DIR, filename)
        df.to_csv(filepath, sep=',', header=True, index=False)
        self.logger.info("Exported {}".format(filename))

    def export_dill_file(self, data, filename):
        """
        Exports dill file containing the data
        :param data:
        :param filename:
        :return:
        """
        DATA_DIR = self.config['project']['DATA_DIR']
        filepath = os.path.join(DATA_DIR, filename)
        with open(filepath, 'wb') as f:
            dill.dump(data, f)
        self.logger.info("Exported {}".format(filename))
