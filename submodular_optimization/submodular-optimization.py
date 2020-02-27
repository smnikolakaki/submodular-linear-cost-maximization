"""
This class implements driver program for the submodular-optimization project
"""

import logging
import sys
import yaml
from jobs.run_experiments import ExperimentDriver
from jobs.process_guru_dataset import GuruDataProcessor
from jobs.process_freelancer_dataset import FreelancerDataProcessor

class SubmodularOptimization(object):
    """
    This class implements the driver program for the submodular-optimization project
    """

    def __init__(self, config_):
        """
        Constructor
        :param config_:
        :return:
        """
        self.config = config_

        # Initialize the logging
        if self.config['project']['project_logging_level'] == 'DEBUG':
            logging_level = logging.DEBUG
        elif self.config['project']['project_logging_level'] == 'INFO':
            logging_level = logging.INFO
        else:
            logging_level = logging.INFO

        logging.basicConfig(
            format="LOG: %(asctime)-15s:[%(filename)s]: %(message)s",
            datefmt='%m/%d/%Y %I:%M:%S %p')

        self.logger = logging.getLogger("so_logger")
        self.logger.setLevel(logging_level)

    def run(self):
        """
        Execute jobs
        :param:
        :return:
        """
        self.logger.info("Starting project: {}".format(self.config['project']['project_name']))

        # Get job list
        job_list = self.config['jobs']['job_list']

        # Execute jobs
        for job_name in job_list:
            if job_name == "process_guru_data":
                job = GuruDataProcessor(self.config)
                job.run()
            if job_name == "process_freelancer_data":
                job = FreelancerDataProcessor(self.config)
                job.run()
            if job_name == "run_experiments":
                job = ExperimentDriver(self.config)
                job.run()

        self.logger.info("Finished")


if __name__ == "__main__":

    # Load configuration
    if len(sys.argv) > 1:
        config_yaml_path = sys.argv[1]
    else:
        config_yaml_path = "config/submodular-optimization.yaml"

    with open(config_yaml_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Start driver
    driver = SubmodularOptimization(config)
    driver.run()
