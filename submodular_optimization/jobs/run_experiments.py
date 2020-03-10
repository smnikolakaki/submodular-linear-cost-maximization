"""
This class runs the experiments
"""
import logging
from experiments.experiment_00 import Experiment00
from experiments.experiment_01 import Experiment01
from experiments.experiment_02 import Experiment02
from experiments.experiment_03 import Experiment03
from experiments.experiment_test import ExperimentTest
from experiments.experiment_00_parallel import Experiment00P
from experiments.experiment_01_parallel import Experiment01P

class ExperimentDriver(object):
    """
    Creates experiment driver
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
        Run experiments
        :param:
        :return:
        """
        self.logger.info("Starting experiments")

        # Get experiment list
        expt_list = self.config['experiments']['expt_list']

        # Run experiments
        for expt_name in expt_list:
            if expt_name == "experiment_00":
                expt = Experiment00(self.config)
                expt.run()
            if expt_name == "experiment_01":
                expt = Experiment01(self.config)
                expt.run()
            if expt_name == "experiment_02":
                expt = Experiment02(self.config)
                expt.run()
            if expt_name == "experiment_03":
                expt = Experiment03(self.config)
                expt.run()
            if expt_name == "experiment_test":
                expt = ExperimentTest(self.config)
                expt.run()
            if expt_name == "experiment_00_parallel":
                expt = Experiment00P(self.config)
                expt.run()
            if expt_name == "experiment_01_parallel":
                expt = Experiment01P(self.config)
                expt.run()

        self.logger.info("Finished experiments")
