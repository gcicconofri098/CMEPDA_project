""" This module handles the logging part of the package.
It sets the log path for all the modules, and the global level of logging.
"""

import logging
import sys
import parameters

LOG_PATH= '/gpfs/ddn/cms/user/cicco/miniconda3/CMEPDA/logs/'

def setup_logging(log_filename):

    """ Sets up the logging settings.
        
    Args:
        log_filename (str): name of the log file for the specific module

    Returns:
        logging: a logging object with the wanted settings
    """

    logging.basicConfig(level= parameters.log_value)
    logger = logging.getLogger(log_filename)
    file_handler=logging.FileHandler(LOG_PATH + log_filename + '.log')

    logger.addHandler(file_handler)
    #logger.addHandler(logging.StreamHandler(sys.stdout))
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)


    return logger