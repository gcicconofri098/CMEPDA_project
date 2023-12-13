''' Contains the global parameters for the package (hyperparameters for the GNN are listed in another file).

    Args:

    :param int n_hits: number of hits per event.
    :param int log_value: level of logging.
    :param bool graph_drawer: flag that enables the graph drawer function.
    :param bool debug_value: flag that defines whether to use the debugging function or not.
    :param bool use_sliced_tensor: flag that defines whether to use a sliced tensor instead of the whole file.
'''

import logging


n_hits = 20

log_value = logging.INFO
graph_drawer = False
debug_value = False
use_sliced_tensor = False