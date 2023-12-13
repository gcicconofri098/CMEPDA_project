''' Contains the global parameters for the package (hyperparameters for the GNN are listed in another file).

    Args:

        n_hits (int): number of hits per event.
        log_value (int): level of logging.
        graph_drawer (bool): flag that enables the graph drawer function.
        debug_value (bool): flag that defines whether to use the debugging function or not.
        use_sliced_tensor (bool): flag that defines whether to use a sliced tensor instead of the whole file.
'''

import logging


n_hits = 20

log_value = logging.INFO
graph_drawer = False
debug_value = False
use_sliced_tensor = False