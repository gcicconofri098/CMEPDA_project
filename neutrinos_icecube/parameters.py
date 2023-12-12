import logging
"""
    Contains the global parameters for the package. 
    Hyperparameters for the GNN are listed in another file    
"""
n_hits = 20 #number of hits per event

log_value = logging.INFO #sets the level of logging
graph_drawer = False #flag that enables the graph drawer function
debug_value = False #flag that defines whether to use the debugging function or not
use_sliced_tensor = False #flag that defines whether to use a sliced tensor instead of the whole file