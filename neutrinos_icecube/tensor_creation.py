""" Creates the torch_geometric.Data for handling the graphs.

Returns:
    list of torch_geometric.Data: List of Data, one for each event
"""

import pandas as pd
import torch
from torch_cluster import knn_graph
from torch_geometric.data import Data

import parameters
import hyperparameters
from graph_visualizer import graph_visualisation
from logging_conf import setup_logging

logger = setup_logging('tensor_creation')

def tensor_creator(df, targets, **kwargs):
    """ Takes the pandas Dataframes and creates the torch Tensor for the GNN
    Args:
        df (pandas.DataFrame): dataframe with the features
        targets (pandas.DataFrame): dataframe with the targets
        label (str): string that differentiates training and test samples for debugging
    Returns:
        data_list (list of torch_geometric.Data): a  list that contains Data 
    """
    
    if 'label' in kwargs: #takes into account whether there is a label in the function call 
        label = kwargs.get('label')

    #takes the number of the events present in the pandas dataframe
    unique_events = pd.unique(df.index.get_level_values(0))

    sliced_unique_events = unique_events[:1000] #subset of events for debugging
    logger.debug(f"sliced events: {sliced_unique_events}")

    data_list = []

    #loops on the events IDs, creates a Data object for each event(a graph for each event)

    if parameters.use_sliced_tensor:
        events = sliced_unique_events
    else:
        events = unique_events

    for idx, event_id in enumerate(events):
        # Extract hits for the current event
        event_data = df[df.index.get_level_values(0) == event_id].copy()
        event_targets = targets[targets["event_id"] == event_id].copy()

        #removes the padding needed for unstacking
        
        nan_value = float("NaN")
        event_data.replace(0.0, nan_value, inplace=True)
        event_data.dropna(how="all", axis=1, inplace=True)
 
        # extract node features and targets
        node_features = event_data[["charge", "x", "y", "z", "time"]]
        node_targets = event_targets[["azimuth", "zenith"]]

        #creates the Data object

        logger.debug("print shape of x and y to be put in Data")
        logger.debug(torch.Tensor(node_features.values.reshape(5, -1).T).shape)
        logger.debug(torch.Tensor(node_targets.values).reshape(-1, 2).shape)

        data = Data(
            x=torch.Tensor(node_features.values.reshape(5, -1).T),
            y=torch.Tensor(node_targets.values).reshape(-1, 2),
        )
        # Add the Data object to the list
        data_list.append(data)
        if(idx % 500 == 0):
            print(f"processing event number: {idx}")
        logger.debug(f"Node Features Shape: {data.x.shape}")
        logger.debug(f"Node Targets Shape: {data.y.shape}")

        #creates the edges for each node considering the KNN neighbors
        data.edge_index = knn_graph(data.x, k=hyperparameters.n_neighbors, loop=False)

        #prints some graphs if wanted
        if parameters.use_sliced_tensor and parameters.graph_drawer:
            graph_visualisation(data, event_id, label)

        # creates an additional feature that takes into account the sum of charges for each cluster
        cluster_charge = torch.zeros(
            data.x.size(0), dtype=data.x.dtype, device=data.x.device
        )

        for i in range(data.edge_index.size(1)):
            idx_0 = data.edge_index[0, i]
            idx_1 = data.edge_index[1, i]
            charge_value = data.x[idx_1, 0]

            cluster_charge[idx_0] += charge_value

        #adds the additional feature to the node, to make more relevant the nodes inside clusters with large charge
        data.x = torch.cat([data.x, cluster_charge.view(-1, 1)], dim=-1)

    return data_list