import pandas as pd
import torch
import torch_geometric
from torch_cluster import knn_graph
from torch_geometric.data import Data

from neutrinos_icecube.graph_visualizer import graph_visualisation


def tensor_creator(df, targets, label):
    """
    Takes the pandas Dataframes and creates the torch Tensor for the GNN
    Args:
        df (pandas Dataframe): dataframe with the features
        targets (pandas Dataframe): dataframe with the targets
        label (str): string that differentiates training and test samples for debugging
    """


    graph_drawer = False #flag that enables the graph visualisation function

    #takes the names of the events 
    unique_events = pd.unique(df.index.get_level_values(0))

    sliced_unique_events = unique_events[:3] #subset of events for debugging
    # print(sliced_unique_events)

    data_list = []

    # def minkowski_distance(x,y):
    #     spatial_distance = torch.norm(x[:3] - y[:3], p=2)  # Minkowski spatial distance
    #     temporal_distance = torch.abs(x[3] - y[3])  # Absolute temporal distance
    #     return (spatial_distance**2 + temporal_distance**2)**(1/2)

    #loops on the events IDs, creates a Data object for each event(a graph for each event)

    for idx, event_id in enumerate(unique_events):
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

        # print("print shape of x and y to be put in Data")
        # print(torch.Tensor(node_features.values.reshape(5, -1).T).shape)
        # print(torch.Tensor(node_targets.values).reshape(-1, 2).shape)

        data = Data(
            x=torch.Tensor(node_features.values.reshape(5, -1).T),
            y=torch.Tensor(node_targets.values).reshape(-1, 2),
        )
        # Add the Data object to the list
        data_list.append(data)
        if(idx % 100 == 0):
            print(f"processin event number: {idx}")
        # print("Node Features Shape:", data.x.shape)
        # print("Node Targets Shape:", data.y.shape)

        n_neighbors = 7 #!number of neighbors for each node 

        #creates the edges for each node considering the KNN neighbors
        data.edge_index = knn_graph(data.x, k=n_neighbors, loop=False)

        #prints some graphs if wanted
        if graph_drawer and event_id in sliced_unique_events:
            graph_visualisation(data)
        else:
            None

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