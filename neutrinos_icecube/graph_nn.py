import math
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing
from torch_cluster import knn_graph
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import logging

torch.set_num_threads(20)

#pylint:disable = invalid-name

def angular_dist_score(y_true, y_pred):
    """
    calculate the MAE of the angular distance between two directions.
    The two vectors are first converted to cartesian unit vectors,
    and then their scalar product is computed, which is equal to
    the cosine of the angle between the two vectors. The inverse
    cosine (arccos) thereof is then the angle between the two input vectors

    Args:
    y_true: target tensor
    y_pred: output tensor from the neural network

    Returns:
        _type_: _description_
    """

    # print(y_true.shape)
    # print(y_pred.shape)
    az_pred = y_pred[:, 0]
    az_true = y_true[:, 0]
    zen_true = y_true[:, 1]
    zen_pred = y_pred[:, 1]

    # print(az_pred.shape)
    # print(az_true.shape)
    # print(zen_pred.shape)
    # print(zen_true.shape)

    # Combine non-finite masks for all input tensors
    non_finite_mask = (
        ~torch.isfinite(az_true)
        | ~torch.isfinite(zen_true)
        | ~torch.isfinite(az_pred)
        | ~torch.isfinite(zen_pred)
    )

    # Apply the mask to filter out non-finite values
    az_true = az_true[~non_finite_mask]
    zen_true = zen_true[~non_finite_mask]
    az_pred = az_pred[~non_finite_mask]
    zen_pred = zen_pred[~non_finite_mask]

    # Pre-compute all sine and cosine values
    sa1 = torch.sin(az_true)
    ca1 = torch.cos(az_true)
    sz1 = torch.sin(zen_true)
    cz1 = torch.cos(zen_true)
    sa2 = torch.sin(az_pred)
    ca2 = torch.cos(az_pred)
    sz2 = torch.sin(zen_pred)
    cz2 = torch.cos(zen_pred)

    # Scalar product of the two cartesian vectors (x = sz*ca, y = sz*sa, z = cz)
    scalar_prod = sz1 * sz2 * (ca1 * ca2 + sa1 * sa2) + (cz1 * cz2)

    # Scalar product of two unit vectors is always between -1 and 1
    # Clip to avoid numerical instability
    scalar_prod = torch.clamp(scalar_prod, -1.0, 1.0)

    # Convert back to an angle (in radians)
    return (torch.abs(torch.acos(scalar_prod)))


def dataset_skimmer(df, geom):
    """
    Prepares the dataset for padding operation.

    Args:
        df (pandas Dataframe): contains information on the event hits
        geom (pandas Dataframe): contains information on the geometry of the experiment

    Returns:
        pandas Dataframe: contains the skimmed hits for the events, with information on the geometry
    """
    # the flag "auxiliary" takes into account information from the MC about the goodness of the hit
    df = df[
        df["auxiliary"] == False
    ]  

    df = df.drop(labels="auxiliary", axis=1)

    df_with_geom = df.merge(geom, how="left", on="sensor_id").reset_index(
        drop=True
    )  # merges the two feature datasets

    # changes the type of two columns
    df_with_geom["event_id"].astype(np.int32)
    df_with_geom["charge"].astype(np.float16)

    # sort the events by charge and drop hits where the same sensor lit,
    # keeping the hit with the highest value of charge

    df_with_geom2 = (df_with_geom.sort_values("charge", ascending=False)
                                .drop_duplicates(["event_id", "sensor_id"])
                            )  # keep the sorting on the charge

    # add a counter of hits per event, and drops hits after the 25th one

    df_with_geom2["n_counter"] = df_with_geom2.groupby("event_id").cumcount()

    df_with_geom2 = df_with_geom2[df_with_geom2.n_counter < 20]
    
    #print(df_with_geom2)

    return df_with_geom2

def padding_function(df_with_geom):
    """
    adds a zero-padding to take into account the different number of hits per event

    Args:
        df_with_geom (pandas Dataframe): dataframe with feature information

    Returns:
        pandas Dataframe: dataframe with the same number of hit for each event
    """
    # compute the number of hits per event
    maxima = df_with_geom.groupby("event_id")["n_counter"].max().values

    # find the number of rows to be added during the padding

    n_counter_1 = np.where(maxima > 19, maxima, 19)
    diff = np.array([n_counter_1 - maxima])

    #set a multi-index on the dataframe
    df_with_geom.set_index(["event_id", "n_counter"], inplace=True)

    n_rows = np.sum(diff)

    #take the array of event IDs
    ev_ids = np.unique(df_with_geom.index.get_level_values(0).values)

    zeros = np.zeros((n_rows, 6), dtype=np.int32)

    #reshape the arrays to the correct shape
    diff_reshaped = diff.flatten()
    ev_ids_reshaped = np.reshape(ev_ids, (len(ev_ids), 1))

    #create a new index witht the events IDs to be used on the padded dataframe
    new_index = np.repeat(ev_ids_reshaped, diff_reshaped)

    # creates a dataframe filled with zeros to be cancatenated to the data dataframe
    pad_df = pd.DataFrame(
        zeros, index=new_index, columns=df_with_geom.columns
    ).reset_index(drop=False)
    #renames the column of the indexes and drops the old one
    pad_df["event_id"] = pad_df["index"]
    pad_df = pad_df.drop(labels=["index"], axis=1)

    #creates the hit counter for the padded dataframe and sets event_id and n_counter as multi-index
    pad_df["n_counter"] = (
        pad_df.groupby("event_id").cumcount()
        + df_with_geom.groupby("event_id").cumcount().max()
        + 1
    )
    pad_df = pad_df.set_index(["event_id", "n_counter"])

    # concatenates the two dataframes, and group the hits by event id

    df_final = pd.concat([df_with_geom, pad_df])

    df_final = df_final.sort_index()
    df_final = df_final.reset_index(drop=False)

    #creates a new index that counts all the hits in an event and drops unnecessary columns

    df_final["counter"] = df_final.groupby("event_id").cumcount()

    df_final = df_final.set_index(["event_id", "counter"])

    #drops unnecessary columns
    df_final = df_final.drop(labels=["n_counter", "sensor_id"], axis=1)
    print(df_final)
    return df_final

def targets_definer(df_final):
    """
        creates a dataframe that contains the targets for each event
    Args:
        df_final (pandas Dataframe): feature dataframe from which the event IDs are taken

    Returns:
        pandas Dataframe: dataframe with azimuth and zenith for each event
    """
    res = pd.read_parquet("/scratchnvme/cicco/cmepda/train_meta.parquet")

    #the dataset contains information on all the datasets, 
    # so targets for the events considered need to be extracted

    # gets the list of event IDs

    events = df_final.index.get_level_values(0).unique()

    #takes only the targets for the events present in the feature dataframe
    res1 = res[res.event_id.isin(events)]

    res1 = res1.sort_index()

    #drops unnecessary columns
    res1 = res1.drop(
        labels=["first_pulse_index", "last_pulse_index", "batch_id"], axis=1
    )
    print("printing targets")
    print(res1)
    return res1


def unstacker(df_final):
    """
    Creates a dataframe where each row contains one event
    Args:
        df_final (pandas Dataframe): dataframe containing one hit per row

    Returns:
        pandas Dataframe: dataframe containing one event per row
    """
    print(df_final)

    # unstack the dataset on the counter level of index, so that all the hits per event are set in a single row
    df_final1 = df_final.unstack()

    print(df_final1)

    return df_final1


def model_creator():
    """
    creates the model for the neural network

    Returns:
        _type_: model of the neural network
    """

    class DNNLayer(MessagePassing):
        """
        custom layer for the Graph Neural Network (GNN) that inherits from the Message Passing layer
        Args:
            MessagePassing (_type_): layer for GNN
        """

        def __init__(self, in_channels, out_channels):
            
            super().__init__(aggr="mean") #aggregation method for the nodes used the mean

            #creates a MLP used for message
            self.mlp = nn.Sequential(
                nn.Linear(2 * in_channels, out_channels),
                nn.ReLU(),
                nn.Linear(out_channels, out_channels),
            )
            self.simpler_mlp = nn.Sequential(
                nn.Linear(out_channels, out_channels),
                nn.ReLU(),
                nn.Linear(out_channels, out_channels),
            )

        def forward(self, h, edge_index):
            h1 = self.propagate(edge_index, h=h)
            h2 = self.simpler_mlp(h1)
            return h2

        def message(self, h_j, h_i):
            #message that will be propagated in the forward function. It is made considering the EdgeConv idea
            input = torch.cat([h_i, h_j - h_i], dim=-1)
            return self.mlp(input)

    class Graph_Network(nn.Module):
        """
        Creates the effective GNN
        Args:
            nn (_type_): _description_
        """
        def __init__(self):
            super().__init__()
            N_features = 30
            #defines the layer that will be then used in the GNN
            self.f1 = DNNLayer(6, N_features)
            self.f2 = DNNLayer(N_features, N_features)
            self.f3 = DNNLayer(N_features, N_features)
            self.f4 = DNNLayer(N_features, N_features)
            self.f5 = DNNLayer(N_features, N_features)

            self.global_pooling = torch_geometric.nn.global_mean_pool

            self.output = nn.Linear(N_features, 2)

        def forward(self, data):
            """
            assembles the GNN
            Args:
                data (Dataset): dataset that contains both the features and the targets

            Returns:
                _type_: _description_
            """
            x = data.x
            edge_index = data.edge_index
            # print("shape before f1",x.shape)

            h = self.f1(h=x, edge_index=edge_index)
            # print("shape after f1",h.shape)

            h = h.relu()
            # print("shape after relu",h.shape)

            h = self.f2(h=h, edge_index=edge_index)
            # print("shape after f2",h.shape)

            h = h.relu()
            # print("shape after relu",h.shape)

            h = self.f3(h=h, edge_index=edge_index)
            # print("shape after f3",h.shape)

            h = h.relu()
            # print("shape after relu",h.shape)

            h = self.f4(h=h, edge_index= edge_index)

            h = h.relu()

            h = self.f5(h=h, edge_index= edge_index)

            h = h.relu()


            h = self.global_pooling(h, data.batch)

            # print("shape after global pooling", h.shape)

            h = self.output(h)

            # print("shape after linear layer", h.shape)
            return h

    Model = Graph_Network()
    # print(Model)

    return Model


def tensor_creator(df, targets, label):
    """
    Takes the pandas Dataframes and creates the torch Tensor for the GNN
    Args:
        df (pandas Dataframe): dataframe with the features
        targets (pandas Dataframe): dataframe with the targets
        label (str): string that differentiates training and test samples for debugging
    """
    def graph_visualisation(data):
        """
        plots the graphs on the xy plane and yz plane.
        Args:
            data (torch tensor): torch tensor with both targets and features
        """
        g = torch_geometric.utils.to_networkx(data)
        x = data.x

        pos = {i: (x[i, 1].item(), x[i, 2].item()) for i in range(len(x))}
        subset_graph = g.subgraph(range(len(x)))
        #uses the charge of the hits as color scheme
        node_colors = x[:, 0].numpy()
        node_size = 40

        #! AZIMUTH

        fig, ax = plt.subplots()
        nx.draw(
            subset_graph,
            pos=pos,
            node_color=node_colors,
            node_size=node_size,
            cmap="viridis",
            with_labels=False,
            ax=fig.add_subplot(121),
        )
        plt.show()

        # plt.axhline(0, color='black', linestyle='--', linewidth=1)
        # plt.axvline(0, color='black', linestyle='--', linewidth=1)

        # x_min, x_max = x[:, 1].min().item(), x[:, 1].max().item()
        # y_min, y_max = x[:, 2].min().item(), x[:, 2].max().item()
        # plt.xlim(x_min, x_max)
        # plt.ylim(y_min, y_max)

        # # Add axis labels
        # plt.text(x_min - 0.1 * (x_max - x_min), y_min - 0.1 * (y_max - y_min), 'x', ha='center')
        # plt.text(x_min - 0.15 * (x_max - x_min), y_min + 0.5 * (y_max - y_min), 'y', va='center', rotation='vertical')

        # # Add tick values
        # plt.text(x_min, y_min - 0.05 * (y_max - y_min), f'{x_min:.2f}', ha='center')
        # plt.text(x_max, y_min - 0.05 * (y_max - y_min), f'{x_max:.2f}', ha='center')
        # plt.text(x_min - 0.08 * (x_max - x_min), y_min, f'{y_min:.2f}', va='center', rotation='vertical')
        # plt.text(x_min - 0.08 * (x_max - x_min), y_max, f'{y_max:.2f}', va='center', rotation='vertical')

        ax = plt.gca()
        ax.set(xlabel="x", ylabel="y")
        text_azimuth = data.y[:, 0]
        text_zenith = data.y[:, 1]

        scatter = plt.scatter(
            [], [], c=[], cmap="viridis", vmin=node_colors.min(), vmax=node_colors.max()
        )

        plt.colorbar(scatter, label="Charge")
        plt.text(
            0.5,
            1.05,
            "azimuth is: " + str(text_azimuth),
            transform=plt.gca().transAxes,
            fontsize=12,
            ha="center",
        )

        plt.savefig("graph_proj_x_y_" + str(event_id) + "_" + str(label) + ".png")
        plt.close()

        #! ZENITH

        fig, ax = plt.subplots()

        pos1 = {i: (x[i, 2].item(), x[i, 3].item()) for i in range(len(x))}
        subset_graph1 = g.subgraph(range(len(x)))

        node_colors = x[:, 0].numpy()
        scatter = plt.scatter(
            [], [], c=[], cmap="viridis", vmin=node_colors.min(), vmax=node_colors.max()
        )

        # print("node colors", node_colors)
        # print(data.x)
        nx.draw(
            subset_graph1,
            pos=pos1,
            node_color=node_colors,
            node_size=node_size,
            cmap="viridis",
            with_labels=False,
            ax=fig.add_subplot(121),
        )
        plt.show()
        plt.text(
            0.5,
            1.05,
            "zenith is:" + str(text_zenith),
            transform=plt.gca().transAxes,
            fontsize=12,
            ha="center",
        )

        # x_min, x_max = x[:, 2].min().item(), x[:, 2].max().item()
        # y_min, y_max = x[:, 3].min().item(), x[:, 3].max().item()
        # plt.xlim(x_min, x_max)
        # plt.ylim(y_min, y_max)

        # # Add axis labels
        # plt.text(x_min - 0.1 * (x_max - x_min), y_min - 0.11 * (y_max - y_min), 'y', ha='center')
        # plt.text(x_min - 0.15 * (x_max - x_min), y_min + 0.5 * (y_max - y_min), 'z', va='center', rotation='vertical')

        # # Add tick values
        # plt.text(x_min, y_min - 0.05 * (y_max - y_min), f'{x_min:.2f}', ha='center')
        # plt.text(x_max, y_min - 0.05 * (y_max - y_min), f'{x_max:.2f}', ha='center')
        # plt.text(x_min - 0.08 * (x_max - x_min), y_min, f'{y_min:.2f}', va='center', rotation='vertical')
        # plt.text(x_min - 0.08 * (x_max - x_min), y_max, f'{y_max:.2f}', va='center', rotation='vertical')

        ax = plt.gca()
        ax.set(xlabel="x", ylabel="y")

        plt.colorbar(scatter, label="Charge")

        plt.savefig("graph_proj_y_z_" + str(event_id) + "_" + str(label) + ".png")

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


def training_function(model, dataset_train, dataset_test):
    """
    Trains the GNN
    Args:
        model (_type_): Model defined for the GNN
        dataset_train (Dataset): Dataset used for the training
        dataset_test (Dataset): Dataset used for the test

    Returns:
        _type_: _description_
    """

    debug = False


    class MyDataset(Dataset):
        """
        Custom Dataset for handling the data
        Args:
            Dataset (Dataset): _description_
        """
        def __init__(self, data_list):
            self.data_list = data_list

        def __len__(self):
            return len(self.data_list)

        def __getitem__(self, idx):
            data = self.data_list[idx]
            return data

    def root_mean_squared_error(y_true, y_pred):
        """
        Definition of RMSE
        Args:
            y_true (torch.tensor): truth values for azimuth and zenith
            y_pred (torch.tensor): predicted values for azimuth and zenith

        Returns:
            float: RMSE calculated
        """
        squared_diff = (y_true - y_pred) ** 2
        mean_squared_error = torch.mean(squared_diff)
        rmse = torch.sqrt(mean_squared_error)
        return rmse

    custom_dataset_train = MyDataset(dataset_train)
    custom_dataset_test = MyDataset(dataset_test)

    loss_func = torch.nn.MSELoss()

    train_loader = DataLoader(custom_dataset_train, batch_size=256, shuffle=True)
    test_loader = DataLoader(custom_dataset_test, batch_size=256, shuffle=True)

    # for batch in train_loader:
    # print(type(batch))
    # print(batch.batch)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    def train(model, optimizer, loader):
        """
        Train function
        Args:
            model (_type_): _description_
            optimizer (_type_): _description_
            loader (Dataset): training dataset

        Returns:
            _type_: _description_
        """
        model.train()
        total_loss = 0
        total_rmse = 0

        batch_train_loss = []

        batch_train_rmse = []

        for batch_idx, data in enumerate(loader):
            optimizer.zero_grad()

            output = model(data)
            #loss = angular_dist_score(data.y, output)
            loss = loss_func(output, data.y)
            
            # print(type(loss))
            # if torch.isnan(loss).any:
            #     print(f"NaN in training loss at batch: {batch_idx}")
            #     print(f"Input is: {data}")
            #     print(f"Outputs: {output}")
            #     print(f'Loss tensor: {loss}')

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_rmse += root_mean_squared_error(data.y, output).item()
            # print(type(rmse))
            # if math.isnan(rmse):
            #     print(f"NaN in training RMSE at batch: {batch_idx}")
            #     print(f"Input is: {data}")
            #     print(f"Outputs: {output}")
            #     print(f"RMSE: {rmse}")


            # batch_average_loss = total_loss / len(loader)
            # batch_average_rmse = total_rmse / len(loader)

            # print("average training loss for each batch", loss.item())
            # print("average training rmse for each batch", rmse)

            batch_train_loss.append(loss.item())
            batch_train_rmse.append(root_mean_squared_error(data.y, output).item())

        average_loss = total_loss / len(train_loader.dataset)
        average_rmse = total_rmse / len(train_loader.dataset)
        if debug:
            return batch_train_loss, batch_train_rmse
        else:
            return average_loss, average_rmse

    def evaluate(model, loader):
        """
        Function for validation
        Args:
            model (_type_): _description_
            loader (Dataset): test dataset

        Returns:
            _type_: _description_
        """
        model.eval()
        total_loss = 0.0
        total_rmse = 0.0

        batch_test_loss = []
        batch_test_rmse = []

        with torch.no_grad():
            for batch_idx, data in enumerate(loader):
                output = model(data)
                #loss = angular_dist_score(data.y, output)

                # if torch.isnan(loss).any:
                #     print(f"NaN in test loss at batch: {batch_idx}")
                #     print(f"Input is: {data}")
                #     print(f"Outputs: {output}")
                #     print(f'Loss tensor: {loss}')

                loss = loss_func(output, data.y)
                total_loss += loss.item()
                total_rmse += root_mean_squared_error(data.y, output).item()



                # if math.isnan(rmse):
                #     print(f"NaN in test RMSE at batch: {batch_idx}")
                #     print(f"Input is: {data}")
                #     print(f"Outputs: {output}")
                #     print(f"RMSE: {rmse}")


                # batch_average_loss = total_loss / len(loader)
                # batch_average_rmse = total_rmse / len(loader)

                # print("average test loss for each batch", loss.item())
                # print("average test rmse for each batch", rmse)

                batch_test_loss.append(loss.item())
                batch_test_rmse.append(root_mean_squared_error(data.y, output).item())

        average_loss = total_loss / len(test_loader.dataset)
        average_rmse = total_rmse / len(test_loader.dataset)

        if debug:
            return batch_test_loss, batch_test_rmse
        else:
            return average_loss, average_rmse

    train_losses = []
    test_losses = []

    train_rmses = []
    test_rmses = []

    number_of_epochs = 2 if debug else 171

    for epoch in range(1, number_of_epochs):

        train_loss, train_rmse = train(model, optimizer, train_loader)
        test_loss, test_rmse = evaluate(model, test_loader)
        #checks if either the loss function or the RMSE, both for training and validation, have NaN values,
        #and stops the training if so
        if not debug and (
               math.isnan(train_loss)
            or math.isnan(train_rmse)
            or math.isinf(train_loss)
            or math.isinf(train_rmse)
            or math.isnan(test_loss)
            or math.isnan(test_rmse)
            or math.isinf(test_loss)
            or math.isinf(test_rmse)
        ):
            print("Training stopped due to NaN or infinite values.")
            break

        test_losses.append(test_loss)
        train_losses.append(train_loss)

        test_rmses.append(test_rmse)
        train_rmses.append(train_rmse)
        if not debug:
            print(
                f"Epoch: {epoch:02d}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Train RMSE: {train_rmse:.4f}, Test RMSE: {test_rmse: .4f}"
            )
    #if debug is True, prints the plot of loss and RMSE batch per batch for epoch 1 only

    if debug:
        #! ONE EPOCH ONLY, BATCH PER BATCH

        plt.figure(figsize=(10, 5))
        plt.title("Training and Validation Loss")
        plt.plot(test_loss, label="val")
        plt.plot(train_loss, label="train")
        plt.xlabel("iterations")
        plt.ylabel("Loss")
        plt.yscale('log')
        plt.legend()
        plt.show()
        plt.savefig("graph_log_batch_mean_loss_30_0_0001_loss_MSE.png")
        plt.close()

        plt.figure(figsize=(10, 5))
        plt.title("Training and Validation RMSE")
        plt.plot(test_rmse, label="val")
        plt.plot(train_rmse, label="train")
        plt.xlabel("iterations")
        plt.ylabel("RMSE")
        plt.yscale('log')
        plt.legend()
        plt.show()
        plt.savefig("graph_log_batch_mean_RMSE_30_0_0001_loss_MSE.png")
        plt.close()

    else:
        #plots the loss and RMSE for all the epochs
        plt.figure(figsize=(10, 5))
        plt.title("Training and Validation Loss")
        plt.plot(test_losses, label="val")
        plt.plot(train_losses, label="train")
        plt.xlabel("iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()
        plt.savefig("graph_mean_170_epochs_20_hits_5_DNN_loss_30_0_001_loss_MSE_batch_256_simpler_mlp_knn_7_2files.png")
        plt.close()

        plt.figure(figsize=(10, 5))
        plt.title("Training and Validation RMSE")
        plt.plot(test_rmses, label="val")
        plt.plot(train_rmses, label="train")
        plt.xlabel("iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()
        plt.savefig("graph_mean_170_epochs_20_hits_5_DNN_RMSE_30_0_001_loss_MSE_batch_256_simpler_mlp_knn_7_2files.png")
        plt.close()


if __name__ == "__main__":

    DATA_PATH = "/scratchnvme/cicco/cmepda/"
    DATA_FILES = [
        "batch_1.parquet",
        "batch_2.parquet",
        # "batch_10.parquet",
        # "batch_11.parquet",
        # "batch_100.parquet",
        # "batch_101.parquet",
    ]  # , 'batch_102.parquet', 'batch_103.parquet']
    geometry = pd.read_csv("/scratchnvme/cicco/cmepda/sensor_geometry.csv")
    combined_data = pd.DataFrame()
    combined_res = pd.DataFrame()

    for data_file in DATA_FILES:
        dataframe = pd.read_parquet(DATA_PATH + data_file).reset_index()
        dataframe_final = dataset_skimmer(dataframe, geometry)

        dataframe_final1 = padding_function(dataframe_final)

        targets = targets_definer(dataframe_final1)

        print("unstacking")

        dataframe_final3 = unstacker(dataframe_final1)
        print(dataframe_final3)

        combined_data = pd.concat([combined_data, dataframe_final3], ignore_index=False)

        combined_res = pd.concat([combined_res, targets], ignore_index=False)

    print("combined data",combined_data)
    print(combined_res)

    print("creating the model")

    print("splitting the dataset")
    #splits the dataset into training and test 
    X_train, X_test, Y_train, Y_test = train_test_split(
        combined_data, combined_res, test_size=0.3, random_state=42
    )

    print(X_train.shape)
    print(Y_train.shape)

    print("creating the training tensor")

    data_train = tensor_creator(X_train, Y_train, "train")

    print("creating the test tensor")

    data_test = tensor_creator(X_test, Y_test, "test")
    print("creating the model")

    model = model_creator()

    print("starting the training")

    training_function(model, data_train, data_test)
