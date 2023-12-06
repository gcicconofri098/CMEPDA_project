import torch
import torch_geometric
import torch.nn as nn
from torch_geometric.nn import MessagePassing
import matplotlib.pyplot as plt
import networkx as nx


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