""" Creates and implements the model of the Graph Neural Network.

    Returns:
        (Graph_Network): An istance of the class Graph_Network
"""

import torch
import torch_geometric

import torch.nn as nn
from torch_geometric.nn import MessagePassing

from logging_conf import setup_logging

import parameters 
import hyperparameters

logger = setup_logging('model_log')

def model_creator():

    """ Creates the model for the neural network

    Returns:
        _type_: model of the neural network
    """

    class DNNLayer(MessagePassing):

        """
        Custom layer for the Graph Neural Network (GNN) that inherits from the Message Passing layer
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels

        Attributes:
            mlp (nn.Sequential): MLP for message passing
            simpler_mlp (nn.Sequential): additional MLP added after the propagation
        
        Methods
            forward(torch.Tensor, torch.Tensor): Performs a forward pass through the DNNLayer
        """

        def __init__(self, in_channels, out_channels):
            
            super().__init__(aggr="mean") #aggregation method for the nodes used the mean

            #creates a MLP used as message
            self.mlp = nn.Sequential(
                nn.Linear(2 * in_channels, out_channels),
                nn.ReLU(),
                nn.Dropout(p=hyperparameters.dropout_value),
                nn.Linear(out_channels, out_channels),
            )
            self.simpler_mlp = nn.Sequential(
                nn.Linear(out_channels, out_channels),
                nn.ReLU(),
                nn.Linear(out_channels, out_channels),
            )

        def forward(self, h, edge_index):

            """ Performs a forward pass through the DNNLayer.

            Args:
                h (torch.Tensor): Node features
                edge_index (torch.Tensor): Graph connection defined by the edge indices

            Returns:
                torch.Tensor: Output tensor
            """

            h1 = self.propagate(edge_index, h=h)
            h2 = self.simpler_mlp(h1)
            return h2

        def message(self, h_j, h_i):

            """ Defines the message to be propagated in the forward function.
                The message is creating considering the EdgeConv concept.

            
            Args:
                h_j (torch.Tensor): Target node features
                h_i (torch.Tensor): Source node features

            Returns:
                torch.Tensor: Output tensor
            """

            new_input = torch.cat([h_i, h_j - h_i], dim=-1)
            return self.mlp(new_input)

    class Graph_Network(nn.Module):

        """
        Implementation of the GNN.
        Args:
            None
        
        Attributes:
            f1, f2, f3,f4,f5 (DNNLayer): Feedforward layers with additional Dropout layer
            global_pooling (torch_geometric.nn.Pooling): Global pooling layer for aggregating the node features
            output (nn.Linear): Output layer
        
        Methods:
            forward(data): Performs a pass through the model
        """

        def __init__(self):
            super().__init__()

            #defines the layers that will be then used in the GNN
            self.f1 = DNNLayer(6, hyperparameters.N_features)
            self.f2 = DNNLayer(hyperparameters.N_features, hyperparameters.N_features)
            self.f3 = DNNLayer(hyperparameters.N_features, hyperparameters.N_features)
            self.f4 = DNNLayer(hyperparameters.N_features, hyperparameters.N_features)
            self.f5 = DNNLayer(hyperparameters.N_features, hyperparameters.N_features)

            self.global_pooling = torch_geometric.nn.global_mean_pool

            self.output = nn.Linear(hyperparameters.N_features, 2)

        def forward(self, data):

            """ Performs a forward pass through the GNN model.
            Args:
                data (Data): Data that contains both the features and the targets

            Returns:
                torch.Tensor: Output tensor after GNN processing
            """

            x = data.x
            edge_index = data.edge_index
            logger.debug(f"shape before f1: {x.shape}")

            h = self.f1(h=x, edge_index=edge_index)
            logger.debug(f"shape after f1: {h.shape}")

            h = h.relu()
            logger.debug(f"shape after relu1: {h.shape}")

            h = self.f2(h=h, edge_index=edge_index)
            logger.debug(f"shape after f2: {h.shape}")

            h = h.relu()
            logger.debug(f"shape after relu2: {h.shape}")

            h = self.f3(h=h, edge_index=edge_index)
            logger.debug(f"shape after f3: {h.shape}")

            h = h.relu()
            logger.debug(f"shape after relu3: {h.shape}")

            h = self.f4(h=h, edge_index= edge_index)
            logger.debug(f"shape after f4: {h.shape}")

            h = h.relu()
            logger.debug(f"shape after relu4: {h.shape}")

            h = self.f5(h=h, edge_index= edge_index)
            logger.debug(f"shape after f5: {h.shape}")

            h = h.relu()
            logger.debug(f"shape after relu5: {h.shape}")

            h = self.global_pooling(h, data.batch)

            logger.debug(f"shape after global pooling: {h.shape}")

            h = self.output(h)

            logger.debug(f"output shape: {h.shape}")
            return h

    graph_model = Graph_Network()
    print(graph_model)

    return graph_model