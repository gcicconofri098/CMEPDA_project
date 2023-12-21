'''File that contains the chosen parameters for the Neural Network.

    Parameters:
    
        n_neighbors (int): number of neighbors for the knn_graph
        N_features (int): dimensionality of the hidden layers
        dropout_value (float): fraction of dropped connections for the dropout layer
        patience (int): number of epochs to wait before stopping the training
        min_delta (float): value of the validation loss difference between two epochs
'''

n_neighbors = 8
N_features = 45
dropout_value = 0.1
learning_rate = 0.001
number_epochs = 190

#Early stopping

patience = 5
min_delta = 0.001