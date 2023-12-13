'''File that contains the chosen parameters for the Neural Network.

    Args:

        :param int n_neighbors: number of neighbors for the knn_graph.
        :param int N_features: dimensionality of the hidden layers.
        :param float dropout_value: fraction of dropped connections for the dropout layer.
        :param int patience: number of epochs to wait before stopping the training.
        :param float min_delta: value of the validation loss difference between two epochs.
'''

n_neighbors = 7
N_features = 40
dropout_value = 0.1

#Early stopping

patience = 5
min_delta = 0.001