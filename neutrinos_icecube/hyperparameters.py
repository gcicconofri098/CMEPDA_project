"""File that contains the chosen parameters for the Neural Network."""

n_neighbors = 7             #number of neighbors for the knn_graph
N_features = 40             #dimensionality of the hidden layers
dropout_value = 0.1         #sets the value for the dropout layer

#Early stopping

patience = 5                #number of epochs to wait before stopping the training
min_delta = 0.001           #value of the validation loss difference between two epochs