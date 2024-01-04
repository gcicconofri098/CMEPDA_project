'''File that contains the chosen parameters for the Neural Network.
    

    Settable hyperparameters:
    
        n_neighbors (int): number of neighbors for the knn_graph
        N_features (int): dimensionality of the hidden layers
        dropout_value (float): fraction of dropped connections for the dropout layer
        patience (int): number of epochs to wait before stopping the training
        min_delta (float): value of the validation loss difference between two epochs
'''

n_neighbors = 10
N_features = 50
N_layers = 5 #doesn't actually do anything, just a placeholder
dropout_value = 0.1
learning_rate = 0.0005
number_epochs = 190
batch_size = 512

#Early stopping

patience = 8
min_delta = 0.001


#Grid_search hyperparameters

learning_rate_grid = [0.01, 0.005, 0.001]
