""" Handles the training of the GNN.

Returns:

    lists: List of the loss and RMSE for each epoch, for both training and test 
"""
import os
import math
import torch
from torch_geometric.loader import DataLoader

from angular_distance_loss import angular_dist_score
from logs.logging_conf import setup_logging
import parameters
import hyperparameters

logger = setup_logging('training_log')

def training_function(model, custom_dataset_train, custom_dataset_val):

    """ Trains the GNN.

    Args:

        model (Graph_Network): Istance of the class Graph_Network
        dataset_train (Dataset): Dataset used for the training
        dataset_test (Dataset): Dataset used for the test

    Returns:

        _type_: _description_
    
    """

    class EarlyStopper:

        """Implements the Early Stopping callback."""
        
        def __init__(self, patience=1, min_delta=0):
            self.patience = patience
            self.min_delta = min_delta
            self.counter = 0
            self.min_validation_loss = float('inf')

        def early_stop(self, validation_loss):
            if validation_loss < self.min_validation_loss:
                self.min_validation_loss = validation_loss
                self.counter = 0
            elif validation_loss > (self.min_validation_loss + self.min_delta):
                self.counter += 1
                if self.counter >= self.patience:
                    return True
            return False



    def root_mean_squared_error(y_true, y_pred):

        """ Defines the RMSE.

        Args:

            y_true (torch.tensor): truth values for azimuth and zenith
            y_pred (torch.tensor): predicted values for azimuth and zenith

        Returns:

            rmse (float): RMSE calculated
        
        """

        squared_diff = (y_true - y_pred) ** 2
        mean_squared_error = torch.mean(squared_diff)
        rmse = torch.sqrt(mean_squared_error)
        return rmse

    loss_func = torch.nn.L1Loss()

    train_loader = DataLoader(custom_dataset_train, batch_size=hyperparameters.batch_size, shuffle=True)
    val_loader = DataLoader(custom_dataset_val, batch_size=hyperparameters.batch_size, shuffle=True)

    # for batch in train_loader:
    # print(type(batch))
    # print(batch.batch)



    def train(model, optimizer, loader):

        """ Training function.

        Args:

            model (Graph_Network): istance of the class Graph_Network
            optimizer (torch.optim.Adam): optimizer used in the training
            loader (Dataset): training dataset

        Returns:

            average_loss, average_rmse (float, float): average training loss and training RMSE through the epoch.
            
            batch_train_loss, batch_train_rmse (list, list): if in debug mode, list of training loss and training RMSE
                through the epoch, not mediated. 

        """

        logger.debug(len(loader))
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

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_rmse += root_mean_squared_error(data.y, output).item()

            batch_train_loss.append(loss.item())
            batch_train_rmse.append(root_mean_squared_error(data.y, output).item())

        average_loss = total_loss / len(loader)
        average_rmse = total_rmse / len(loader)
        if parameters.debug_value:
            return batch_train_loss, batch_train_rmse
        else:
            return average_loss, average_rmse

    def evaluate(model, loader):

        """ Function for validation.

        Args:
            model (Graph_Network): istance of the class Graph_Network
            loader (Dataset): Validation dataset

        Returns:
        
            average_loss, average_rmse (float, float): average validation loss and validation RMSE through the epoch.
            
            batch_train_loss, batch_train_rmse (list, list): if in debug mode, list of validation loss and validation RMSE
                through the epoch, not mediated. 
        """

        model.eval()
        total_loss = 0.0
        total_rmse = 0.0

        batch_val_loss = []
        batch_val_rmse = []

        logger.debug(len(loader))


        with torch.no_grad():
            for batch_idx, data in enumerate(loader):
                output = model(data)

                loss = loss_func(output, data.y)
                total_loss += loss.item()
                total_rmse += root_mean_squared_error(data.y, output).item()

                batch_val_loss.append(loss.item())
                batch_val_rmse.append(root_mean_squared_error(data.y, output).item())


        average_loss = total_loss / len(loader)
        average_rmse = total_rmse / len(loader)

        if parameters.debug_value:
            return batch_val_loss, batch_val_rmse
        else:
            return average_loss, average_rmse



    number_of_epochs = 1 if parameters.debug_value else hyperparameters.number_epochs

    best_val_loss = float('inf')
    best_model_weights = None
    
    best_lr = 0
    for lr_grid in hyperparameters.learning_rate_grid:

        train_losses = []
        val_losses = []

        train_rmses = []
        val_rmses = []

        current_loop_best_val_loss = float('inf')
        current_loop_best_weights = None
        early_stopper = EarlyStopper(patience=hyperparameters.patience, min_delta=hyperparameters.min_delta)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr_grid)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience= 7)

        for epoch in range(1, number_of_epochs +1):

            train_loss, train_rmse = train(model, optimizer, train_loader)
            val_loss, val_rmse = evaluate(model, val_loader)
            
            scheduler.step(val_loss)
            #checks if either the loss function or the RMSE, both for training and validation, have NaN values,
            #and stops the training if so
            if (
                math.isnan(train_loss)
                or math.isnan(train_rmse)
                or math.isinf(train_loss)
                or math.isinf(train_rmse)
                or math.isnan(val_loss)
                or math.isnan(val_rmse)
                or math.isinf(val_loss)
                or math.isinf(val_rmse)
            ):
                print("Training stopped due to NaN or infinite values.")
                break
            

            val_losses.append(val_loss)
            train_losses.append(train_loss)

            val_rmses.append(val_rmse)
            train_rmses.append(train_rmse)

            logger.info(
                    f"Epoch: {epoch:02d}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Train RMSE: {train_rmse:.4f}, Validation RMSE: {val_rmse: .4f}"
                )
            
            if val_loss < current_loop_best_val_loss:
                current_loop_best_val_loss = val_loss
                current_loop_best_weights = model.state_dict()
                logger.info(f"current best validation loss value is {current_loop_best_val_loss} for learning rate {lr_grid}")

            if early_stopper.early_stop(val_loss):
                logger.info("Stopping the training with early stopping")
                print(f"current best validation loss value is {current_loop_best_val_loss} for learning rate {lr_grid}")
                break

        if current_loop_best_val_loss < best_val_loss:
            best_val_loss = current_loop_best_val_loss
            best_model_weights = current_loop_best_weights
            best_lr = lr_grid
            best_lr_str = str(best_lr).replace(".", "_")
            print(f"best value loss is {best_val_loss} with learning rate {lr_grid}")
            selected_train_losses, selected_val_losses, selected_train_rmses, selected_val_rmses = train_losses, val_losses, train_rmses, val_rmses
    
    print(f"final best value loss is {best_val_loss} with learning rate {best_lr}")

    torch.save(best_model_weights, 'neutrinos_icecube/saved_models/model_lr_' + best_lr_str + 'early_stop_RLR_'+ str(parameters.n_hits)+'_hits_'+str(hyperparameters.N_layers) +'DNN_ '+str(hyperparameters.N_features) +'_loss_MAE_batch_ '+str(hyperparameters.batch_size) + '_dropout_simpler_mlp_knn_' + str(hyperparameters.n_neighbors) +'.pth')

    return selected_train_losses, selected_val_losses, selected_train_rmses, selected_val_rmses, best_lr
