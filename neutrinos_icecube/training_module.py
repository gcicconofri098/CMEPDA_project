""" Handles the training of the GNN.

Returns:
    lists: List of the loss and RMSE for each epoch, for both training and test 
"""

import math
import torch
from torch_geometric.loader import DataLoader

from angular_distance_loss import angular_dist_score
from logging_conf import setup_logging
import parameters
import hyperparameters

logger = setup_logging('training_log')

def training_function(model, custom_dataset_train, custom_dataset_test):

    """ Trains the GNN.

    Args:
        model (_type_): Model defined for the GNN
        dataset_train (Dataset): Dataset used for the training
        dataset_test (Dataset): Dataset used for the test

    Returns:
        _type_: _description_
    """

    class EarlyStopper:
        """_summary_
        """
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

    early_stopper = EarlyStopper(patience=hyperparameters.patience, min_delta=hyperparameters.min_delta)


    def root_mean_squared_error(y_true, y_pred):

        """ Defines the RMSE.
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

    loss_func = torch.nn.MSELoss()

    train_loader = DataLoader(custom_dataset_train, batch_size=512, shuffle=True)
    test_loader = DataLoader(custom_dataset_test, batch_size=512, shuffle=True)

    # for batch in train_loader:
    # print(type(batch))
    # print(batch.batch)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    def train(model, optimizer, loader):

        """ Training function.
        Args:
            model (Graph_Network): istance of the class Graph_Network
            optimizer (torch.optim.Adam): optimizer used in the training
            loader (Dataset): training dataset

        Returns:
            average_loss, average_rmse (float, float): average loss and RMSE through the epoch.
            
            batch_train_loss, batch_train_rmse (list, list): if in debug mode, list of loss and RMSE
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

        average_loss = total_loss / len(loader)
        average_rmse = total_rmse / len(loader)
        if parameters.debug_value:
            return batch_train_loss, batch_train_rmse
        else:
            return average_loss, average_rmse

    def evaluate(model, loader):

        """ Function for validation.

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

        logger.debug(len(loader))


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


        average_loss = total_loss / len(loader)
        average_rmse = total_rmse / len(loader)

        if parameters.debug_value:
            return batch_test_loss, batch_test_rmse
        else:
            return average_loss, average_rmse

    train_losses = []
    test_losses = []

    train_rmses = []
    test_rmses = []

    number_of_epochs = 2 if parameters.debug_value else 191

    for epoch in range(1, number_of_epochs):

        train_loss, train_rmse = train(model, optimizer, train_loader)
        test_loss, test_rmse = evaluate(model, test_loader)
        #checks if either the loss function or the RMSE, both for training and validation, have NaN values,
        #and stops the training if so
        if (
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
        
        if early_stopper.early_stop(test_loss):
            logger.info("Stopping the training with early stopping")
            break

        test_losses.append(test_loss)
        train_losses.append(train_loss)

        test_rmses.append(test_rmse)
        train_rmses.append(train_rmse)

        logger.info(
                f"Epoch: {epoch:02d}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Train RMSE: {train_rmse:.4f}, Test RMSE: {test_rmse: .4f}"
            )
    return train_losses, test_losses, train_rmses, test_rmses

