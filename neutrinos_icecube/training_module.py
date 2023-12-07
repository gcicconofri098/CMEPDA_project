import sys
import math
import logging
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch_geometric
from torch_geometric.loader import DataLoader

from angular_distance_loss import angular_dist_score

import parameters

logging.basicConfig(filename='training_log.log', level= parameters.log_value)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

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
        logging.debug(len(loader))
        model.train()
        total_loss = 0
        total_rmse = 0

        batch_train_loss = []

        batch_train_rmse = []

        for batch_idx, data in enumerate(loader):
            optimizer.zero_grad()

            output = model(data)

            loss = angular_dist_score(data.y, output)
            #loss = loss_func(output, data.y)
            loss_tensor = torch.tensor(loss, requires_grad=True)
            # print(type(loss))
            # if torch.isnan(loss).any:
            #     print(f"NaN in training loss at batch: {batch_idx}")
            #     print(f"Input is: {data}")
            #     print(f"Outputs: {output}")
            #     print(f'Loss tensor: {loss}')

            loss_tensor.backward()
            optimizer.step()
            total_loss += loss_tensor.item()
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

        logging.debug(len(loader))

        with torch.no_grad():
            for batch_idx, data in enumerate(loader):
                output = model(data)
                loss = angular_dist_score(data.y, output)

                # if torch.isnan(loss).any:
                #     print(f"NaN in test loss at batch: {batch_idx}")
                #     print(f"Input is: {data}")
                #     print(f"Outputs: {output}")
                #     print(f'Loss tensor: {loss}')

                #loss = loss_func(output, data.y)
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

    number_of_epochs = 2 if parameters.debug_value else 171

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

        test_losses.append(test_loss)
        train_losses.append(train_loss)

        test_rmses.append(test_rmse)
        train_rmses.append(train_rmse)

        logging.info(
                f"Epoch: {epoch:02d}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Train RMSE: {train_rmse:.4f}, Test RMSE: {test_rmse: .4f}"
            )
    return train_losses, test_losses, train_rmses, test_rmses

