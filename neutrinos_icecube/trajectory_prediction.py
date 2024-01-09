
""" Main module. Predicts the trajectory of the simulated neutrinos."""

import sys
import os
from pathlib import Path
import pandas as pd
import torch
import pickle
from sklearn.model_selection import train_test_split

from angular_distance_loss import angular_dist_score
from datasets.sample_loader import sample_loader
from pandas_handler import dataset_skimmer, padding_function, unstacker, targets_definer
from model_creation import model_creator
from tensor_creation import tensor_creator
from training_module import training_function
from plots.plots_loss_and_rmse import single_epoch_batch_loss_plots, loss_plots
from dataset_for_training import dataset_creator
import parameters as parameters
from logs.logging_conf import setup_logging

logger = setup_logging('main_log')

torch.set_num_threads(20)


#pylint:disable = invalid-name

if __name__ == "__main__":

    # Checks if CUDA is available
    if torch.cuda.is_available():

        num_gpus = torch.cuda.device_count()
        
        print(f"CUDA is available with {num_gpus} GPU(s).")

        torch.cuda.set_device(0)

        # Gets the name of the current GPU
        current_gpu_name = torch.cuda.get_device_name(0)
        print(f"Using GPU: {current_gpu_name}")

        # Sets the device for PyTorch tensors to the GPU
        device = torch.device("cuda")
    else:
        print("CUDA is not available. Using CPU.")
        # If CUDA is not available, set the device to CPU
        device = torch.device("cpu")


    DATA_PATH =os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets/')
    print(DATA_PATH)
    if os.path.isfile(DATA_PATH + 'data_train_40_hits.pickle') == False or os.path.isfile(DATA_PATH + 'data_test_40_hits.pickle') == False or os.path.isfile(DATA_PATH + 'data_val_40_hits.pickle') == False:

        print("some of the datasets were not created, creating the datasets")

        DATA_FILES = [
            "batch_1.parquet",
            #"batch_2.parquet",
            ]

        combined_data = pd.DataFrame()
        combined_res = pd.DataFrame()

        for data_file in DATA_FILES:
            try:

                dataframe = sample_loader(flag='dataset')

                targets_df = sample_loader(flag='targets')
            
            except OSError as e:
                print(f"dataset not found: {e}")
                pass


            logger.info("creating the pandas dataframe")

            dataframe_final = dataset_skimmer(dataframe)

            dataframe_final1 = padding_function(dataframe_final)

            dataframe_final3 = unstacker(dataframe_final1)

            logger.debug(dataframe_final3)

            logger.info("pandas dataframe have been unstacked")

            targets_dataframe = targets_definer(dataframe_final3, targets_df)
            
            logger.info("targets have been defined")

            combined_data = pd.concat([combined_data, dataframe_final3], ignore_index=False)

            combined_res = pd.concat([combined_res, targets_dataframe], ignore_index=False)

        logger.debug(combined_data)
        logger.debug(combined_res)

        logger.info("creating the model")

        logger.info("splitting the dataset")

        #splits the dataset into training and test 
        X_train, X_temp, Y_train, Y_temp = train_test_split(
            combined_data, combined_res, test_size=0.7, random_state=42
        )

        X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42)

        logger.debug("printing tensor shape")
        logger.debug(f"X_train tensor: {X_train.shape}")
        logger.debug(f"Y_train tensor: {Y_train.shape}")

        logger.info("creating the train torch tensor")

        if parameters.use_sliced_tensor:

            torch_tensor_train = tensor_creator(X_train, Y_train, label="train")
        else: 
            torch_tensor_train = tensor_creator(X_train, Y_train)

        logger.info("creating the validation tensor")

        if parameters.use_sliced_tensor:

            torch_tensor_val = tensor_creator(X_val, Y_val, label="val")
        else:
            torch_tensor_val = tensor_creator(X_val, Y_val)

        logger.info("creating the test torch tensor")
        if parameters.use_sliced_tensor:

            torch_tensor_test = tensor_creator(X_test, Y_test, label="test")
        else:
            torch_tensor_test = tensor_creator(X_test, Y_test)
        
        with open(DATA_PATH +'data_train_40_hits.pickle', 'xb') as output_train:
            pickle.dump(torch_tensor_train, output_train)

        with open(DATA_PATH + 'data_val_40_hits.pickle', 'xb') as output_val:
            pickle.dump(torch_tensor_val, output_val)
        
        with open(DATA_PATH + 'data_test_40_hits.pickle', 'xb') as output_test:
            pickle.dump(torch_tensor_test, output_test)
        
        print("created the datasets, proceding with the training")

    print("opening the datasets")

    with open(DATA_PATH + 'data_train_40_hits.pickle', 'rb') as data_train:
        
        tensor_train = pickle.load(data_train)
    if not parameters.optimal_hyperparameters_found:
        
        logger.info(f"optimal_hyperparameters_found set to {parameters.optimal_hyperparameters_found}, creating the tensor")
        with open(DATA_PATH + 'data_val_40_hits.pickle', 'rb') as data_val:
            tensor_val_or_test = pickle.load(data_val)
    else: 

        logger.info(f"optimal_hyperparameters_found set to {parameters.optimal_hyperparameters_found}, creating the tensor")
        with open(DATA_PATH + 'data_test_40_hits.pickle', 'rb') as data_test:
        
            tensor_val_or_test = pickle.load(data_test)

    logger.info("creating the model")

    model = model_creator()

    logger.info("creating the data tensors")

    dataset_train = dataset_creator(tensor_train)
    dataset_val_or_test = dataset_creator(tensor_val_or_test)
    logger.info("starting the training")

    train_losses, val_or_test_losses, train_rmses, val_or_test_rmses, best_lr = training_function(model, dataset_train, dataset_val_or_test)

    if not parameters.debug_value:
        loss_plots(train_losses, val_or_test_losses, train_rmses, val_or_test_rmses, best_lr)
    else:
        single_epoch_batch_loss_plots(train_losses, val_or_test_losses, train_rmses, val_or_test_rmses)

