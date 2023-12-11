import math
import pandas as pd
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split

import numpy as np

from angular_distance_loss import angular_dist_score
from pandas_handler import dataset_skimmer, padding_function, unstacker, targets_definer
from model_creation import model_creator
from tensor_creation import tensor_creator
from training_module import training_function
from plots.plots_loss_and_rmse import single_batch_loss_plots, loss_plots
from dataset_for_training import dataset_creator
import parameters as parameters
from logging_conf import setup_logging

logger = setup_logging('main_log')


torch.set_num_threads(20)


#pylint:disable = invalid-name

if __name__ == "__main__":

    DATA_PATH = "/scratchnvme/cicco/cmepda/"
    DATA_FILES = [
        "batch_1.parquet",
        "batch_2.parquet",
        # "batch_10.parquet",
        # "batch_11.parquet",
        # "batch_100.parquet",
        # "batch_101.parquet",
    ]  # , 'batch_102.parquet', 'batch_103.parquet']



    combined_data = pd.DataFrame()
    combined_res = pd.DataFrame()

    for data_file in DATA_FILES:
        dataframe = pd.read_parquet(DATA_PATH + data_file).reset_index()

        targets = pd.read_parquet("/scratchnvme/cicco/cmepda/train_meta.parquet")


        dataframe_final = dataset_skimmer(dataframe)

        dataframe_final1 = padding_function(dataframe_final)

        targets = targets_definer(dataframe_final1, targets)

        logger.info("unstacking")

        dataframe_final3 = unstacker(dataframe_final1)
        logger.debug(dataframe_final3)



        combined_data = pd.concat([combined_data, dataframe_final3], ignore_index=False)

        combined_res = pd.concat([combined_res, targets], ignore_index=False)

    logger.debug(combined_data)
    logger.debug(combined_res)

    logger.info("creating the model")

    logger.info("splitting the dataset")

    #splits the dataset into training and test 
    X_train, X_test, Y_train, Y_test = train_test_split(
        combined_data, combined_res, test_size=0.3, random_state=42
    )

    logger.debug("printing tensor shape")
    logger.debug(f"X_train tensor: {X_train.shape}")
    logger.debug(f"Y_train tensor: {Y_train.shape}")

    logger.info("creating the train torch tensor")

    tensor_train = tensor_creator(X_train, Y_train, "train")

    logger.info("creating the test torch tensor")

    tensor_test = tensor_creator(X_test, Y_test, "test")
    logger.info("creating the model")

    model = model_creator()

    dataset_train, dataset_test = dataset_creator(tensor_train, tensor_test)

    print("starting the training")

    train_losses, test_losses, train_rmses, test_rmses = training_function(model, dataset_train, dataset_test)

    if not parameters.debug_value:
        loss_plots(train_losses, test_losses, train_rmses, test_rmses)
    else:
        single_batch_loss_plots(train_losses, test_losses, train_rmses, test_rmses)

