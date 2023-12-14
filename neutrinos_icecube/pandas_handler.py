
'''This module handles the creation and processing of the pandas DataFrames.

Returns:

    (pandas.DataFrame, pandas.DataFrame): pandas.DataFrame of the features and the targets.
'''

import pandas as pd
import numpy as np

import parameters

from logging_conf import setup_logging

logger = setup_logging('dataframe_creation')

def dataset_skimmer(df):

    """ Prepares the dataset for padding operation.

    Arguments:

        df (pandas Dataframe): contains information on the event hits


    Returns:
    
        pandas Dataframe: contains the skimmed hits for the events, with information on the geometry
    """
    
    #loads the csv that contains information on the detector as a pandas dataframe

    geometry = pd.read_csv("/scratchnvme/cicco/cmepda/sensor_geometry.csv")

    logger.debug(geometry) 

    # the flag "auxiliary" takes into account information from the MC about the goodness of the hit
    df = df[
        df["auxiliary"] == False
    ]  

    df = df.drop(labels="auxiliary", axis=1)

    df_with_geom = df.merge(geometry, how="left", on="sensor_id").reset_index(
        drop=True
    )  # merges the two feature datasets

    # changes the type of two columns
    df_with_geom["event_id"].astype(np.int32)
    df_with_geom["charge"].astype(np.float16)

    # sort the events by charge and drop hits where the same sensor lit,
    # keeping the hit with the highest value of charge

    df_with_geom2 = (df_with_geom.sort_values("charge", ascending=False)
                                .drop_duplicates(["event_id", "sensor_id"])
                            )  # keep the sorting on the charge

    # add a counter of hits per event, and drops hits after the 25th one

    df_with_geom2["n_counter"] = df_with_geom2.groupby("event_id").cumcount()

    df_with_geom2 = df_with_geom2[df_with_geom2.n_counter < parameters.n_hits]
    
    logger.debug("printing the dataframe after the skim function")
    logger.debug(df_with_geom2)
    logger.debug(df_with_geom2.shape)

    return df_with_geom2

def padding_function(df_with_geom):

    """ Adds a zero-padding to take into account the different number of hits per event

    Args:
        df_with_geom (pandas Dataframe): dataframe with feature information

    Returns:
        pandas Dataframe: dataframe with the same number of hit for each event
    """

    # compute the number of hits per event
    maxima = df_with_geom.groupby("event_id")["n_counter"].max().values

    logger.debug(maxima)

    # find the number of rows to be added during the padding

    n_counter_1 = np.where(maxima > (parameters.n_hits -1), maxima, (parameters.n_hits -1))
    diff = np.array([n_counter_1 - maxima])

    #set a multi-index on the dataframe
    df_with_geom.set_index(["event_id", "n_counter"], inplace=True)

    n_rows = np.sum(diff)

    #take the array of event IDs
    ev_ids = np.unique(df_with_geom.index.get_level_values(0).values)

    logger.debug(ev_ids.shape)

    zeros = np.zeros((n_rows, 6), dtype=np.int32)
    
    logger.debug(zeros)

    #reshape the arrays to the correct shape
    diff_reshaped = diff.flatten()
    ev_ids_reshaped = np.reshape(ev_ids, (len(ev_ids), 1))

    logger.debug(ev_ids_reshaped.shape)

    #create a new index witht the events IDs to be used on the padded dataframe
    new_index = np.repeat(ev_ids_reshaped, diff_reshaped)
    logger.debug(df_with_geom)
    # creates a dataframe filled with zeros to be cancatenated to the data dataframe
    pad_df = pd.DataFrame(
        zeros, index=new_index, columns=df_with_geom.columns
    ).reset_index(drop=False)
    #renames the column of the indexes and drops the old one
    pad_df["event_id"] = pad_df["index"]
    pad_df = pad_df.drop(labels=["index"], axis=1)

    #creates the hit counter for the padded dataframe and sets event_id and n_counter as multi-index
    pad_df["n_counter"] = (
        pad_df.groupby("event_id").cumcount()
        + df_with_geom.groupby("event_id").cumcount().max()
        + 1
    )
    pad_df = pad_df.set_index(["event_id", "n_counter"])

    # concatenates the two dataframes, and group the hits by event id

    df_final = pd.concat([df_with_geom, pad_df])

    df_final = df_final.sort_index()
    df_final = df_final.reset_index(drop=False)

    #creates a new index that counts all the hits in an event and drops unnecessary columns

    df_final["counter"] = df_final.groupby("event_id").cumcount()

    df_final = df_final.set_index(["event_id", "counter"])

    #drops unnecessary columns
    df_final = df_final.drop(labels=["n_counter", "sensor_id"], axis=1)
    logger.debug(df_final)
    logger.debug(df_final.shape)
    return df_final

def unstacker(df_final):

    """ Creates a dataframe where each row contains one event
    Args:
        df_final (pandas Dataframe): dataframe containing one hit per row

    Returns:
        pandas Dataframe: dataframe containing one event per row
    """

    logger.debug(df_final)

    # unstack the dataset on the counter level of index, so that all the hits per event are set in a single row
    df_final1 = df_final.unstack()

    logger.debug("printing unstacked shape")
    logger.debug(df_final1.shape)

    return df_final1

def targets_definer(df_final, targets):
    
    """ Creates a dataframe that contains the targets for each event
    Args:
        df_final (pandas Dataframe): feature dataframe from which the event IDs are taken

    Returns:
        pandas Dataframe: dataframe with azimuth and zenith for each event
    """

    #the dataset contains information on all the datasets, 
    # so targets for the events considered need to be extracted

    # gets the list of event IDs

    events = df_final.index.get_level_values(0).unique()

    #takes only the targets for the events present in the feature dataframe
    targets1 = targets[targets.event_id.isin(events)]

    targets1 = targets1.sort_index()

    logger.debug(targets1)    

    #drops unnecessary columns
    targets1 = targets1.drop(
        labels=["first_pulse_index", "last_pulse_index", "batch_id"], axis=1
    )
    logger.debug("printing targets")
    logger.debug(targets1)
    return targets1