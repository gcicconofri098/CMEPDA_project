import pandas as pd

def targets_definer(df_final):
    """
        creates a dataframe that contains the targets for each event
    Args:
        df_final (pandas Dataframe): feature dataframe from which the event IDs are taken

    Returns:
        pandas Dataframe: dataframe with azimuth and zenith for each event
    """
    res = pd.read_parquet("/scratchnvme/cicco/cmepda/train_meta.parquet")

    #the dataset contains information on all the datasets, 
    # so targets for the events considered need to be extracted

    # gets the list of event IDs

    events = df_final.index.get_level_values(0).unique()

    #takes only the targets for the events present in the feature dataframe
    res1 = res[res.event_id.isin(events)]

    res1 = res1.sort_index()

    #drops unnecessary columns
    res1 = res1.drop(
        labels=["first_pulse_index", "last_pulse_index", "batch_id"], axis=1
    )
    print("printing targets")
    print(res1)
    return res1

