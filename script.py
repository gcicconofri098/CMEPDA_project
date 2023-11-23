# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import gc
from tensorflow import math
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
import keras.backend as K
import sys
import os

tf.compat.v1.enable_eager_execution()


import logging

# Set up logging
K.set_session(
    K.tf.compat.v1.Session(
        config=K.tf.compat.v1.ConfigProto(
            intra_op_parallelism_threads=10, inter_op_parallelism_threads=10
        )
    )
)


def angular_dist_score(y_true, y_pred):
    az_pred = y_pred[:, 0]
    az_true = y_true[:, 0]
    zen_true = y_true[:, 1]
    zen_pred = y_pred[:, 1]

    # Check for non-finite values in input data
    non_finite_mask_az_true = ~tf.math.is_finite(az_true)
    non_finite_mask_zen_true = ~tf.math.is_finite(zen_true)
    non_finite_mask_az_pred = ~tf.math.is_finite(az_pred)
    non_finite_mask_zen_pred = ~tf.math.is_finite(zen_pred)

    # Combine non-finite masks for all input tensors
    non_finite_mask = (
        non_finite_mask_az_true
        | non_finite_mask_zen_true
        | non_finite_mask_az_pred
        | non_finite_mask_zen_pred
    )

    # Apply the mask to filter out non-finite values
    az_true = tf.boolean_mask(az_true, ~non_finite_mask)
    zen_true = tf.boolean_mask(zen_true, ~non_finite_mask)
    az_pred = tf.boolean_mask(az_pred, ~non_finite_mask)
    zen_pred = tf.boolean_mask(zen_pred, ~non_finite_mask)

    # Continue with the rest of the calculations with filtered tensors
    # ...

    # Pre-compute all sine and cosine values
    sa1 = tf.sin(az_true)
    ca1 = tf.cos(az_true)
    sz1 = tf.sin(zen_true)
    cz1 = tf.cos(zen_true)

    sa2 = tf.sin(az_pred)
    ca2 = tf.cos(az_pred)
    sz2 = tf.sin(zen_pred)
    cz2 = tf.cos(zen_pred)

    # Scalar product of the two cartesian vectors (x = sz*ca, y = sz*sa, z = cz)
    scalar_prod = sz1 * sz2 * (ca1 * ca2 + sa1 * sa2) + (cz1 * cz2)

    # Scalar product of two unit vectors is always between -1 and 1
    # Clip to avoid numerical instability
    scalar_prod = tf.clip_by_value(scalar_prod, -1.0, 1.0)

    # Convert back to an angle (in radians)
    return tf.reduce_mean(tf.abs(tf.acos(scalar_prod)))


# %% [code] {"execution":{"iopub.status.busy":"2023-07-16T13:52:48.646392Z","iopub.execute_input":"2023-07-16T13:52:48.646691Z","iopub.status.idle":"2023-07-16T13:52:48.653524Z","shell.execute_reply.started":"2023-07-16T13:52:48.646662Z","shell.execute_reply":"2023-07-16T13:52:48.652109Z"}}
def variables_definition(geom):
    # geom['sensor_azimuth'] = np.arctan(geom['y']/geom['x']).astype(np.float32)
    # geom['sensor_zenith'] = np.arctan(np.sqrt(geom['x']**2 + geom['y']**2)/geom['z']).astype(np.float32)

    return geom


# %% [code] {"execution":{"iopub.status.busy":"2023-07-16T13:52:48.654882Z","iopub.execute_input":"2023-07-16T13:52:48.655667Z","iopub.status.idle":"2023-07-16T13:52:48.668759Z","shell.execute_reply.started":"2023-07-16T13:52:48.655618Z","shell.execute_reply":"2023-07-16T13:52:48.667076Z"}}
def dataset_skimmer(df, geom):
    df = df[
        df["auxiliary"] == False
    ]  # auxiliary label takes into account the quality of the hits

    df = df.drop(labels="auxiliary", axis=1)

    df["charge"].astype(np.float16())

    df_with_geom = df.merge(geom, how="left", on="sensor_id").reset_index(drop=True)

    del df

    # df_with_geom = df_with_geom.drop(['x', 'y', 'z'], axis = 1)
    df_with_geom["event_id"].astype(np.int32)
    # drop hits where the same sensor lit, keeping the hit with the highest value of charge

    df_with_geom = df_with_geom.sort_values("charge", ascending=False).drop_duplicates(
        ["event_id", "sensor_id"]
    )  # keep the sorting on the charge

    df_with_geom2 = df_with_geom[df_with_geom.charge > 25]

    # add a counter of hits per event, and drops hits after the 20th one

    df_with_geom2["n_counter"] = df_with_geom2.groupby("event_id").cumcount()

    mean = df_with_geom2["n_counter"].mean()

    # print("mean number of rows is:", mean)

    df_with_geom2 = df_with_geom2[df_with_geom2.n_counter < 15]

    time_0 = df_with_geom2.groupby("event_id")["time"].min().values
    # print(time_0)
    # df_with_geom2.head(15)
    return df_with_geom2, time_0


# %% [code] {"execution":{"iopub.status.busy":"2023-07-16T13:52:48.670987Z","iopub.execute_input":"2023-07-16T13:52:48.671405Z","iopub.status.idle":"2023-07-16T13:52:48.688311Z","shell.execute_reply.started":"2023-07-16T13:52:48.671374Z","shell.execute_reply":"2023-07-16T13:52:48.687294Z"}}
def padding_function(df_with_geom):
    # find the number of hits per event

    maxima = df_with_geom.groupby("event_id")["n_counter"].max().values

    # find time 0 per event

    # print(df_with_geom)

    # find the number of rows to be added during the padding

    n_counter_1 = np.where(maxima > 14, maxima, 14)
    diff = np.array([n_counter_1 - maxima])
    df_with_geom.set_index(["event_id", "n_counter"], inplace=True)
    # print(diff.shape)
    # sys.stdout.flush()

    n_rows = np.sum(diff)

    ev_ids = np.unique(df_with_geom.index.get_level_values(0).values)
    # print("ev_ids", ev_ids.shape)
    # sys.stdout.flush()

    zeros = np.zeros((n_rows, 6), dtype=np.int32)

    diff_reshaped = diff.flatten()
    print("len diff", len(diff_reshaped))
    print("shape diff", diff_reshaped.shape)
    sys.stdout.flush()
    ev_ids_reshaped = np.reshape(ev_ids, (len(ev_ids), 1))

    print("len ev", len(ev_ids_reshaped))
    print("shape ev", ev_ids_reshaped.shape)

    new_index = np.repeat(ev_ids_reshaped, diff_reshaped)

    # print("new_index:", new_index)
    sys.stdout.flush()

    # print(df_with_geom.columns)
    # sys.stdout.flush()
    # creates a dataframe filled with zeros to be cancatenated to the data dataframe

    pad_df = pd.DataFrame(
        zeros, index=new_index, columns=df_with_geom.columns
    ).reset_index(drop=False)
    pad_df["event_id"] = pad_df["index"]
    pad_df = pad_df.drop(labels=["index"], axis=1)
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

    del df_with_geom, pad_df
    # create a new index that counts all the hits in an event and drops the old counter and the sensor id

    df_final["counter"] = df_final.groupby("event_id").cumcount()
    # print(df_final)
    # sys.stdout.flush()

    df_final = df_final.set_index(["event_id", "counter"])

    df_final = df_final.drop(labels=["n_counter", "sensor_id"], axis=1)
    print("printing df_final")
    # print(df_final)

    return df_final


# %% [code] {"execution":{"iopub.status.busy":"2023-07-16T13:52:48.689330Z","iopub.execute_input":"2023-07-16T13:52:48.690267Z","iopub.status.idle":"2023-07-16T13:52:48.706932Z","shell.execute_reply.started":"2023-07-16T13:52:48.690237Z","shell.execute_reply":"2023-07-16T13:52:48.706069Z"}}
def dataset_preprocesser(df_final):
    # shifts the padded hits to avoid overlap with data hits

    # mask_charge = df_final['charge'] == 0
    # df_final.loc[mask_charge, 'charge'] = -1

    # mask_x = df_final['x'] == 0
    # df_final.loc[mask_x, 'x'] = -2

    # mask_y = df_final['y'] == 0
    # df_final.loc[mask_y, 'y'] = -2

    # mask_z = df_final['z'] == 0
    # df_final.loc[mask_z, 'z'] = -2

    # possibly add more preprocess steps

    return df_final


# %% [code] {"execution":{"iopub.status.busy":"2023-07-16T13:52:48.708244Z","iopub.execute_input":"2023-07-16T13:52:48.709122Z","iopub.status.idle":"2023-07-16T13:52:48.720913Z","shell.execute_reply.started":"2023-07-16T13:52:48.709067Z","shell.execute_reply":"2023-07-16T13:52:48.719875Z"}}
def targets_definer(df_final):
    """_summary_
        questa funczione fa cazzate
    Args:
        df_final (_type_): _description_

    Returns:
        _type_: _description_
    """
    res = pd.read_parquet("/scratchnvme/cicco/cmepda/train_meta.parquet")

    # because the feature dataset contains information on all the events, results for the single batch need to be extracted
    # this is a big problem memory-wise, because it takes most of the available memory

    # get the list of event ids

    events = df_final.index.get_level_values(0).unique()
    res1 = res[res.event_id.isin(events)]

    res1 = res1.sort_index()
    print(res1)
    sys.stdout.flush()
    del res
    return res1


# %% [code] {"execution":{"iopub.status.busy":"2023-07-16T13:52:48.722707Z","iopub.execute_input":"2023-07-16T13:52:48.723513Z","iopub.status.idle":"2023-07-16T13:52:48.738352Z","shell.execute_reply.started":"2023-07-16T13:52:48.723471Z","shell.execute_reply":"2023-07-16T13:52:48.737372Z"}}
def unstacker(df_final):
    # df_final = df_final.drop(labels = ['sensor_zenith'], axis = 1)

    # unstack the dataset on the counter level of index, so that all the hits per event are set in a single row
    df_final1 = df_final.unstack()

    # print(df_final1)

    # now we must reorder the columns so that the first 4 are for particle 0, the next 4 for particle 1, etc.
    # we can do this by sorting the columns by the second level (the particle number)
    # df_final1 = df_final1.sort_index(axis=1, level=1)
    # reset the value of the index

    # df_final2 = df_final1.reset_index(drop = False)

    # df_final1['time'] = df_final1['time'].sub(time_0, axis= 0)

    # print(df_final1.head())

    # print(df_final2)

    # df_final2 = df_final2.sort_values(by = df_final2.index.get_level_values(1), axis = 1)

    return df_final1


# %% [code]


# %% [code] {"execution":{"iopub.status.busy":"2023-07-16T13:52:48.739509Z","iopub.execute_input":"2023-07-16T13:52:48.741460Z","iopub.status.idle":"2023-07-16T13:52:48.752581Z","shell.execute_reply.started":"2023-07-16T13:52:48.741420Z","shell.execute_reply":"2023-07-16T13:52:48.751280Z"}}
def model_compiler():
    class Attention_block(tf.keras.layers.Layer):
        def __init__(self, **kwargs: dict):
            super().__init__()
            self.multihead = tf.keras.layers.MultiHeadAttention(**kwargs)
            self.norm_layer = tf.keras.layers.LayerNormalization()
            self.add = tf.keras.layers.Add()

    class Global_attention(Attention_block):
        def call(self, x: tf.Tensor):
            output = self.multihead(query=x, value=x, key=x)
            x = self.add([x, output])
            x = self.norm_layer(x)

            return x

    class SelfAttentionPooling(tf.keras.layers.Layer):
        def __init__(self, **kwargs: dict):
            super().__init__()

        def build(self, input_shape):
            feature_dim = input_shape[-1]
            self.dense = tf.keras.layers.Dense(
                units=feature_dim, use_bias=False, input_shape=(feature_dim,)
            )

        def call(self, x: tf.Tensor):
            attention_weights = tf.nn.softmax(((self.dense(x))))

            pooled = tf.reduce_sum(attention_weights * x, axis=1)
            print(tf.shape(pooled))
            return pooled

    input_layer = layers.Input(shape=[5, 15])

    batch_norm = layers.BatchNormalization()(input_layer)

    dense_1 = layers.Dense(units=512, activation="relu")(batch_norm)

    dropout1 = layers.Dropout(0.1)(dense_1)

    dense_2 = layers.Dense(units=512, activation="relu")(dropout1)

    dropout2 = layers.Dropout(0.1)(dense_2)

    dense_12 = layers.Dense(units=512, activation="relu")(dropout2)

    dropout12 = layers.Dropout(0.1)(dense_12)

    dense_23 = layers.Dense(units=512, activation="relu")(dropout12)

    dropout22 = layers.Dropout(0.1)(dense_23)

    dense_22 = layers.Dense(units=512, activation="relu")(dropout22)

    dropout3 = layers.Dropout(0.1)(dense_22)

    attention = Global_attention(num_heads=4, key_dim=512)(dropout3)

    dense_3 = layers.Dense(units=512, activation="relu")(attention)

    dropout4 = layers.Dropout(0.1)(dense_3)

    # attention_2 = Global_attention(num_heads=5, key_dim=512)(dropout4)

    dense_34 = layers.Dense(units=512, activation="relu")(dropout4)

    dropout42 = layers.Dropout(0.1)(dense_34)

    pooling = SelfAttentionPooling()(dropout42)

    dense_4 = layers.Dense(units=512, activation="relu")(pooling)

    dropout5 = layers.Dropout(0.1)(dense_4)

    dense_35 = layers.Dense(units=512, activation="relu")(dropout5)

    dropout42 = layers.Dropout(0.1)(dense_35)

    dense_5 = layers.Dense(units=512, activation="relu")(dropout42)

    dropout6 = layers.Dropout(0.1)(dense_5)

    pre_output = layers.Add()([dropout6, pooling])
    pre_output2 = layers.LayerNormalization()(pre_output)

    output_1 = layers.Dense(units=2, activation="linear")(pre_output2)

    model = Model(inputs=input_layer, outputs=output_1)

    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0001),
        loss=angular_dist_score,
        metrics=[tf.keras.metrics.RootMeanSquaredError()],
    )

    print(model.summary())

    return model


# %% [code] {"execution":{"iopub.status.busy":"2023-07-16T13:52:48.754280Z","iopub.execute_input":"2023-07-16T13:52:48.754838Z","iopub.status.idle":"2023-07-16T13:52:48.771893Z","shell.execute_reply.started":"2023-07-16T13:52:48.754801Z","shell.execute_reply":"2023-07-16T13:52:48.770853Z"}}
def variables_definer(res1):
    # X = df_final1.loc[:,['charge','sensor_azimuth', 'sensor_zenith']] #variables to be taken from the batch
    Y = res1.loc[
        :, ["azimuth", "zenith"]
    ]  # target for the training to be taken from the metadata file

    # print("input shape in variables definer",X.shape)
    # plt.hist(Y['azimuth'].values, bins = 80)

    # plt.savefig('azimuth.png')
    # plt.close()

    # plt.hist(Y['zenith'].values, bins = 80)

    # plt.savefig('zenith.png')
    # plt.close()
    return Y


# %% [code] {"execution":{"iopub.status.busy":"2023-07-16T13:52:48.774898Z","iopub.execute_input":"2023-07-16T13:52:48.775591Z","iopub.status.idle":"2023-07-16T13:52:48.785084Z","shell.execute_reply.started":"2023-07-16T13:52:48.775556Z","shell.execute_reply":"2023-07-16T13:52:48.783943Z"}}
def training(X, Y, model):
    callback = keras.callbacks.EarlyStopping(
        monitor="loss", patience=2, min_delta=0.0001
    )

    history = model.fit(
        X, Y, batch_size=256, epochs=100, validation_split=0.3
    )  # , callbacks=[callback],verbose=1)
    history_df = pd.DataFrame(history.history)
    print(history.history)
    # Start the plot at epoch 3. You can change this to get a different view.
    history_df.loc[3:, ["loss", "val_loss"]].plot()
    plt.savefig("loss_3_rmsprop_512_ch_0_6_15_hits_6_files.png")
    plt.close()

    history_df.loc[
        3:, ["root_mean_squared_error", "val_root_mean_squared_error"]
    ].plot()
    plt.savefig("rmse_3_rmsprop_512_ch_0_6_15_hits_6_files.png")


# %% [code] {"execution":{"iopub.status.busy":"2023-07-16T13:52:48.786587Z","iopub.execute_input":"2023-07-16T13:52:48.786946Z"}}


if __name__ == "__main__":
    data_path = "/scratchnvme/cicco/cmepda/"
    data_files = [
        "batch_1.parquet",
        "batch_2.parquet",
        "batch_10.parquet",
        "batch_11.parquet",
        "batch_100.parquet",
        "batch_101.parquet",
    ]  # , 'batch_102.parquet', 'batch_103.parquet']
    geometry = pd.read_csv("/scratchnvme/cicco/cmepda/sensor_geometry.csv")
    combined_data = pd.DataFrame()
    combined_res = pd.DataFrame()

    for data_file in data_files:
        dataframe = pd.read_parquet(data_path + data_file).reset_index()
        geometry = variables_definition(geometry)
        dataframe_with_geometry, start_time = dataset_skimmer(dataframe, geometry)
        del dataframe
        gc.collect()
        dataframe_final = padding_function(dataframe_with_geometry)
        del dataframe_with_geometry
        gc.collect()
        dataframe_final1 = dataset_preprocesser(dataframe_final)
        del dataframe_final
        gc.collect()
        # dataframe_final2 = dataframe_final1.sample(frac=1)
        targets = targets_definer(dataframe_final1)

        print("unstacking")

        dataframe_final3 = unstacker(dataframe_final1)
        # del dataframe_final1
        gc.collect()

        combined_data = pd.concat([combined_data, dataframe_final3], ignore_index=True)

        combined_res = pd.concat([combined_res, targets], ignore_index=True)

    # X, Y = variables_definer(combined_data, combined_res)
    Y = variables_definer(combined_res)
    print("variables are defined")

    print(Y)

    print(f"input has {len(combined_data)} rows")
    print(f"target has {len(combined_res)} rows")

    print("reshaping")
    print(combined_data)
    reshaped_data = np.reshape(combined_data.values, (len(combined_data.values), 5, 15))
    # reshaped_data2 = reshaped_data.transpose(0, 2, 1)

    print(reshaped_data)

    # input_data = reshaped_data.values
    model_compiled = model_compiler()

    print("model is compiled")

    # combined_data.to_pickle('combined.pkl')

    print("Starting training")
    training(reshaped_data, Y, model_compiled)


# %%
