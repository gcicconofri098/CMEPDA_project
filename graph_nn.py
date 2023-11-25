import torch
import torch_geometric
from torch_geometric.data import Data
from torch.utils.data import Dataset
from torch_cluster import knn_graph
from torch_geometric.nn import MessagePassing
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
import torch
import torch.nn as nn

def angular_dist_score(y_true, y_pred):
    az_pred = y_pred[:,0]
    az_true = y_true[0, :]
    zen_true = y_true[1,:]
    zen_pred = y_pred[:, 1]

    # Check for non-finite values in input data
    non_finite_mask_az_true = ~torch.isfinite(az_true)
    non_finite_mask_zen_true = ~torch.isfinite(zen_true)
    non_finite_mask_az_pred = ~torch.isfinite(az_pred)
    non_finite_mask_zen_pred = ~torch.isfinite(zen_pred)

    # Combine non-finite masks for all input tensors
    non_finite_mask = (non_finite_mask_az_true | non_finite_mask_zen_true |
                      non_finite_mask_az_pred | non_finite_mask_zen_pred)

    # Apply the mask to filter out non-finite values
    az_true = az_true[~non_finite_mask]
    zen_true = zen_true[~non_finite_mask]
    az_pred = az_pred[~non_finite_mask]
    zen_pred = zen_pred[~non_finite_mask]

    # Pre-compute all sine and cosine values
    sa1 = torch.sin(az_true)
    ca1 = torch.cos(az_true)
    sz1 = torch.sin(zen_true)
    cz1 = torch.cos(zen_true)
    
    sa2 = torch.sin(az_pred)
    ca2 = torch.cos(az_pred)
    sz2 = torch.sin(zen_pred)
    cz2 = torch.cos(zen_pred)

    # Scalar product of the two cartesian vectors (x = sz*ca, y = sz*sa, z = cz)
    scalar_prod = sz1 * sz2 * (ca1 * ca2 + sa1 * sa2) + (cz1 * cz2)
    
    # Scalar product of two unit vectors is always between -1 and 1
    # Clip to avoid numerical instability
    scalar_prod = torch.clamp(scalar_prod, -1.0, 1.0)
    
    # Convert back to an angle (in radians)
    return torch.mean(torch.abs(torch.acos(scalar_prod)))


def dataset_skimmer(df, geom):
    df = df[df['auxiliary']==False] #auxiliary label takes into account the quality of the hits

    df = df.drop(labels='auxiliary', axis=1)

    df['charge'].astype(np.float16())


    df_with_geom = df.merge(geom, how = 'left', on = 'sensor_id').reset_index(drop = True)
    
    del df
    
    #df_with_geom = df_with_geom.drop(['x', 'y', 'z'], axis = 1)
    df_with_geom['event_id'].astype(np.int32)
    #drop hits where the same sensor lit, keeping the hit with the highest value of charge 
    
    df_with_geom2 = df_with_geom.sort_values('charge', ascending = False).drop_duplicates(['event_id', 'sensor_id'])  #keep the sorting on the charge
    
    #df_with_geom2 = df_with_geom[df_with_geom.charge >25]


    #add a counter of hits per event, and drops hits after the 20th one
     
    df_with_geom2['n_counter'] = df_with_geom2.groupby('event_id').cumcount()

    mean = df_with_geom2['n_counter'].mean()

    #print("mean number of rows is:", mean)

    df_with_geom2  = df_with_geom2[df_with_geom2.n_counter<15]
    
    time_0 = df_with_geom2.groupby('event_id')['time'].min().values
    #print(time_0)
    #df_with_geom2.head(15)
    return df_with_geom2, time_0

# %% [code] {"execution":{"iopub.status.busy":"2023-07-16T13:52:48.670987Z","iopub.execute_input":"2023-07-16T13:52:48.671405Z","iopub.status.idle":"2023-07-16T13:52:48.688311Z","shell.execute_reply.started":"2023-07-16T13:52:48.671374Z","shell.execute_reply":"2023-07-16T13:52:48.687294Z"}}
def padding_function(df_with_geom):
    
    #find the number of hits per event
    
    maxima = df_with_geom.groupby('event_id')['n_counter'].max().values

    #find time 0 per event


    #print(df_with_geom)

    #find the number of rows to be added during the padding

    n_counter_1 = np.where(maxima > 14, maxima, 14)
    diff = np.array([n_counter_1 - maxima])
    df_with_geom.set_index(['event_id', 'n_counter'], inplace= True)
    #print(diff.shape)
    #sys.stdout.flush()

    n_rows = np.sum(diff)

    ev_ids = np.unique(df_with_geom.index.get_level_values(0).values)
    #print("ev_ids", ev_ids.shape)
    #sys.stdout.flush()

    zeros = np.zeros((n_rows, 6), dtype = np.int32)
    
    diff_reshaped = diff.flatten()
    print("len diff",len(diff_reshaped))
    print("shape diff", diff_reshaped.shape)
    sys.stdout.flush()
    ev_ids_reshaped = np.reshape(ev_ids, (len(ev_ids), 1))
 
    print("len ev",len(ev_ids_reshaped))
    print("shape ev", ev_ids_reshaped.shape)
    

    new_index = np.repeat(ev_ids_reshaped, diff_reshaped)

    #print("new_index:", new_index)
    sys.stdout.flush()

    #print(df_with_geom.columns)
    #sys.stdout.flush()
    #creates a dataframe filled with zeros to be cancatenated to the data dataframe
    
    pad_df = pd.DataFrame(zeros, index = new_index ,columns=df_with_geom.columns).reset_index(drop = False)
    pad_df['event_id'] = pad_df['index']
    pad_df = pad_df.drop(labels = ['index'], axis =1)
    pad_df['n_counter'] = pad_df.groupby('event_id').cumcount() + df_with_geom.groupby('event_id').cumcount().max() +1
    pad_df = pad_df.set_index(['event_id', 'n_counter'])
    
    #concatenates the two dataframes, and group the hits by event id

    df_final = pd.concat([df_with_geom, pad_df])
    
    df_final = df_final.sort_index()
    df_final = df_final.reset_index(drop= False)

    del df_with_geom, pad_df
    #create a new index that counts all the hits in an event and drops the old counter and the sensor id

    df_final['counter'] = df_final.groupby('event_id').cumcount()
    #print(df_final)
    #sys.stdout.flush()

    df_final = df_final.set_index(['event_id', 'counter'])

    df_final = df_final.drop(labels= ['n_counter', 'sensor_id'], axis = 1)
    print("printing df_final")
    #print(df_final)

    return df_final

# %% [code] {"execution":{"iopub.status.busy":"2023-07-16T13:52:48.689330Z","iopub.execute_input":"2023-07-16T13:52:48.690267Z","iopub.status.idle":"2023-07-16T13:52:48.706932Z","shell.execute_reply.started":"2023-07-16T13:52:48.690237Z","shell.execute_reply":"2023-07-16T13:52:48.706069Z"}}
def dataset_preprocesser(df_final):
    
    #shifts the padded hits to avoid overlap with data hits

    # mask_charge = df_final['charge'] == 0
    # df_final.loc[mask_charge, 'charge'] = -1

    # mask_x = df_final['x'] == 0
    # df_final.loc[mask_x, 'x'] = -2
    
    # mask_y = df_final['y'] == 0
    # df_final.loc[mask_y, 'y'] = -2
    
    # mask_z = df_final['z'] == 0
    # df_final.loc[mask_z, 'z'] = -2
    
    
    #possibly add more preprocess steps
    
    return df_final

# %% [code] {"execution":{"iopub.status.busy":"2023-07-16T13:52:48.708244Z","iopub.execute_input":"2023-07-16T13:52:48.709122Z","iopub.status.idle":"2023-07-16T13:52:48.720913Z","shell.execute_reply.started":"2023-07-16T13:52:48.709067Z","shell.execute_reply":"2023-07-16T13:52:48.719875Z"}}
def targets_definer(df_final):
    """_summary_
        questa funczione fa cose
    Args:
        df_final (_type_): _description_

    Returns:
        _type_: _description_
    """
    res = pd.read_parquet('/scratchnvme/cicco/cmepda/train_meta.parquet')

    #because the feature dataset contains information on all the events, results for the single batch need to be extracted 
    #this is a big problem memory-wise, because it takes most of the available memory 
    
    #get the list of event ids
    
    events = df_final.index.get_level_values(0).unique()
    res1  = res[res.event_id.isin(events)]

    res1 = res1.sort_index()
    res1 = res1.drop(labels = ['first_pulse_index', 'last_pulse_index', 'batch_id'], axis=1)

    print(res1)
    sys.stdout.flush()
    del res
    return res1

def unstacker(df_final):

    #df_final = df_final.drop(labels = ['sensor_zenith'], axis = 1)    

    #unstack the dataset on the counter level of index, so that all the hits per event are set in a single row
    df_final1 = df_final.unstack()

    #df_with_targets = df_final1.merge(targets, how = 'left', on = df_final1.index.get_level_values(0)).reset_index(drop = True)


    print(df_final1)

    # now we must reorder the columns so that the first 4 are for particle 0, the next 4 for particle 1, etc.
    # we can do this by sorting the columns by the second level (the particle number)
    #df_final1 = df_final1.sort_index(axis=1, level=1)
    #reset the value of the index
    
    # df_final2 = df_final1.reset_index(drop = False)
    
    #df_final1['time'] = df_final1['time'].sub(time_0, axis= 0)


    #print(df_final1.head())


    # print(df_final2)

    # df_final2 = df_final2.sort_values(by = df_final2.index.get_level_values(1), axis = 1)

    return df_final1

def model_creator():
    
    class DNNLayer(MessagePassing):
        def __init__(self, in_channels, out_channels):
            super().__init__(aggr = 'max')
            self.mlp = nn.Sequential(nn.Linear(2*in_channels, out_channels), nn.ReLU(), nn.Linear(out_channels, out_channels))
        
        print("test")

        def forward(self, h, edge_index):
            return self.propagate(edge_index, h=h)
        
        def message(self, h_j, h_i):
            print(h_i)
            
            input = torch.cat([h_i, h_j-h_i], dim = -1)
            return self.mlp(input)

        print("DNN layer created successfully")

    class Graph_Network(nn.Module):
        def __init__(self):
            super().__init__()
            N_features = 20

            self.f1 = DNNLayer(6, N_features)
            self.f2 = DNNLayer(N_features, N_features)
            self.f3 = DNNLayer(N_features, N_features)  
            self.output = nn.Linear(N_features, 2)
        
        def forward(self, data):
            x = data.x
            edge_index = data.edge_index
            print("shape before f1",x.shape)

            h = self.f1( h = x, edge_index = edge_index)
            print("shape after f1",h.shape)

            h = h.relu()
            print("shape after relu",h.shape)

            h = self.f2( h=h,  edge_index = edge_index)
            print("shape after f2",h.shape)

            h = h.relu()
            print("shape after relu",h.shape)

            h = self.f3( h=h,  edge_index = edge_index)
            print("shape after f3",h.shape)

            h = h.relu()
            print("shape after relu",h.shape)
            
            h = self.output(h)

            return h
    
    model = Graph_Network()
    print(model)

    return model




def tensor_creator(df, targets):

    unique_events = pd.unique(df.index.get_level_values(0))
    sliced_unique_events = unique_events[:2]
    print(sliced_unique_events)

    data_list = []



    # def minkowski_distance(x,y):
    #     spatial_distance = torch.norm(x[:3] - y[:3], p=2)  # Minkowski spatial distance
    #     temporal_distance = torch.abs(x[3] - y[3])  # Absolute temporal distance
    #     return (spatial_distance**2 + temporal_distance**2)**(1/2)

    for event_id in sliced_unique_events:
        # Extract data for the current event
        event_data = df[df.index.get_level_values(0) == event_id]
        event_targets = targets[targets['event_id'] == event_id]

        # Extract node features
        node_features = event_data[['charge', 'x', 'y', 'z', 'time']]
        node_targets = event_targets[['azimuth', 'zenith']]
        ##print(node_features)
        data = Data(x = torch.Tensor(node_features.values.reshape(5,-1).T), y = torch.Tensor(node_targets.values).reshape(-1,1))
        # Add the Data object to the list
        ##print(data.x)
        ##print(data.y)
        data_list.append(data)

        print(f"Event ID: {event_id}")
        print("Node Features Shape:", data.x.shape)
        print("Node Targets Shape:", data.y.shape)


        nNeighbors = 5
        data.edge_index = knn_graph(data.x, k=nNeighbors)

        print("Edge Index Shape (After):", data.edge_index.shape)

        ##print(data.edge_index)

        ##print(data)

        #print(data.edge_index.size(1))
        
        #print(data.edge_index)
        cluster_charge = torch.zeros(data.x.size(0), dtype=data.x.dtype, device=data.x.device)

        for i in range(data.edge_index.size(1)):
            #print(data.x[data.edge_index[1, i], 1])
            idx_0 = data.edge_index[0, i]
            idx_1 = data.edge_index[1, i]
            charge_value = data.x[idx_1, 0]
    
            cluster_charge[idx_0] += charge_value

        print(cluster_charge.shape)
        print(cluster_charge)
        data.x = torch.cat([data.x, cluster_charge.view(-1, 1)], dim=-1)

        print("Final Node Features Shape:", data.x.shape)
        print("Final Edge Index Shape:", data.edge_index.shape)


        print(f"finished event: {event_id}")
    print("info on data.x")
    print(data.x.shape)
    print("info on data.edge_index")
    print(data.edge_index.shape)
    # concatenated_data = Data(x=torch.cat([d.x for d in data_list], dim=0),
    #                      edge_index=torch.cat([d.edge_index for d in data_list], dim=1))
    # print(concatenated_data)


    return data_list


def training_function(model, dataset_train, dataset_test):


    print(dataset_train[0])
    print(dataset_test[0])
    class MyDataset(Dataset):
        def __init__(self, data_list):
            print("Initializing dataset")
            self.data_list = data_list
        
        def __len__(self):
            return len(self.data_list)
        
        def __getitem__(self, idx):
            
            data = self.data_list[idx]
            print("type from __getitem__", type(data))
            return data
        



    custom_dataset_train = MyDataset(dataset_train)
    custom_dataset_test = MyDataset(dataset_test)


    train_loader = DataLoader(custom_dataset_train, batch_size = 128, shuffle = False)
    test_loader = DataLoader(custom_dataset_test, batch_size = 128, shuffle = False)

    for batch in train_loader:
        print(type(batch))

    optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)


    def train(model, optimizer, loader):
        model.train()
        for data in loader:
            optimizer.zero_grad()
            
            output = model(data)
            loss = angular_dist_score(data.y, output)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        return total_loss / len(train_loader.dataset)

    def evaluate(model, loader):
        model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for data in loader:
                output = model(data.x, data.edge_index)
                loss = angular_dist_score(data.y, output)
                total_loss += loss.item()

        average_loss = total_loss / len(loader)
        return average_loss
    
    train_losses = []
    test_losses = []

    for epoch in range(1, 100):
        train_loss = train(model, optimizer, train_loader)
        test_loss = evaluate(model, test_loader)
        
        test_losses.append(test_loss)
        train_losses.append(train_loss)
        print(f'Epoch: {epoch:02d}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')
 
    plt.figure(figsize=(10,5))
    plt.title("Training and Validation Loss")
    plt.plot(test_losses,label="val")
    plt.plot(train_losses,label="train")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    plt.savefig("test_graph.png")


if __name__ == "__main__":
    data_path = "/scratchnvme/cicco/cmepda/"
    data_files = [
        "batch_1.parquet",
        # "batch_2.parquet",
        # "batch_10.parquet",
        # "batch_11.parquet",
        # "batch_100.parquet",
        # "batch_101.parquet",
    ]  # , 'batch_102.parquet', 'batch_103.parquet']
    geometry = pd.read_csv("/scratchnvme/cicco/cmepda/sensor_geometry.csv")
    combined_data = pd.DataFrame()
    combined_res = pd.DataFrame()

    for data_file in data_files:
        dataframe = pd.read_parquet(data_path + data_file).reset_index()
        dataframe_with_geometry, start_time = dataset_skimmer(dataframe, geometry)
        del dataframe

        dataframe_final = padding_function(dataframe_with_geometry)
        del dataframe_with_geometry

        dataframe_final1 = dataset_preprocesser(dataframe_final)
        del dataframe_final

        # dataframe_final2 = dataframe_final1.sample(frac=1)
        targets = targets_definer(dataframe_final1)

        print("unstacking")

        dataframe_final3 = unstacker(dataframe_final1)
        print(dataframe_final3)
        # del dataframe_final1


    model = model_creator()
    X_train, X_test, Y_train, Y_test = train_test_split(dataframe_final3, targets, test_size=0.3, random_state=None)

    print(X_train)
    #X_train_unstacked = X_train.stack().reset_index()

    #X_test_unstacked = X_test.stack().reset_index()


    print(X_train.iloc[0])
    # X_train = pd.melt(X_train, id_vars = ['event_id'], value_vars= ['x', 'y', 'z', 'time', 'charge'])

    # X_test = pd.melt(X_train, id_vars = ['event_id'], value_vars= ['x', 'y', 'z', 'time', 'charge'])



    data_train = tensor_creator(X_train, Y_train)
    data_test = tensor_creator(X_test, Y_test)
    training = training_function(model, data_train, data_test)

