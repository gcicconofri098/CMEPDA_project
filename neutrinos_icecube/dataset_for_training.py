from torch.utils.data import Dataset

def dataset_creator(train_tensor, test_tensor):
    """
    Creates a dataset from the Data tensor and creates a Dataset object

    Args:
        train_tensor (Data object): Data object for the training 
        test_tensor (Data object): Data object for the test

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
        
    
    custom_dataset_train = MyDataset(train_tensor)
    custom_dataset_test = MyDataset(test_tensor)

    return custom_dataset_train, custom_dataset_test
