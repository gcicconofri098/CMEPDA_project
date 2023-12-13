""" Module that handles the creation of the Datasets using the list of Data created with the module tensor_creation. """
from torch.utils.data import Dataset

def dataset_creator(train_tensor, test_tensor):

    """
    Creates a Dataset from the list of Data in input.

    Args:
        train_tensor (list of torch_geometric.Data): list of Data for the training 
        test_tensor (list of torch_geometric.Data): list of Data for the test

    Returns:
        custom_dataset_train (torch.utils.data.Dataset): Dataset for the training
        custom_dataset_test (torch.utils.data.Dataset): Dataset for the testing
    """

    class MyDataset(Dataset):

        """
        Custom Dataset for handling the Data. 
        It takes the list of Data and implements the custom
        Args:
            Dataset (torch.utils.data.Dataset): class for handling datasets
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
