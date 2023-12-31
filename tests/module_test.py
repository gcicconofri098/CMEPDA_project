import unittest
import logging
import pandas as pd
import numpy as np
import sys


sys.path.append('../neutrinos_icecube/')

from pandas_handler import dataset_skimmer, padding_function, unstacker, targets_definer
from tensor_creation import tensor_creator
from datasets.sample_loader import sample_loader, sample_loader_non_local_testing
import parameters

IS_TEST_LOCAL = False

if not IS_TEST_LOCAL:
    pandas_dataset = sample_loader_non_local_testing(flag='dataset').reset_index(drop= True)
    targets = sample_loader_non_local_testing(flag='targets')
else:
    pandas_dataset = sample_loader(flag='dataset').reset_index(drop = True)
    targets = sample_loader(flag='targets')


class PandasTestModule(unittest.TestCase):
    """Class that checks that the pandas DataFrames have the correct shape.
    """
    
    def setup(self):

        logging.disable(logging.CRITICAL)
        
    def test_dataframe_shape(self):
        """Checks that the pandas dataframes have the correct shape after the processing.
           The expected shape after the skimming is (n_rows,8).
           n_rows is not known, because it depends on the number of hits per event
           The expected shape after applying the padding is (expected_rows, 5), 
           where expected rows is the number of events times the number of hits set with the
           parameter n_hits.
           The expected shape after unstacking is (n_events, expected_features), 
           where expected_features is the number of features per hit times the number of hits.
           The expected shape for the target dataframe is (n_events, 3)

        """

        n_events = len(pd.unique(pandas_dataset.event_id))

        skimmed_dataset = dataset_skimmer(pandas_dataset)
        print("print skimmed dataset")
        print(skimmed_dataset)

        self.assertEqual(skimmed_dataset.shape[1], 8, "Failed test on skimmed dataset")

        padded_df = padding_function(skimmed_dataset)
        
        expected_n_rows = n_events * parameters.n_hits

        print(expected_n_rows)

        print(len(padded_df.index.get_level_values(0)))

        actual_rows =len(padded_df.index.get_level_values(0))

        self.assertEqual(actual_rows, expected_n_rows, "Failed test on padded dataset, dim0")
        self.assertEqual(padded_df.shape[1], 5,  "Failed test on padded dataset, dim1")

        unstacked_df = unstacker(padded_df)

        expected_columns = 5 * parameters.n_hits

        self.assertEqual(len(unstacked_df.index.get_level_values(0)), n_events,  "Failed test on unstacked dataset, dim0")
        self.assertEqual(unstacked_df.shape[1], expected_columns)

        needed_targets = targets_definer(unstacked_df, targets)

        self.assertEqual(len(needed_targets.index.get_level_values(0)), n_events, "Failed test on targets dataset, dim0")
        self.assertEqual(needed_targets.shape[1], 3, "Failed test on targets dataset, dim1")


class TorchTensorTestModule(unittest.TestCase):

    """Class that checks that the torch-geometric.Data is correctly created
    """

    def initialisation(self):
        """Initialises the pandas dataframes that will be used to test the module that creates the torch tensors
        """
        skimmed_dataset = dataset_skimmer(pandas_dataset)
        padded_df = padding_function(skimmed_dataset)
        unstacked_df = unstacker(padded_df)
        needed_targets = targets_definer(unstacked_df, targets)

        return unstacked_df, needed_targets
    
    def test_tensor_shape(self):
        """ Creates the list of torch_geometric.Data, then takes a casual element of the list and checks that
            the shapes are the ones expected and that the connections inside the graph are actually made.
            The expected shape for Data.x is (actual_n_hits, 6)
            where actual_n_hits is the number of hits per event without padding (maximum is n_hìts).
            The expected shape for Data.y is (1,2).
            For the edge indexes, only the request that the connection are made is done.
        """
        df, targets_df = self.initialisation()

        data_list = tensor_creator(df, targets_df)
        
        self.assertTrue(len(data_list)>0, "The list of data is empty")

        random_index = np.random.choice(len(data_list))
        
        self.assertTrue(len(data_list[random_index].edge_index) >0, "Failed to create connections in the knn_graph")

        self.assertEqual(data_list[random_index].x.shape[1], 6, "Failed test on data.x, dim1")

        self.assertEqual(data_list[random_index].y.shape, (1,2), "Failed test that checks the shape of the target tensor (1,2)")


if __name__ == '__main__':
    unittest.main()