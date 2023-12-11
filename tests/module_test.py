import unittest
import logging
import pandas as pd
import numpy as np
import sys


sys.path.append('../neutrinos_icecube/')

from pandas_handler import dataset_skimmer, padding_function, unstacker, targets_definer
import parameters

IS_TEST_LOCAL = False

if not IS_TEST_LOCAL:
    pandas_dataset = pd.read_parquet("/gpfs/ddn/cms/user/cicco/miniconda3/CMEPDA/datasets/subset.parquet").reset_index(drop= True)
    targets = pd.read_parquet("/gpfs/ddn/cms/user/cicco/miniconda3/CMEPDA/datasets/res_subset.parquet")

else:
    pandas_dataset = pd.read_parquet("/scratchnvme/cicco/cmepda/batch_1.parquet").reset_index(drop = True)
    targets = pd.read_parquet("/scratchnvme/cicco/cmepda/train_meta.parquet")

class PandasTestModule(unittest.TestCase):
    """Class that inherits from the unittest module that checks the correct shape of the pandas dataframes

    Args:
        unittest (_type_): _description_
    """
    
    def setup(self):
        logging.disable(logging.CRITICAL)
        
    def test_dataframe_shape(self):
        """Checks that the pandas dataframe after the skimming has the correct shape. #! RISCRIVI
           The expected shape is, at the end of the skimming, (n_rows,6)
           The number of rows depends on the cut on the maximum number of hits requested
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

    def initialisation(self):


if __name__ == '__main__':
    unittest.main()