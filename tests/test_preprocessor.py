import unittest
from unittest import TestCase

import numpy as np
import pandas as pd

from pandas import DataFrame
from preprocessor import Preprocessor


class TestPreprocessor(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.df = pd.read_csv("./../input/train.csv")
        super().setUpClass()

    def setUp(self) -> None:
        super().setUp()
        self.cut = Preprocessor(drop_original_columns=False, engineer_title=True, drop_nans=False, engineer_deck=True)
        return

    def test_title(self):
        df = TestPreprocessor.df

        X_y, X, y  = self.cut.fit_transform(df)
        # index by passengerid
        X_pid = DataFrame(data=X.to_numpy(), index=X['PassengerId'].to_numpy(), columns=X.columns)


        # Then
        self.assertIn('Title', X_y.columns)
        self.assertIn('Title_cat', X_y.columns)
        titles_passengerid_1_to_8 = X_pid.loc[1:8]['Title'].to_list()
        self.assertSequenceEqual(titles_passengerid_1_to_8, ['Mr', 'Mrs', 'Miss', 'Mrs', 'Mr', 'Mr', 'Mr', 'Master'])



if __name__ == '__main__':
    unittest.main()
