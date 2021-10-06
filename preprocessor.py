from typing import Sequence

import re
import numpy as np
import pandas as pd

from pandas import DataFrame, Series, concat
from pandas.core.groupby import SeriesGroupBy, DataFrameGroupBy
from sklearn.base import TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.utils._testing import MinimalTransformer


class Preprocessor(TransformerMixin, MinimalTransformer):

    def __init__(self, drop_original_columns: bool = True, engineer_title: bool = True, drop_nans: bool=True, engineer_deck: bool = True) -> None:
        super().__init__()
        self.engineer_deck_ = engineer_deck
        self.drop_nans = drop_nans
        self.engineer_title_ = engineer_title
        self.drop_original_columns = drop_original_columns
        self.title_list_ = ['Mrs', 'Mr', 'Master', 'Miss', 'Major', 'Rev',
                                      'Dr', 'Ms', 'Mlle','Col', 'Capt', 'Mme', 'Countess',
                                      'Don', 'Jonkheer']

    def fit(self, X, y=None, **fit_params) -> object:
        return self

    def transform(self, X, y=None):
        """
        Preprocessing: Sex, Embarked -> Categorial
        Preprocessing: Drop Columns Name, Ticket, Cabin
        :param df:
        :param drop_columns_onehot_encoded drop columns that are one hot encoded
        :return: tuple (X, y)
        """

        # todo: fare & age -> set median on nan

        # feature engineering
        # DECK
        df_pp = X
        if self.engineer_deck_:
            df_pp = self.engineer_deck(X, 'Deck')
            # engineer Deck_cat
            df_pp = self.ordinal_categories(df_pp, ['Deck'])
            if self.drop_original_columns:
                df_pp = df_pp.drop('Deck', axis=1)
            # deck missing indicator
            df_pp = self.engineer_deck_missing_indicator(df=df_pp, column_name='Deck_cat_missing')
            # calculate median Deck no for each class to fill nan of column Deck
            df_pp = self.fill_missing_deck(df_pp)


        # AGE: fill nan of age with median age
        df_pp = self.fill_age(df_pp)

        # TITLE
        if self.engineer_title_:
            df_pp = self.engineer_title(df_pp, 'Title')
            df_pp = self.ordinal_categories(df_pp, ['Title'])
            if self.drop_original_columns:
                df_pp = df_pp.drop(['Title'], axis=1)

        df_pp = self.ordinal_categories(df_pp, ['Sex', 'Embarked'])


        # for col in ["Sex_cat", "Parch", "Pclass", "SibSp", "Embarked_cat"]:
        #     df_pp = self.onehot_categories(df=df_pp, column=col)
        #     if self.drop_original_columns:
        #         df_pp = df_pp.drop(col, axis=1)

        if self.drop_original_columns:
            df_pp = df_pp.drop(axis=1, labels=["PassengerId", "Name", "Ticket", 'Cabin', 'Sex', 'Embarked'])

        if self.drop_nans:
            df_pp = df_pp.dropna()

        # df_pp = self.outlier_detection(df_pp)

        return (df_pp, df_pp.drop(axis=1, labels=["Survived"]), df_pp["Survived"])

    def fill_missing_deck(self, df: DataFrame) -> DataFrame:

        pclass_group: DataFrameGroupBy = df.groupby(by='Pclass')
        pclass_to_deck_median: Series = df.groupby(by='Pclass').Deck_cat.mean().astype(int)

        def handle_pclass_group(group: DataFrame):
            def fill_deck(row: Series) -> Series:
                if pd.isna(row['Deck_cat']):
                    row_tmp = row.copy()
                    row_tmp['Deck_cat'] = pclass_to_deck_median[row['Pclass']]
                    return row_tmp
                return row

            return group.apply(func=fill_deck, axis=1) # row-wise


        # group by Pclass WITH nans in Deck_cat and fill by joining with group Pclass->median
        df_transformed = pclass_group.apply(handle_pclass_group)

        return df_transformed.astype(dtype={'Deck_cat': 'int'})


    def ordinal_categories(self, data: DataFrame, cols: [str]) -> DataFrame:
        """
        Preprocessing: Transform object columns to categories and appends '_cat' to column name
        :param df:
        :return:
        """
        encoder = OrdinalEncoder(dtype=float)
        np_encoded_cat = encoder.fit_transform(data[cols])
        # DataFrame with dtype  pd.Int64Dtype() for having ints with pd.NA (works cause dropna is called later so no ps.NA values
        df_cat = DataFrame(data=np_encoded_cat, columns=["{}_cat".format(col) for col in cols], index=data.index, dtype=pd.Int64Dtype())
        return concat([data, df_cat], axis=1)

    def onehot_categories(self, df: DataFrame, column: str) -> DataFrame:
        encoder = OneHotEncoder(sparse=False)
        np_encoded: np.array = encoder.fit_transform(df[[column]])
        feature_names: np.array = encoder.get_feature_names_out()
        df_encoded = DataFrame(data=np_encoded, columns=feature_names.tolist(), dtype=bool, index=df.index)
        return concat([df, df_encoded], axis=1)

    def outlier_detection(self, df: DataFrame) -> DataFrame:
        lof = LocalOutlierFactor(n_neighbors=20)
        lof_mask = lof.fit_predict(df[["Fare"]])

        return df.iloc[lof_mask != -1]

    def engineer_deck(self, df_pp: DataFrame, column_name: str) -> DataFrame:

        s_deck: Series = df_pp['Cabin'][df_pp['Cabin'].notna()]\
            .map(lambda c:c.strip())\
            .map(lambda c:c[:1])

        return df_pp.assign(**{'{}'.format(column_name): s_deck})

    def engineer_deck_missing_indicator(self, df: DataFrame, column_name: str) -> DataFrame:
        """
        Adds column indicating whether Deck_cat has missing values
        :param df:
        :return:
        """
        return df.assign(**{'{}'.format(column_name): df['Deck_cat'].isna()})

    def fill_age(self, df: DataFrame) -> DataFrame:

        df_tmp = df.copy()
        df_tmp['Age'].fillna(df['Age'].median(), inplace=True)

        # imputer = SimpleImputer(strategy='median')
        # df_tmp['Age'] = imputer.fit_transform(df_tmp[['Age']])

        return df_tmp

    def engineer_title(self, df: DataFrame, column_name: str) -> DataFrame:
        this = self
        def find_first_element_in_titlelist(elements: Sequence[str]) -> str:
            found_elements: list[str] = [i for i in elements if i in this.title_list_]
            if len(found_elements) > 0:
                return found_elements[0]
            else:
                return np.nan

        df_tmp = df.copy()
        df_tmp[column_name] = df['Name']\
            .map(lambda name:re.split('\s|\,|\.', name))\
            .map(find_first_element_in_titlelist)
        return df_tmp
