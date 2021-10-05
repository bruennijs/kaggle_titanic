import string

import numpy as np

from pandas import DataFrame, Series, concat
from pandas.core.groupby import SeriesGroupBy, DataFrameGroupBy
from sklearn.base import TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder


class Preprocessor(TransformerMixin):

    def __init__(self, drop_original_columns: bool = True) -> None:
        super().__init__()
        self.drop_original_columns = drop_original_columns

    def fit_transform(self, X, y=None, **fit_params):
        """
        Preprocessing: Sex, Embarked -> Categorial
        Preprocessing: Drop Columns Name, Ticket, Cabin
        :param df:
        :param drop_columns_onehot_encoded drop columns that are one hot encoded
        :return: tuple (X, y)
        """

        # todo: fare & age -> set median on nan

        # feature engineering
        df_pp = self.engineer_deck(X)
        if self.drop_original_columns:
            df_pp = df_pp.drop('Cabin', axis=1)

        # engineer Deck_cat
        df_pp = self.ordinal_categories(df_pp, ['Deck'])
        if self.drop_original_columns:
            df_pp = df_pp.drop('Deck', axis=1)

            # deck missing indicator
        df_pp = self.engineer_deck_missing(df_pp, column_name='Deck_cat_missing')

        # calculate median Deck no for each class to fill nan of column Deck
        df_pp = self.fill_missing_deck(df_pp)


        df_pp = df_pp.dropna()

        df_pp = self.ordinal_categories(df_pp, ['Sex', 'Embarked'])
        if self.drop_original_columns:
            df_pp = df_pp.drop(['Sex', 'Embarked'], axis=1)


        # for col in ["Sex_cat", "Parch", "Pclass", "SibSp", "Embarked_cat"]:
        #     df_pp = self.onehot_categories(df=df_pp, column=col)
        #     if self.drop_original_columns:
        #         df_pp = df_pp.drop(col, axis=1)



        df_pp = self.drop_unrelevant_columns(df_pp)

        # df_pp = self.outlier_detection(df_pp)

        return (df_pp, df_pp.drop(axis=1, labels=["Survived"]), df_pp["Survived"])

    def fill_missing_deck(self, df: DataFrame) -> DataFrame:

        pclass_group: DataFrameGroupBy = df.groupby(by='Pclass')
        pclass_to_deck_median: SeriesGroupBy = df.groupby(by='Pclass').Deck_cat.mean()

        def handle_pclass_group(group: DataFrame):
            def fill_deck(row: Series):
                if np.isnan(row['Deck_cat']):
                    row_tmp = row.copy()
                    row_tmp['Deck_cat'] = pclass_to_deck_median[row['Pclass']]
                    return row_tmp
                return row

            return group.apply(func=fill_deck, axis=1) # row-wise


        # group by Pclass WITH nans in Deck_cat and fill by joining with group Pclass->median
        df_transformed = pclass_group.apply(handle_pclass_group)

        return df_transformed.astype(dtype={'Deck_cat': 'int'})


    def ordinal_categories(self, df: DataFrame, cols: [string]) -> DataFrame:
        """
        Preprocessing: Transform object columns to categories and appends '_cat' to column name
        :param df:
        :return:
        """
        encoder = OrdinalEncoder(dtype=int)
        df_tmp = df.dropna()
        np_encoded_cat = encoder.fit_transform(df_tmp[cols])
        df_cat = DataFrame(data=np_encoded_cat, columns=["{}_cat".format(col) for col in cols], index=df_tmp.index)
        return concat([df, df_cat], axis=1)

    def onehot_categories(self, df: DataFrame, column: string) -> DataFrame:
        encoder = OneHotEncoder(sparse=False)
        np_encoded: np.array = encoder.fit_transform(df[[column]])
        feature_names: np.array = encoder.get_feature_names_out()
        df_encoded = DataFrame(data=np_encoded, columns=feature_names.tolist(), dtype=bool, index=df.index)
        return concat([df, df_encoded], axis=1)

    def drop_unrelevant_columns(self, df: DataFrame) -> DataFrame:
        """
         Preprocessing: Drop Columns
        :param df:
        :return:
        """
        return df\
            .drop(axis=1, labels=["PassengerId", "Name", "Ticket"])

    def outlier_detection(self, df: DataFrame) -> DataFrame:
        lof = LocalOutlierFactor(n_neighbors=20)
        lof_mask = lof.fit_predict(df[["Fare"]])

        return df.iloc[lof_mask != -1]

    def engineer_deck(self, df_pp: DataFrame) -> DataFrame:

        s_deck: Series = df_pp['Cabin'][df_pp['Cabin'].notna()]\
            .map(lambda c:c.strip())\
            .map(lambda c:c[:1])

        return df_pp.assign(Deck=s_deck)

    def engineer_deck_missing(self, df: DataFrame, column_name: string) -> DataFrame:
        """
        Adds column indicating whether Deck_cat has missing values
        :param df:
        :return:
        """
        return df.assign(**{'{}'.format(column_name): df['Deck_cat'].isna()})





