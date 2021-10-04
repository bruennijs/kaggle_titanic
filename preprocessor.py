import string

import numpy as np

from pandas import DataFrame, concat
from sklearn.base import TransformerMixin
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder


class Preprocessor(TransformerMixin):

    def fit_transform(self, X, y=None, **fit_params):
        """
        Preprocessing: Sex, Embarked -> Categorial
        Preprocessing: Drop Columns Name, Ticket, Cabin
        :param df:
        :param drop_columns_onehot_encoded drop columns that are one hot encoded
        :return: tuple (X, y)
        """
        df_pp = self.drop(X)

        df_pp = self.ordinal_categories(df_pp)

        for col in ["Sex", "Parch", "Pclass", "SibSp", "Embarked"]:
            df_pp = self.onehot_categories(df=df_pp, column=col)
            if fit_params.get('drop_columns_onehot_encoded'):
                df_pp = df_pp.drop(col, axis=1)

        # df_pp = self.outlier_detection(df_pp)

        return (df_pp, df_pp.drop(axis=1, labels=["Survived"]), df_pp["Survived"])

    def ordinal_categories(self, df: DataFrame) -> DataFrame:
        """
        Preprocessing: Transform object columns to categories:
        Sex, Embarked
        :param df:
        :return:
        """
        encoder = OrdinalEncoder(dtype=np.int)
        np_encoded_cat = encoder.fit_transform(df[['Sex', 'Embarked']])
        df_pp: DataFrame = df.assign(Sex=np_encoded_cat[:,0]) \
            .assign(Embarked=np_encoded_cat[:,1])
        return df_pp

    def onehot_categories(self, df: DataFrame, column: string) -> DataFrame:
        encoder = OneHotEncoder(sparse=False)
        np_encoded: np.array = encoder.fit_transform(df[[column]])
        feature_names: np.array = encoder.get_feature_names_out()
        df_encoded = DataFrame(data=np_encoded, columns=feature_names.tolist(), dtype=bool, index=df.index)
        return concat([df, df_encoded], axis=1)

    def drop(self, df: DataFrame) -> DataFrame:
        """
         Preprocessing: Drop Columns
        :param df:
        :return:
        """
        return df\
            .drop(axis=1, labels=["PassengerId", "Name", "Ticket", "Cabin"])\
            .dropna()

    def outlier_detection(self, df: DataFrame) -> DataFrame:
        lof = LocalOutlierFactor(n_neighbors=20)
        lof_mask = lof.fit_predict(df[["Fare"]])

        return df.iloc[lof_mask != -1]





