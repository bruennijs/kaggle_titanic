import numpy as np

from pandas import DataFrame, Series
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import OrdinalEncoder, StandardScaler


class Preprocessor:

    def process_categories(self, df: DataFrame) -> DataFrame:
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


    def process(self, df: DataFrame) -> tuple[DataFrame, DataFrame, Series]:
        """
        Preprocessing: Sex, Embarked -> Categorial
        Preprocessing: Drop Columns Name, Ticket, Cabin
        :param df:
        :return: tuple (X, y)
        """
        df_pp = self.drop(df)

        df_pp = self.process_categories(df_pp)


        # df_pp = self.outlier_detection(df_pp)

        # df_pp = df_pp.assign(Fare=StandardScaler().fit_transform(df_pp[["Fare"]].to_numpy()))

        return (df_pp, df_pp.drop(axis=1, labels=["Survived"]), df_pp["Survived"])




