from pandas import DataFrame
from sklearn.preprocessing import OrdinalEncoder

class Preprocessor :
    def process_categories(self, df: DataFrame) -> DataFrame:
        """
        Preprocessing: Transform object columns to categories:
        Sex, Embarked
        :param df:
        :return:
        """
        encoder = OrdinalEncoder()
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
        return df.drop(axis=1, labels=["Name", "Ticket", "Cabin"]).dropna()

    def process(self, df: DataFrame) -> DataFrame:
        df_pp = self.process_categories(df)
        return self.drop(df_pp)



