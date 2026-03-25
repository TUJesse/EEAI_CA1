import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from Config import *
import random
seed =0
random.seed(seed)
np.random.seed(seed)

class Data():
    def __init__(self,
                 X: np.ndarray,
                 df: pd.DataFrame) -> None:

        X_DL = df[Config.TICKET_SUMMARY] + ' ' + df[Config.INTERACTION_CONTENT]
        X_DL = X_DL.to_numpy()
        y = df.y.to_numpy()

        #ref: https://www.geeksforgeeks.org/machine-learning/an-introduction-to-multilabel-classification/
        y_two_chain = np.asarray(df[Config.CHAIN_TWO_COLS])
        y_three_chain = np.asarray(df[Config.FORMATTED_TYPE_COLS])


        self.X_train, self.X_test, self.y_train, self.y_test= train_test_split(X, y, test_size=0.3, random_state=42)
        self.y = y
        self.embeddings = X


        self.two_X_train, self.two_X_test, self.two_y_train, self.two_y_test= train_test_split(X, y_two_chain, test_size=0.3, random_state=42)
        self.two_y = y_two_chain

        self.three_X_train, self.three_X_test, self.three_y_train, self.three_y_test = train_test_split(X, y_three_chain, test_size=0.3, random_state=42)
        self.three_y = y_three_chain

    def get_type(self):
        return  [self.y, self.two_y, self.three_y]
    def get_X_train(self):
        return  [self.X_train, self.two_X_train, self.three_X_train]
    def get_X_test(self):
        return  [self.X_test, self.two_X_test, self.three_X_test]
    def get_type_y_train(self):
        return  [self.y_train, self.two_y_train, self.three_y_train]
    def get_type_y_test(self):
        return  [self.y_test, self.two_y_test, self.three_y_test]
    def get_train_df(self):
        return  self.train_df
    def get_embeddings(self):
        return  self.embeddings
    def get_type_test_df(self):
        return  self.test_df
    def get_X_DL_test(self):
        return self.X_DL_test
    def get_X_DL_train(self):
        return self.X_DL_train
    def get_classification_type(self):
        return ["y2", "y2 + y3", "y2 + y3 + y4"]

