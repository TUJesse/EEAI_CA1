import numpy as np
import pandas as pd
from model.base import BaseModel
from sklearn.linear_model import SGDClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from Config import *
from utils import write_to_file

class SGD(BaseModel):
    def __init__(self,
                 model_name: str,
                 embeddings: np.ndarray) -> None:
        super(SGD, self).__init__()
        self.model_name = model_name
        self.embeddings = embeddings
        self.has_been_called = False

        self.mdl = SGDClassifier(random_state=0, class_weight='balanced', n_jobs=-1)
        self.predictions = None
        self.data_transform()

    
    def train(self, X_train, y_train) -> None:
        self.mdl = self.mdl.fit(X_train, y_train)

    def predict(self, X_test: pd.Series):
        predictions = self.mdl.predict(X_test)
        self.predictions = predictions

    def print_results(self, X_test, y_test, classification_name: str):

        scoring = self.mdl.score(X_test, y_test)
        self.scoring = scoring

        print(f"{classification_name}: \nAccuracy = {scoring}")
        write_to_file(Config.OUTPUT_FILE, f"\n{classification_name}: \nAccuracy = {scoring}")
        self.been_called()
    
    def been_called(self):
        if self.has_been_called == False:
            self.has_been_called = True
            self.mdl = MultiOutputClassifier(SGDClassifier(random_state=0, class_weight='balanced', n_jobs=-1))


    def data_transform(self) -> None:
        ...

