from model.SGD import SGD
from model.randomforest import RandomForest
from model.adaboost import AdaBoost
from model.voting import Voting
from model.hist_gb import Hist_GB
from model.random_trees_ensembling import RandomTreesEmbedding
from utils import write_to_file
from Config import *


def model_predict(data, df):
    results = []
    print("RandomForest")
    write_to_file(Config.OUTPUT_FILE, "RandomForest")
    model = RandomForest("RandomForest", data.get_embeddings())

    for X_trains, y_trains, X_tests, y_tests, classification_types in zip(data.get_X_train(), data.get_type_y_train(), data.get_X_test(), data.get_type_y_test(), data.get_classification_type()):
        model.train(X_trains, y_trains)
        model.predict(X_tests)
        model.print_results(X_tests, y_tests, classification_types)

    
    print("\nHist_GB")
    write_to_file(Config.OUTPUT_FILE, "\nHist_GB")
    model = Hist_GB("Hist_GB", data.get_embeddings())

    for X_trains, y_trains, X_tests, y_tests, classification_types in zip(data.get_X_train(), data.get_type_y_train(), data.get_X_test(), data.get_type_y_test(), data.get_classification_type()):
        model.train(X_trains, y_trains)
        model.predict(X_tests)
        model.print_results(X_tests, y_tests, classification_types)

    
    print("\nSGD")
    write_to_file(Config.OUTPUT_FILE, "\nSGD")
    model = SGD("SGD", data.get_embeddings())

    for X_trains, y_trains, X_tests, y_tests, classification_types in zip(data.get_X_train(), data.get_type_y_train(), data.get_X_test(), data.get_type_y_test(), data.get_classification_type()):
        model.train(X_trains, y_trains)
        model.predict(X_tests)
        model.print_results(X_tests, y_tests, classification_types)

    
    print("\nAdaBoost")
    write_to_file(Config.OUTPUT_FILE, "\nAdaBoost")
    model = AdaBoost("AdaBoost", data.get_embeddings())

    for X_trains, y_trains, X_tests, y_tests, classification_types in zip(data.get_X_train(), data.get_type_y_train(), data.get_X_test(), data.get_type_y_test(), data.get_classification_type()):
        model.train(X_trains, y_trains)
        model.predict(X_tests)
        model.print_results(X_tests, y_tests, classification_types)


    print("\nVoting")
    write_to_file(Config.OUTPUT_FILE, "\nVoting")
    model = Voting("Voting", data.get_embeddings())

    for X_trains, y_trains, X_tests, y_tests, classification_types in zip(data.get_X_train(), data.get_type_y_train(), data.get_X_test(), data.get_type_y_test(), data.get_classification_type()):
        model.train(X_trains, y_trains)
        model.predict(X_tests)
        model.print_results(X_tests, y_tests, classification_types)


    print("\nRandomTreesEmbedding")
    write_to_file(Config.OUTPUT_FILE, "\nRandomTreesEmbedding")
    model = RandomTreesEmbedding("RandomTreesEmbedding", data.get_embeddings())

    for X_trains, y_trains, X_tests, y_tests, classification_types in zip(data.get_X_train(), data.get_type_y_train(), data.get_X_test(), data.get_type_y_test(), data.get_classification_type()):
        model.train(X_trains, y_trains)
        model.predict(X_tests)
        model.print_results(X_tests, y_tests, classification_types)


def model_evaluate(model, data):
    model.print_results(data)