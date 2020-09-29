import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import tree
from sklearn import linear_model
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn import metrics
from sklearn import preprocessing
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, explained_variance_score, r2_score
import pickle
import os

def load_database_file():
    file = open('data\LX_categ_dataset_unfolded_database_array', 'rb')
    database = pickle.load(file)
    file.close()
    print('######################',type(database))
    print('######################', database.shape)
    return database

def load_target_file():
    file = open('data\LX_categ_dataset_unfolded_target_imdb_score_array', 'rb')
    target_imdb_score = pickle.load(file)
    file.close()
    print('######################',type(target_imdb_score))
    print('######################', target_imdb_score.shape)
    return target_imdb_score



def regression_scores(original_val, predicted_val, model_name):
    print("Regression Model Name: ", model_name)
    mean_abs_error = mean_absolute_error(original_val, predicted_val)
    mean_sqr_error = mean_squared_error(original_val, predicted_val)
    median_abs_error = median_absolute_error(original_val, predicted_val)
    explained_var_score = explained_variance_score(original_val, predicted_val)
    r2__score = r2_score(original_val, predicted_val)

    file1 = open('reports\\regression_scores.txt', 'w')
    file1.write("\n")
    file1.write("Regression Scores(train_test_split):")
    file1.write("\n")
    file1.write("Mean Absolute Error    :"+str(mean_abs_error))
    file1.write("\n")
    file1.write("Mean Squared Error     :"+str(mean_sqr_error))
    file1.write("\n")
    file1.write("Median Absolute Error  :"+str(median_abs_error))
    file1.write("\n")
    file1.write("Explained Var Score    :"+str(explained_var_score))
    file1.write("\n")
    file1.write("R^2 Score              :"+str(r2__score))
    file1.write("\n")
    file1.write("\n")
    file1.write("\n")
    file1.close()

# simple task
def inverse_scaling(scaled_val):
    min_max_scaler = preprocessing.MinMaxScaler()
    unscaled_val = min_max_scaler.inverse_transform(scaled_val)
    return unscaled_val


def roundval(value):
    return value.round()


def to_millions(value):
    return value / 10000000

def to_tenth(value):
    return value * 10


# evaluation
# plotting actual vs predicted for all data
def prediction_performance_plot(original_val, predicted_val, model_name, start, end, n, plot_type, prediction_type):
    # inverse transform and convert to millions
    ##original_val = to_millions(inverse_scaling(original_val))
    ##predicted_val = to_millions(inverse_scaling(predicted_val))

    original_val =  to_tenth(original_val)
    predicted_val = to_tenth(predicted_val)

    print("\n")
    plt.title(
        "\n" + prediction_type + " Prediction Performance using " + model_name + "(Actual VS Predicted)" + plot_type + "\n")
    if plot_type == "all":
        plt.plot(original_val, c='g', label="Actual")
        plt.plot(predicted_val, c='b', label="Prediction")
    if plot_type == "seq":
        plt.plot(original_val[start: end + 1], c='g', label="Actual")
        plt.plot(predicted_val[start: end + 1], c='b', label="Prediction")
    if plot_type == "random":
        original_val_list = []
        predicted_val_list = []
        for k in range(n):
            i = random.randint(0, len(predicted_val) - 1)
            original_val_list.append(original_val[i])
            predicted_val_list.append(predicted_val[i])
        plt.plot(original_val_list, c='g', label="Actual")
        plt.plot(predicted_val_list, c='b', label="Prediction")
    plt.legend(["Actual", "Predicted"], loc='center left', bbox_to_anchor=(1, 0.8))
    plt.ylabel('Prediction (In Millions)', fontsize=14)
    plt.grid()
    ##plt.show()
    print('################################################################')
    print('SAVING IMAGES',plot_type)
    print('################################################################')
    if plot_type == 'seq':
        os.remove("reports\Prediction_Performance_Sequence.png")
        plt.savefig('reports\Prediction_Performance_Sequence.png')
        print('COMPLETED SAVING################################################################')
    else:
        os.remove("reports\Prediction_Performance_Random.png")
        plt.savefig('reports\Prediction_Performance_Random.png')
        print('COMPLETED SAVING################################################################')


# printing actual vs predicted in a range
def print_original_vs_predicted(original_val, predicted_val, i, j, n, model_name, print_type, prediction_type):
    # inverse transform and convert to millions
    ##original_val = to_millions(inverse_scaling(original_val))
    ##predicted_val = to_millions(inverse_scaling(predicted_val))

    original_val =  to_tenth(original_val)
    predicted_val = to_tenth(predicted_val)


    file2 = open('reports\\regression_original_vs_predicted_scores.txt', 'w')
    file2.write("\n")

    print("\n" + prediction_type + " Comparision of Actual VS Predicted" + print_type + "\n")
    if print_type == "seq":
        if j < len(predicted_val):
            for k in range(i, j + 1):
                file2.write("Actual" +str(prediction_type) + " : "+str(original_val[k]))
                file2.write("\n")
                file2.write("Predicted " + str(prediction_type) + " : " + str(predicted_val[k]))
                file2.write("\n")
    if print_type == "random":
        for k in range(n):
            i = random.randint(0, len(predicted_val) - 1)
            file2.write("Actual" +str(prediction_type) + " : "+str(original_val[i]))
            file2.write("\n")
            file2.write("Predicted " + str(prediction_type) + " : " + str(predicted_val[i]))
            file2.write("\n")
    file2.write("\n\n")
    file2.close()


# plotting actual vs predicted in a randomly using a bar chart
def bar_plot_original_vs_predicted_rand(original_val, predicted_val, n, model_name, pred_type):
    # inverse transform and convert to millions
    ##original_val = to_millions(inverse_scaling(original_val))
    ##predicted_val = to_millions(inverse_scaling(predicted_val))

    original_val =  to_tenth(original_val)
    predicted_val = to_tenth(predicted_val)


    original_val_list = []
    predicted_val_list = []
    for k in range(n):
        i = random.randint(0, len(predicted_val) - 1)
        original_val_list.append(original_val[i])
        predicted_val_list.append(predicted_val[i])

    original_val_df = pd.DataFrame(original_val_list)
    predicted_val_df = pd.DataFrame(predicted_val_list)

    actual_vs_predicted = pd.concat([original_val_df, predicted_val_df], axis=1)

    actual_vs_predicted.plot(kind="bar", fontsize=12, color=['g', 'b'], width=0.7)
    plt.title(
        "\nUsing Categorical and Numerical Features\n" + model_name + " : Actual " + pred_type + "VS Predicted " + pred_type + "(Random)")
    plt.ylabel('Gross (In Millions)', fontsize=14)
    plt.ylabel('Gross (In M', fontsize=14)
    plt.xticks([])
    plt.legend(["Actual ", "Predicted"], loc='center left', bbox_to_anchor=(1, 0.8))
    plt.grid()
    ##plt.show()
    plt.savefig('reports\Barplot_original_vs_predicted_rand.png')


# Plot features
# calculate mean
def meanbyfeature(data, feature_name, meanby_feature):
    mean_data = data.groupby(feature_name).mean()
    mean = mean_data[meanby_feature]
    mean_sort = mean.sort(meanby_feature, inplace=False, ascending=False)
    return mean_sort


def plot(data, kind, title, n_rows):
    plt.title(title, fontsize=15)
    data[:n_rows].plot(kind=kind)
    ###plt.show()


def show_features(database):
    print("\n",
          "--------------------------------------------------------------------------------------------------------")
    database.info()
    print("\n",
          "--------------------------------------------------------------------------------------------------------")


def load_prediction_explainability_metrics(movietitle,predictedvalue):
    target_imdb_score = load_target_file()
    database = load_database_file()
    movieiloc = database[database.original_title == movietitle].iloc[-1]
    print('------->', movieiloc.name)
    target_data_actual = target_imdb_score[movieiloc.name]

    test_target=target_data_actual
    predicted_gross = predictedvalue
    model_name = "Decision Tree Regression"
    pred_type= "(IMDB Score Prediction)"

    print('test_target     =',test_target)
    print('predicted_gross =', predicted_gross)

    print('test_target.shape    =',test_target.shape)
    print('predicted_gross.shape=', predicted_gross.shape)

    print('test_target.type     =',type(test_target))
    print('predicted_gross.type =', type(predicted_gross))

    print('Regression Scoring starting')
    regression_scores(test_target, predicted_gross, model_name)
    print('Regression Scoring Completed')

    predicted_gross=predicted_gross.reshape(-1,1)
    print('Plotting starting')
    prediction_performance_plot(test_target, predicted_gross, model_name, 200, 250, 0, "seq", pred_type)
    print('Plotting Completed - 1')

    prediction_performance_plot(test_target, predicted_gross, model_name, 0, 0, 100, "random", pred_type)
    print('Plotting Completed - 2')

    print_original_vs_predicted(test_target, predicted_gross, 0, 0, 10, model_name, "seq", pred_type)
    print('Plotting Completed - 3')

    bar_plot_original_vs_predicted_rand(test_target, predicted_gross, 20, model_name, pred_type)
    print('regr_without_cross_validation_train_test_perform_plot Completed - 3')


    return 1
