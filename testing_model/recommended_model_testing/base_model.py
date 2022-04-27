import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import necessary modules
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier # RANDOM FOREST
from sklearn import svm, tree # using the svm and tree classifers
import xgboost # special?
from sklearn import preprocessing
import pickle
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import LeavePOut
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import plot_confusion_matrix, \
    accuracy_score, \
    confusion_matrix, \
    make_scorer, \
    ConfusionMatrixDisplay, \
    RocCurveDisplay, roc_curve, \
    auc, \
    classification_report

master_path = "C:/Users/Ikechukwu Anude/Documents/vaccine-data-processing/testing_model/recommended_model_testing/testing_data/master_train_one_hot.csv"

# function to be able to save info
def describe_plot_save(df, path, name="default"):

    # send meta data to csv
    df.describe().to_csv(f'{path}/{name}-meta_stats.csv')
    # f'{path}/{name}-histogram_plot.png'
    df.hist(bins=100)
    plt.suptitle(f'{name} Histogram')
    plt.show()

def turn_orgs_to_bots(df):

    df[df['labels'] == 2] = 0
    return df

def get_rid_of_orgs(df):

    df = df[df['labels'] != 2]
    return df

def test_threshold(row, threshold):
    if row['CAP'] < threshold:
        return 1
    else:
        return 0 # else bot

# testing the matrix
def cap_threshold_test(threshold=0.95, debug=False):
    """
    Produces a Confusion Matrix to evaluate the efficiency of the cap threshold test as recommended in Botometer
    :param threshold: any account with a CAP above 0.95 will be considered a bot.
    :return:
    """
    # sort by type
    master_df = pd.read_csv(master_path)

    mdf = turn_orgs_to_bots(master_df)

    metadata = {

    }

    # instead of dropping the features, what we can do is get rid of the rows above a certain threshold
    #predicted_bots = mdf[mdf['CAP'] > threshold]
    mdf['predictions'] = mdf.apply(lambda row: test_threshold(row, threshold), axis=1)
    correct_predictions = mdf[mdf['predictions'] == mdf['labels']]
    actual_humans = mdf[mdf['labels'] == 1].reset_index(drop=True)
    actual_bots = mdf[mdf['labels'] == 0].reset_index(drop=True)

    # TP, FP, FN, TN
    true_positive = actual_humans[actual_humans['labels'] == actual_humans['predictions']]
    false_positive = actual_bots[actual_bots['labels'] != actual_bots['predictions']]
    false_negative = actual_humans[actual_humans['labels'] != actual_humans['predictions']]
    true_negative = actual_bots[actual_bots['labels'] == actual_bots['predictions']]


    total_amt = len(mdf)
    correct_pred_amt = len(correct_predictions)
    overall_accuracy = np.round(correct_pred_amt/total_amt, 4)

    if debug:
        print("Total Rows: " + str(total_amt))
        print("Correct Predictions: " + str(correct_pred_amt))
        print("Overall Accuracy: " + str(overall_accuracy * 100) + "%")
        print(f'True Positive: {len(true_positive)}')
        print(f'False Positive: {len(false_positive)}')
        print(f'True Negative: {len(true_negative)}')
        print(f'False Negative: {len(false_negative)}')


    metadata = {
        "total_rows": total_amt,
        "correct_predictions": correct_pred_amt,
        "overall_accuracy": overall_accuracy * 100,
        "tp": len(true_positive),
        "fp": len(false_positive),
        "tn": len(true_negative),
        "fn": len(false_negative)
    }

    print(f"Evaluated for threshold: {threshold}")
    return metadata

def plot_accuracy(metatable):
    # plot the overall accuracy against the possible thresholds
    plt.title("Thresholds vs Overall Accuracy")
    x = metatable.keys()
    y = [metatable[i]["overall_accuracy"] for i in metatable.keys()]

    plt.plot(x, y)
    plt.savefig('Thresholds vs Overall Accuracy')

def plot_confusion_matrix_rates(metatable):
    title = "Thresholds vs Confusion Matrix Categories"
    # plot the overall accuracy against the possible thresholds
    plt.title(title)
    x = metatable.keys()
    tp_rate= [metatable[i]["tp"] for i in metatable.keys()]
    fp_rate = [metatable[i]["fp"] for i in metatable.keys()]
    tn_rate = [metatable[i]["tn"] for i in metatable.keys()]
    fn_rate = [metatable[i]["fn"] for i in metatable.keys()]

    plt.plot(x, tp_rate, label="True Positive")
    plt.plot(x, fp_rate, label="False Positive")
    plt.plot(x, tn_rate, label="True Negative")
    plt.plot(x, fn_rate, label="False Negative")

    plt.legend()
    plt.savefig(title)

def validate_threshold_method(num_thresholds = 50):
    possible_thresholds = np.linspace(0.5, 1, num_thresholds)

    # get the metadata and put into secondary dictionary
    metadata_list = {}
    best = (0, 0)
    for threshold in possible_thresholds:
        metadata_list[threshold] = cap_threshold_test(threshold=threshold)

        # find the max accuracy
        if metadata_list[threshold]['overall_accuracy'] > best[0]:
            best = (metadata_list[threshold]['overall_accuracy'], threshold)

    print(f"Best (Accuracy, Threshold): {best}")


    plot_accuracy(metatable=metadata_list)

    # should be able to toggle this one and off
    #plot_confusion_matrix_rates(metatable=metadata_list)


def main():


    pass

main()