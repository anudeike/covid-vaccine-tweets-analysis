import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import necessary modules
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, ConfusionMatrixDisplay
from math import sqrt
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier # RANDOM FOREST
from sklearn import svm, tree # using the svm and tree classifers
import pickle

master_path = "C:/Users/Ikechukwu Anude/Documents/vaccine-data-processing/testing_model/recommended_model_testing/testing_data/master_train_one_hot.csv"

models_test_paths = [r"C:\Users\Ikechukwu Anude\Documents\vaccine-data-processing\testing_model\model\NEW_XGB_Classifier_noStrat_op_whole.dat",
                     r"C:\Users\Ikechukwu Anude\Documents\vaccine-data-processing\testing_model\model\NEW_XGB_Default_Classifier_op_whole_z3_03.dat",
                     r"C:\Users\Ikechukwu Anude\Documents\vaccine-data-processing\testing_model\model\NEW_XGB_Default_Classifier_op_whole_z3.dat",
                     r"C:\Users\Ikechukwu Anude\Documents\vaccine-data-processing\testing_model\model\NEW_XGB_Default_Classifier_op_whole.dat",
                     r"C:\Users\Ikechukwu Anude\Documents\vaccine-data-processing\testing_model\model\Log_reg_model_whole.dat",
                     r"C:\Users\Ikechukwu Anude\Documents\vaccine-data-processing\testing_model\model\NEW_XGB_Default_Classifier_whole.dat",
                     r"C:\Users\Ikechukwu Anude\Documents\vaccine-data-processing\testing_model\model\NEW_XGB_Default_Classifier.dat",
                     r"C:\Users\Ikechukwu Anude\Documents\vaccine-data-processing\testing_model\model\XGB_Default_Classifier_im_3.dat",
                     r"C:\Users\Ikechukwu Anude\Documents\vaccine-data-processing\testing_model\model\NEW_XGB_Default_Classifier_2.dat"]
# function to be able to save info
def describe_plot_save(df, path, name="default"):

    # send meta data to csv
    df.describe().to_csv(f'{path}/{name}-meta_stats.csv')
    # f'{path}/{name}-histogram_plot.png'
    df.hist(bins=100)
    plt.suptitle(f'{name} Histogram')
    plt.show()

def turn_orgs_to_bots(df, inv=False):

    # turn orgs into humans
    if inv:
        df[df['labels'] == 2] = 1
        return df

    df[df['labels'] == 2] = 0
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
    # sort by type (this should be changed - only should be one read of the master df)
    master_df = pd.read_csv(master_path)

    mdf = turn_orgs_to_bots(master_df, inv=True)

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

    print(f"Evaluated for threshold: {threshold}. Accuracy: {metadata['overall_accuracy']}")
    return metadata

def plot_accuracy(metatable, title="Thresholds vs Overall Accuracy"):
    # plot the overall accuracy against the possible thresholds
    plt.title(title)
    x = metatable.keys()
    y = [metatable[i]["overall_accuracy"] for i in metatable.keys()]

    plt.plot(x, y)
    plt.savefig(f"{title}.png")

def plot_confusion_matrix_rates(metatable, title):
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


    plot_accuracy(metatable=metadata_list, title="Thresholds vs Overall Accuracy (Orgs are NonBot)")

    # should be able to toggle this one and off
    plot_confusion_matrix_rates(metatable=metadata_list, title="[Conf Matrix] Thresholds vs Overall Accuracy (Orgs are NonHuman)")

def load_model(path):
    try:
        pl = pickle.load(open(path, 'rb'))
        return pl
    except Exception as e:
        print(f'ERROR loading model: {repr(e)}')

def predict(data_row, predictor):
    reshaped_data = np.array(list(data_row.values())[1:8]).reshape(1, -1)
    return predictor.predict(reshaped_data)[0]

def model_validation(model_path):
    """
    Test a model against the master testing list.
    Create a confusion matrix
    :return:
    """

    predictor = load_model(model_path)

    # get all the rows of the test set
    master_df = pd.read_csv(master_path)

    mdf = turn_orgs_to_bots(master_df, inv=True)

    mdf_dict = mdf.to_dict('records')

    preds = []
    row_num = 0
    for row in mdf_dict:
        preds.append(predict(row, predictor))
        print(f"ROW {row_num}...")
        row_num += 1

    mdf['predictions'] = preds

    correct_predictions = mdf[mdf['predictions'] == mdf['labels']]
    total_amt = len(mdf)
    correct_pred_amt = len(correct_predictions)
    overall_accuracy = np.round(correct_pred_amt / total_amt, 4)
    actual_humans = mdf[mdf['labels'] == 1].reset_index(drop=True)
    actual_bots = mdf[mdf['labels'] == 0].reset_index(drop=True)

    print(overall_accuracy)
    # TP, FP, FN, TN
    true_positive = actual_humans[actual_humans['labels'] == actual_humans['predictions']]
    false_positive = actual_bots[actual_bots['labels'] != actual_bots['predictions']]
    false_negative = actual_humans[actual_humans['labels'] != actual_humans['predictions']]
    true_negative = actual_bots[actual_bots['labels'] == actual_bots['predictions']]

    metadata = {
        "total_rows": total_amt,
        "correct_predictions": correct_pred_amt,
        "overall_accuracy": overall_accuracy * 100,
        "tp": len(true_positive),
        "fp": len(false_positive),
        "tn": len(true_negative),
        "fn": len(false_negative)
    }
    print("TRUE:")
    print(mdf['labels'])

    print("\nPRED: ")
    print(mdf['predictions'])

    # plot the confusion matrix with sklearn
    ConfusionMatrixDisplay.from_predictions(y_true=mdf['labels'], y_pred=mdf['predictions'], normalize='all')
    plt.show()
def main():

    # validate the CAP method with orgs being nonBot:
    validate_threshold_method(50)
    #model_validation(r"C:\Users\Ikechukwu Anude\Documents\vaccine-data-processing\models\new_model_batch\NEW_XGB_Classifier.dat")
    #exit(5)





    pass

main()