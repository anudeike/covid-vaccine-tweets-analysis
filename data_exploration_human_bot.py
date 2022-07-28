import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import necessary modules
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
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
from sklearn.metrics import plot_confusion_matrix, accuracy_score

# the goal of this file
# is to explore the already gathered data
master_path = "data_bank/cleaning_data/master_training_data_id/new_master_train_one_hot_no_outliers_z3.csv"
full = "data_bank/cleaning_data/master_training_data_id/master_train_one_hot_no_dup.csv"
example_path = "data_bank/cleaning_data/master_training_data_id/master_train_one_hot.csv"



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

def get_rid_of_orgs(df):

    df = df[df['labels'] != 2]
    return df

def print_feature_importances(arr):

    feature_names = ["CAP","astroturf","fake_follower","financial","other","overall","self-declared"]


    if len(arr) != len(feature_names):
        print(len(feature_names))
        print(len(arr))
        print("the amount of features must be the same length")
        return

    for i in range(len(feature_names)):
        print('Importance of {}: {:.2f}%'.format(feature_names[i], arr[i] * 100))
#master = pd.read_csv(master_path)
#print(master.groupby(['labels']).count())
#describe_plot_save(master[master["labels"] == 0], "plots/meta-data_17k", name="bots_17k")

def svm_holdout():
    # sort by type
    # bots = master_df[master_df['labels'] == 0]
    # humans = master_df[master_df['labels'] == 1]
    # organizations = master_df[master_df['labels'] == 2]
    master_df = pd.read_csv(example_path)

    bots_humans = turn_orgs_to_bots(master_df, inv=True)

    # split into array for the features and resp vars
    dropped = ['labels', 'id']
    x1 = bots_humans.drop(dropped, axis=1).values
    y1 = bots_humans['labels'].values

    # train using Logistic Regression
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(x1, y1, test_size=0.30, random_state=100,stratify=y1)
    model = svm.SVC()

    # preprocess the data
    X_scaled = preprocessing.scale(X_train)
    X_test_scaled = preprocessing.scale(X_test)

    model.fit(X_scaled, Y_train)


    result = model.score(X_test_scaled, Y_test) # scored using the train and test split

    print("Accuracy: %.2f%%" % (result * 100.0))
    pickle.dump(model, open(f'models/{"SVM_NEW_VALIDATION"}.dat', 'wb'))

def log_reg_holdout():
    # sort by type
    # bots = master_df[master_df['labels'] == 0]
    # humans = master_df[master_df['labels'] == 1]
    # organizations = master_df[master_df['labels'] == 2]
    master_df = pd.read_csv(example_path)

    bots_humans = turn_orgs_to_bots(master_df, inv=True)

    # split into array for the features and resp vars
    dropped = ['labels', 'id']
    x1 = bots_humans.drop(dropped, axis=1).values
    y1 = bots_humans['labels'].values

    # train using Logistic Regression
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(x1, y1, test_size=0.30, random_state=100,stratify=y1)
    model = LogisticRegression()

    # preprocess the data
    X_scaled = preprocessing.scale(X_train)
    X_test_scaled = preprocessing.scale(X_test)

    model.fit(X_scaled, Y_train)


    result = model.score(X_test_scaled, Y_test) # scored using the train and test split

    print("Accuracy: %.2f%%" % (result * 100.0))
    pickle.dump(model, open(f'models/{"LOG_REG_NEW_VALIDATION"}.dat', 'wb'))

def log_reg_holdout_save_model(model_name):
    # sort by type
    # bots = master_df[master_df['labels'] == 0]
    # humans = master_df[master_df['labels'] == 1]
    # organizations = master_df[master_df['labels'] == 2]
    master_df = pd.read_csv(master_path)

    bots_humans = turn_orgs_to_bots(master_df)

    # split into array for the features and resp vars
    x1 = bots_humans.drop('labels', axis=1).values
    y1 = bots_humans['labels'].values

    # remove features from training set
    x1 = bots_humans.drop('astroturf', axis=1).values  # feature dropping

    # train using Logistic Regression
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(x1, y1, test_size=0.30, random_state=100,stratify=y1)
    model = LogisticRegression()

    # preprocess the data
    X_scaled = preprocessing.scale(X_train)
    X_test_scaled = preprocessing.scale(X_test)

    # train the model
    model.fit(X_scaled, Y_train)

    # save the model
    model_path = f'models/{model_name}_LogisticRegression.sav'
    pickle.dump(model, open(model_path, "wb"))

    loaded_model = pickle.load(open(f"models/{model_name}_LogisticRegression.sav", "rb"))
    result = loaded_model.score(X_test_scaled, Y_test) # scored using the train and test split

    print("Accuracy: %.2f%%" % (result * 100.0))
# testing the matrix
def log_reg_holdout_cm(save_path="Log_reg_model_whole"):
    # sort by type
    # bots = master_df[master_df['labels'] == 0]
    # humans = master_df[master_df['labels'] == 1]
    # organizations = master_df[master_df['labels'] == 2]
    master_df = pd.read_csv(master_path)

    bots_humans = turn_orgs_to_bots(master_df)

    # split into array for the features and resp vars
    y1 = bots_humans['labels'].values

    # remove features from training set
    x1 = bots_humans.drop(['labels', 'id'], axis=1).values  # feature dropping

    # train using Logistic Regression
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(x1, y1, test_size=0.30, random_state=100,stratify=y1)
    model = LogisticRegression()

    # preprocess the data
    X_scaled = preprocessing.scale(X_train)
    X_test_scaled = preprocessing.scale(X_test)

    model.fit(X_scaled, Y_train)


    result = model.score(X_test_scaled, Y_test) # scored using the train and test split

    print("Accuracy: %.2f%%" % (result * 100.0))

    # added
    pickle.dump(model, open(f'models/{save_path}.dat', 'wb'))

    # work on the confusion matrix
    title_options = [("Confusion Matrix, without normalization", None),
                     ("Normalized Confusion Matrix", "true")]
    for title, normalize in title_options:
        disp = plot_confusion_matrix(model, X_test_scaled, Y_test, display_labels=["bot", "human"],
                                     cmap=plt.cm.Blues, normalize=normalize)

        disp.ax_.set_title(title)
        print(title)
        print(disp.confusion_matrix)

    plt.show()

def log_reg_kfold(num_fold = 10):
    # sort by type
    # bots = master_df[master_df['labels'] == 0]
    # humans = master_df[master_df['labels'] == 1]
    # organizations = master_df[master_df['labels'] == 2]
    master_df = pd.read_csv(master_path)
    bots_humans = turn_orgs_to_bots(master_df)


    # split into array for the features and resp vars
    x1 = bots_humans.drop('labels', axis=1).values
    y1 = bots_humans['labels'].values

    # remove features from training set
    x1 = bots_humans.drop('astroturf', axis=1).values  # feature dropping

    # set the folds
    kfold = model_selection.KFold(n_splits=num_fold, random_state=100)

    # train using Logistic Regression
    model = LogisticRegression()

    # preprocess the data
    X_scaled = preprocessing.scale(x1)

    # cross validate/fit
    res_kfold = model_selection.cross_val_score(model, X_scaled, y1, cv=kfold)

    print("Accuracy: %.2f%%" % (res_kfold.mean() * 100.0))

def log_reg_kfold_strat(num_fold = 2):
    # sort by type
    # bots = master_df[master_df['labels'] == 0]
    # humans = master_df[master_df['labels'] == 1]
    # organizations = master_df[master_df['labels'] == 2]
    master_df = pd.read_csv(master_path)
    bots_humans = turn_orgs_to_bots(master_df)


    # split into array for the features and resp vars
    x1 = bots_humans.drop('labels', axis=1).values
    y1 = bots_humans['labels'].values

    # remove features from training set
    x1 = bots_humans.drop('astroturf', axis=1).values  # feature dropping

    # set the folds
    kfold = model_selection.StratifiedKFold(n_splits=num_fold, random_state=100)

    # train using Logistic Regression
    model = LogisticRegression()

    # preprocess the data
    X_scaled = preprocessing.scale(x1)

    # cross validate/fit
    res_kfold = model_selection.cross_val_score(model, X_scaled, y1, cv=kfold)

    print("Accuracy: %.2f%%" % (res_kfold.mean() * 100.0))

def log_reg_loocv():
    # doesn't seem to work all that well
    # sort by type
    master_df = pd.read_csv(master_path)
    bots_humans = turn_orgs_to_bots(master_df)


    # split into array for the features and resp vars
    x1 = bots_humans.drop('labels', axis=1).values
    y1 = bots_humans['labels'].values

    # set the folds
    loocv = model_selection.LeaveOneOut()

    # train using Logistic Regression
    model = LogisticRegression()

    # preprocess the data
    X_scaled = preprocessing.scale(x1)

    # cross validate/fit
    res_kfold = model_selection.cross_val_score(model, X_scaled, y1, cv=loocv)

    print("Accuracy: %.2f%%" % (res_kfold.mean() * 100.0))

def log_reg_shuffle(num_fold = 5):
    # sort by type
    # bots = master_df[master_df['labels'] == 0]
    # humans = master_df[master_df['labels'] == 1]
    # organizations = master_df[master_df['labels'] == 2]
    master_df = pd.read_csv(master_path)
    bots_humans = turn_orgs_to_bots(master_df)


    # split into array for the features and resp vars
    x1 = bots_humans.drop('labels', axis=1).values
    y1 = bots_humans['labels'].values

    # set the folds
    kfold = model_selection.ShuffleSplit(n_splits=num_fold, test_size=0.3, random_state=100)

    # train using Logistic Regression
    model = LogisticRegression()

    # preprocess the data
    X_scaled = preprocessing.scale(x1)

    # cross validate/fit
    res_kfold = model_selection.cross_val_score(model, X_scaled, y1, cv=kfold)

    print("Accuracy: %.2f%%" % (res_kfold.mean() * 100.0))

def log_reg_shuffle_strat(num_fold = 10):
    # sort by type
    # bots = master_df[master_df['labels'] == 0]
    # humans = master_df[master_df['labels'] == 1]
    # organizations = master_df[master_df['labels'] == 2]
    master_df = pd.read_csv(master_path)
    bots_humans = turn_orgs_to_bots(master_df)


    # split into array for the features and resp vars
    x1 = bots_humans.drop('labels', axis=1).values
    y1 = bots_humans['labels'].values

    # set the folds
    kfold = model_selection.StratifiedShuffleSplit(n_splits=num_fold, test_size=0.3, random_state=100)

    # train using Logistic Regression
    model = LogisticRegression()

    # preprocess the data
    X_scaled = preprocessing.scale(x1)

    # cross validate/fit
    res_kfold = model_selection.cross_val_score(model, X_scaled, y1, cv=kfold)

    print("Accuracy: %.2f%%" % (res_kfold.mean() * 100.0))

def holdout_all_classifiers(save_path="NEW_XGB_Classifier_noStrat_op_whole_li"):
    # sort by type
    # bots = master_df[master_df['labels'] == 0]
    # humans = master_df[master_df['labels'] == 1]
    # organizations = master_df[master_df['labels'] == 2]

    # MASTER PATH IS TRAINING PATH
    master_df = pd.read_csv(example_path)

    bots_humans = turn_orgs_to_bots(master_df, inv=True)

    # split into array for the features and resp vars
    #removed = ['labels', 'financial', 'self-declared', 'fake_follower', 'CAP']
    x1 = bots_humans.drop(['labels', 'id'], axis=1).values
    # print(bots_humans.drop(['labels', 'id'], axis=1).head(5))
    # return
    y1 = bots_humans['labels'].values


    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(x1, y1, test_size=0.30, random_state=100)

    # write the test to file
    x_test_df = pd.DataFrame(X_test)
    x_test_df["labels"] = Y_test
    x_test_df.to_csv("x_test.csv")
    # y_test_df.to_csv("y_test.csv")

    # preprocess the data
    X_scaled = preprocessing.scale(X_train)
    X_scaled_full_set = preprocessing.scale(x1)
    X_test_scaled = preprocessing.scale(X_test)

    # TRAINING
    model = xgboost.XGBClassifier(n_estimators=50, max_depth=6, learning_rate=0.1)

    optimization_dict = {
        'max_depth': [2, 4, 6],
        'n_estimators': [50, 75, 100],
        'learning_rate': [0.1, 0.01, 1],
    }

    model = model_selection.GridSearchCV(model, optimization_dict,
                          scoring='accuracy', verbose=1, cv=10)

    model.fit(X_scaled, Y_train)

    # model.fit(X_scaled_full_set, y1)

    # list all the features and their importances
    print(model.score(X_test_scaled, Y_test))
    print(f"Best Score: {model.best_score_}\nModel Score {model.score(X_test_scaled, Y_test)}")

    # save the model
    pickle.dump(model, open(f'models/{save_path}.dat', 'wb'))
    print(f"Saved Model to: {save_path}...")

    return model

def holdout_all_classifiers_pruned():
    # sort by type

    # this one prunes outlier and then throws them back together
    master_df = pd.read_csv(master_path)

    bots_humans = turn_orgs_to_bots(master_df)
    #bots_humans = get_rid_of_orgs(master_df)

    # split into array for the features and resp vars
    removed = ['labels', 'financial', 'self-declared', 'fake_follower', 'CAP']
    x1 = bots_humans.drop(['labels', 'id'], axis=1).values
    # print(bots_humans.drop(['labels', 'id'], axis=1).head(5))
    # return
    y1 = bots_humans['labels'].values

    # train using Logistic Regression
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(x1, y1, test_size=0.30, random_state=100,stratify=y1)

    # preprocess the data
    X_scaled = preprocessing.scale(X_train)
    # X_test_scaled = preprocessing.scale(X_test)

    # TRAINING

    model = xgboost.XGBClassifier()

    optimization_dict = {
        'max_depth': [2, 4, 6],
        'n_estimators': [50, 200, 500],
        'learning_rate': [0.1, 0.01, 1],
    }

    model = model_selection.GridSearchCV(model, optimization_dict,
                          scoring='balanced_accuracy', verbose=1)

    model.fit(X_scaled, Y_train)
    #model.fit(X_scaled, y1)

    # list all the features and their importances
    #print_feature_importances(model.feature_importances_)
    #print(model.feature_importances_)
    print(f"Overall Accuracy (Raw): {model.best_score_}")
    #print(model.best_params_)

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

def model_validation(model):
    """
    Test a model against the master testing list
    :return:
    """

    # get all the rows of the test set
    master_df = pd.read_csv(example_path)

    mdf = turn_orgs_to_bots(master_df, inv=True)

    mdf_dict = mdf.to_dict('records')

    preds = []
    for row in mdf_dict:
        reshaped_data = np.array(list(row.values())[1:8]).reshape(1, -1)
        preds.append(model.predict(reshaped_data)[0])

    mdf['predictions'] = preds

    correct_predictions = mdf[mdf['predictions'] == mdf['labels']]
    total_amt = len(mdf)
    correct_pred_amt = len(correct_predictions)
    overall_accuracy = np.round(correct_pred_amt / total_amt, 4)
    actual_humans = mdf[mdf['labels'] == 1].reset_index(drop=True)
    actual_bots = mdf[mdf['labels'] == 0].reset_index(drop=True)

    print(f"Overall Accuracy (Raw): {overall_accuracy}")

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

    print(metadata)

    confusion_matrix(y_true=mdf['labels'], y_pred=mdf['predictions'] )
    # plot the confusion matrix with raw data
    plot_confusion_matrix_rates(metadata)
    print(overall_accuracy)

# log_reg_holdout()
#svm_holdout()
#model = holdout_all_classifiers()
def load_model(path):
    try:
        pl = pickle.load(open(path, 'rb'))
        return pl
    except Exception as e:
        print(f'ERROR loading model: {repr(e)}')

model_validation(load_model(r"C:\Users\Ikechukwu Anude\Documents\vaccine-data-processing\models\SVM_NEW_VALIDATION.dat"))
#log_reg_holdout_cm()
