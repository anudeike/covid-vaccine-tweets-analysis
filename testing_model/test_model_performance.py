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

# the goal of this file
# is to explore the already gathered data
master_path = "data_bank/cleaning_data/master_training_data_id/master_train_one_hot_no_outliers_z3.csv"
example_path = "data_bank/cleaning_data/master_training_data_id/master_train_one_hot.csv"



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

def log_reg_holdout():
    # sort by type
    # bots = master_df[master_df['labels'] == 0]
    # humans = master_df[master_df['labels'] == 1]
    # organizations = master_df[master_df['labels'] == 2]
    master_df = pd.read_csv(example_path)

    bots_humans = turn_orgs_to_bots(master_df)

    # split into array for the features and resp vars
    x1 = bots_humans.drop(['labels', 'financial', 'self-declared', 'fake_follower', 'CAP'], axis=1).values
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
def log_reg_holdout_cm():
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

    model.fit(X_scaled, Y_train)


    result = model.score(X_test_scaled, Y_test) # scored using the train and test split

    print("Accuracy: %.2f%%" % (result * 100.0))


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
    kfold = model_selection.KFold(n_splits=num_fold)

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
    # bots = master_df[master_df['labels'] == 0]
    # humans = master_df[master_df['labels'] == 1]
    # organizations = master_df[master_df['labels'] == 2]
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

def holdout_all_classifiers(save_path="XGB_Default_Classifier"):
    # sort by type
    # bots = master_df[master_df['labels'] == 0]
    # humans = master_df[master_df['labels'] == 1]
    # organizations = master_df[master_df['labels'] == 2]
    master_df = pd.read_csv("training_data/master_train_one_hot.csv")
    bots_humans = turn_orgs_to_bots(master_df)

    # split into array for the features and resp vars
    #removed = ['labels', 'financial', 'self-declared', 'fake_follower', 'CAP']
    x1 = bots_humans.drop(['labels', 'id'], axis=1).values
    # print(bots_humans.drop(['labels', 'id'], axis=1).head(5))
    # return
    y1 = bots_humans['labels'].values


    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(x1, y1, test_size=0.20, random_state=100,stratify=y1)

    # preprocess the data
    X_scaled = preprocessing.scale(X_train)
    X_scaled_full_set = preprocessing.scale(x1)
    X_test_scaled = preprocessing.scale(X_test)

    # TRAINING
    model = xgboost.XGBClassifier(n_estimators=50, max_depth=6, learning_rate=0.1)

    optimization_dict = {
        'max_depth': [2, 4, 6],
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.1, 0.01, 1],
    }

    # model = model_selection.GridSearchCV(model, optimization_dict,
    #                       scoring='accuracy', verbose=1)

    model.fit(X_scaled, Y_train)

    #model.fit(X_scaled_full_set, y1)

    # list all the features and their importances
    print_feature_importances(model.feature_importances_)

    # print(model.feature_importances_)
    overall_model_score = model.score(X_test_scaled, Y_test)

    # get confusion matrix
    y_pred = model.predict(X_test_scaled)
    conf_mat = confusion_matrix(y_true=Y_test, y_pred=y_pred)
    print(f"Overall Model Score: {overall_model_score}")
    print(f"Confusion Matrix: {conf_mat}")


    # print(f'Accuracy: {model.best_score_ * 100}')
    # print(model.best_params_)

    # save the model
    pickle.dump(model, open(f'model/{save_path}_2.dat', 'wb'))

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
    X_test_scaled = preprocessing.scale(X_test)

    # TRAINING

    model = xgboost.XGBClassifier()

    optimization_dict = {
        'max_depth': [2, 4, 6],
        'n_estimators': [50, 200, 500],
        'learning_rate': [0.1, 0.01, 1],
    }

    model = model_selection.GridSearchCV(model, optimization_dict,
                          scoring='accuracy', verbose=1)

    model.fit(X_scaled, Y_train)
    #model.fit(X_scaled, y1)

    # list all the features and their importances
    #print_feature_importances(model.feature_importances_)
    #print(model.feature_importances_)
    print(model.best_score_)
    print(model.best_params_)

# retrain model using k_folds
def k_folds_random_forest(save_path="XGB_Default_Classifier", scoring_technique="roc_auc", num_fold=10):
    # sort by type
    # bots = master_df[master_df['labels'] == 0]
    # humans = master_df[master_df['labels'] == 1]
    # organizations = master_df[master_df['labels'] == 2]
    master_df = pd.read_csv("training_data/master_train_one_hot_no_dup.csv")
    bots_humans = turn_orgs_to_bots(master_df)

    # split into array for the features and resp vars
    # removed = ['labels', 'financial', 'self-declared', 'fake_follower', 'CAP']
    x1 = bots_humans.drop(['labels', 'id'], axis=1).values

    # return
    y1 = bots_humans['labels'].values

    # preprocess the data
    X_scaled = preprocessing.scale(x1)

    # TRAINING
    model = xgboost.XGBClassifier()

    optimization_dict = {
        'max_depth': [6],
        'n_estimators': [100],
        'learning_rate': [0.1],
    }

    kfold = model_selection.KFold(n_splits=num_fold, shuffle=True, random_state=100)

    model = model_selection.GridSearchCV(model,
                                         optimization_dict,
                                         scoring=scoring_technique,
                                         refit='AUC',
                                         verbose=1,
                                         cv=kfold,
                                         return_train_score=True)
    # fit
    model.fit(X_scaled, y1)

    # get the best estimator
    best_estimator = model.best_estimator_

    # get the best score
    best_params = model.best_params_
    best_score = model.best_score_
    results = model.cv_results_

    print(f"Best Score: %.4f%%" % best_score)
    print(f"Best Params: {best_params}")

    # # print(model.feature_importances_)
    # overall_model_score = model.score(X_test_scaled, Y_test)

    pickle.dump(model, open(f'model/{save_path}_im_3.dat', 'wb'))
    # get confusion matrix
    y_pred = model.predict(X_scaled)
    conf_mat = confusion_matrix(y_true=y1, y_pred=y_pred)
    print(f"Confusion Matrix: {conf_mat}")

    # plot the confusion matrix
    plt.title("Confusion Matrix for Random Forest Classifier")
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat)
    disp.plot()
    plt.show()

    # plot the roc curve
    fpr, tpr, thresholds = roc_curve(y1, y_pred)
    roc_auc = auc(fpr, tpr)
    display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='example estimator')
    display.plot()
    plt.show()

    # classification report
    print(f"Classification Report: \n {classification_report(y1, y_pred)}")

    # pickle.dump(model, open(f'model/{save_path}_im_2.dat', 'wb'))


def load_model():
    return pickle.load(open("model/XGB_Default_Classifier.dat", "rb"))


def validate_model():
    # load the data
    master_df = pd.read_csv("training_data/master_train_one_hot.csv")
    bots_humans = turn_orgs_to_bots(master_df)

    # load model
    model = load_model()

    # split into array for the features and resp vars
    # removed = ['labels', 'financial', 'self-declared', 'fake_follower', 'CAP']
    x1 = bots_humans.drop(['labels', 'id'], axis=1).values
    # print(bots_humans.drop(['labels', 'id'], axis=1).head(5))
    # return
    y1 = bots_humans['labels'].values


    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(x1, y1, test_size=0.80, random_state=100,
                                                                        stratify=y1)

    # preprocess the data
    X_scaled = preprocessing.scale(X_train)
    X_scaled_full_set = preprocessing.scale(x1)
    X_test_scaled = preprocessing.scale(X_test)

    score = model.score(X_test_scaled, Y_test)

    print(f'model score: {score}')


def validate_model_k_folds(num_folds=3):
    # load the data
    master_df = pd.read_csv("training_data/master_train_one_hot.csv")
    bots_humans = turn_orgs_to_bots(master_df)

    # load model
    model = load_model()

    # split into array for the features and resp vars
    # removed = ['labels', 'financial', 'self-declared', 'fake_follower', 'CAP']
    x1 = bots_humans.drop(['labels', 'id'], axis=1).values
    # print(bots_humans.drop(['labels', 'id'], axis=1).head(5))
    # return
    y1 = bots_humans['labels'].values

    # use the k_folds technique
    kfold = model_selection.KFold(n_splits=num_folds, random_state=100)

    # preprocess the data
    X_scaled = preprocessing.scale(x1)

    # cross validate/fit
    res_kfold = model_selection.cross_val_score(model, X_scaled, y1, cv=kfold)

    print("Accuracy: %.2f%%" % (res_kfold.mean() * 100.0))

# RUN
#validate_model()
#validate_model_k_folds() # used to validate a model later on
k_folds_random_forest()
#log_reg_holdout()
#holdout_all_classifiers()
#log_reg_holdout_cm()
