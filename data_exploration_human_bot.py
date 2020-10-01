import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import necessary modules
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import LeavePOut
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import StratifiedKFold

# the goal of this file
# is to explore the already gathered data
master_path = "data_bank/cleaning_data/master_training_data_id/master_train_one_hot.csv"



# function to be able to save info
def describe_plot_save(df, path, name="default"):

    # send meta data to csv
    df.describe().to_csv(f'{path}/{name}-meta_stats.csv')
    # f'{path}/{name}-histogram_plot.png'
    df.hist(bins=20)
    plt.suptitle(f'{name} Histogram')
    plt.show()

def turn_orgs_to_bots(df):

    df[df['labels'] == 2] = 0
    return df




#describe_plot_save(bots, "plots/meta_data", name="Bot")

def main():
    # sort by type
    # bots = master_df[master_df['labels'] == 0]
    # humans = master_df[master_df['labels'] == 1]
    # organizations = master_df[master_df['labels'] == 2]
    master_df = pd.read_csv(master_path)

    bots_humans = turn_orgs_to_bots(master_df)

    # split into array for the features and resp vars
    x1 = bots_humans.drop('labels', axis=1).values
    y1 = bots_humans['labels'].values

    # train using Logistic Regression
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(x1, y1, test_size=0.30, random_state=100)
    model = LogisticRegression()

    # preprocess the data
    X_scaled = preprocessing.scale(X_train)

    model.fit(X_scaled, Y_train)

    X_test_scaled = preprocessing.scale(X_test)
    result = model.score(X_test_scaled, Y_test) # scored using the train and test split

    print("Accuracy: %.2f%%" % (result * 100.0))

main()