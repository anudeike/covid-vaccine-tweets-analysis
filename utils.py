import pandas as pd
import botometer
from dotenv import load_dotenv
import os
import time
import luckysocial
from scipy import stats
import numpy as np
import json

"""
This file contains all the utility functions that I will use throughout this repo

Its about time I had one.
"""

# GLOBAL VARS
path = "data/verified-human-usernames.csv"
outpath = "data_bank/cleaning_data/id_labels_humans.tsv"

path_to_id_labels = "data_bank/cleaning_data/id-labels.tsv"
batch_files_output = "data_bank/cleaning_data/output_batches" # this is just the folder

company_names_path = "organization_officials_data/Organization-Data_Extra - Sheet1.csv"

load_dotenv()

rapidapi_key = os.getenv('RAPID_FIRE_KEY')

# authentication
twitter_app_auth = {
    'consumer_key': os.getenv('TWITTER_API_KEY'),
    'consumer_secret': os.getenv('TWITTER_API_SECRET'),
    'access_token': os.getenv('TWITTER_ACCESS_TOKEN'),
    'access_token_secret': os.getenv('TWITTER_ACCESS_SECRET'),
}


bom = botometer.Botometer(wait_on_ratelimit=True,
                          rapidapi_key=rapidapi_key,
                          **twitter_app_auth)


# ======== FILE PROCESSING FUNCTIONS ======= #
def create_file_with_id_and_type(src_path, out_path):
    """
    Takes in a file with twitter use data info and returns a tsv with only the id and class of the user
    :param src_path: Path to original data file
    :param out_path: Output data path
    :return:
    """

    df = pd.read_csv(src_path)
    out_df = df[["id", "type"]]

    out_df.to_csv(out_path,sep="\t",index=False)

# ======== FETCH FROM INFO FROM BOTOMETER ======= #
def create_file_with_botometer_statistics(in_path, out_path, separator="\t"):
    """
    Takes a file that contains an account id and type and passes it to
    botometer's api to get a bunch of statistics on the account.
    :param in_path: The original file with the account id and type
    :param out_path: the path of the output
    :return:
    """
    in_df = pd.read_csv(in_path, sep=separator)

    print(in_df['type'].value_counts())

    # create a dataframe that will be the output
    out_df = pd.DataFrame(columns=["id", "CAP", "astroturf",
                                   "fake_follower", "financial",
                                   "other", "overall", "self-declared","spammer"])

    # get all the ids and the types
    ids = in_df["id"].values
    types = in_df["type"].values

    # this will be used to keep track of categories
    count = 101

    # this is the rate limit
    rate_limit = 100
    timeout = 60

    # check the accounts
    for id, result in bom.check_accounts_in(ids[count:150]):

        # this will be appended to the new dataframe
        row = {}

        try:
            if (result["user"]["majority_lang"] == 'en'):
                # use the english results

                # for each row that'll be appended
                row = {
                    "id": id,
                    "CAP": result['cap']['english'],
                    "astroturf": result['display_scores']['english']['astroturf'],
                    "fake_follower": result['display_scores']['english']['fake_follower'],
                    "financial": result['display_scores']['english']['financial'],
                    "other": result['display_scores']['english']['other'],
                    "overall": result['display_scores']['english']['overall'],
                    "self-declared": result['display_scores']['english']['self_declared'],
                    "spammer": result['display_scores']['english']['spammer'],
                    "type": types[count]
                }
            else:

                row = {
                    "id": id,
                    "CAP": result['cap']['universal'],
                    "astroturf": result['display_scores']['universal']['astroturf'],
                    "fake_follower": result['display_scores']['universal']['fake_follower'],
                    "financial": result['display_scores']['universal']['financial'],
                    "other": result['display_scores']['universal']['other'],
                    "overall": result['display_scores']['universal']['overall'],
                    "self-declared": result['display_scores']['universal']['self_declared'],
                    "spammer": result['display_scores']['universal']['spammer'],
                    "type": types[count]
                }

            # append to dataframe
            out_df = out_df.append(row, ignore_index=True)

            # if the count is mod 75
            if count % rate_limit == 0:
                # export the output df using the latest count as the output
                p = "{}/id_labels_with_cap_second_bots_{}.csv".format(out_path, count)
                out_df.to_csv(p, index=False)

                # time out
                if count > 1:
                    print("is sleeping for {} seconds...".format(timeout))
                    time.sleep(timeout)


            print("{} has been processed. Number: {}".format(id, count))

            # increment the count
            count += 1

        except Exception as e:
            # skip if error
            print("{} Could not be fetched: {}".format(id, e))

            # if the count is mod 75
            if count % rate_limit == 0:
                # export the output df using the latest count as the output
                p = "{}/id_labels_with_cap_second_botstesting_{}.csv".format(out_path, count)
                out_df.to_csv(p, index=False)

                # time out
                if count > 1:
                    print("is sleeping for {} seconds...".format(timeout))
                    time.sleep(timeout) # should be about 5 minutes

            # increment the count
            count += 1
            continue

    # send the info the the df
    p = "{}/id_labels_with_cap_second_bots_{}.csv".format(out_path, count)
    out_df.to_csv(p, index=False)

# FETCH FOR ORGANIZATIONS
def create_file_with_botometer_statistics_org(in_path, out_path):
    """
    Takes a file that contains an account id and type and passes it to
    botometer's api to get a bunch of statistics on the account.
    :param in_path: The original file with the account id and type
    :param out_path: the path of the output
    :return:
    """
    in_df = pd.read_csv(in_path)

    print(in_df['type'].value_counts())

    # create a dataframe that will be the output
    out_df = pd.DataFrame(columns=["id", "CAP", "astroturf",
                                   "fake_follower", "financial",
                                   "other", "overall", "self-declared","spammer"])

    # get all the ids and the types
    ids = in_df["handle"].values
    types = in_df["type"].values

    # this will be used to keep track of categories
    count = 302

    # this is the rate limit
    rate_limit = 120
    timeout = 180

    # check the accounts
    for id, result in bom.check_accounts_in(ids[count:]):

        # this will be appended to the new dataframe
        row = {}

        try:
            if (result["user"]["majority_lang"] == 'en'):
                # use the english results

                # for each row that'll be appended
                row = {
                    "id": id,
                    "CAP": result['cap']['english'],
                    "astroturf": result['display_scores']['english']['astroturf'],
                    "fake_follower": result['display_scores']['english']['fake_follower'],
                    "financial": result['display_scores']['english']['financial'],
                    "other": result['display_scores']['english']['other'],
                    "overall": result['display_scores']['english']['overall'],
                    "self-declared": result['display_scores']['english']['self_declared'],
                    "spammer": result['display_scores']['english']['spammer'],
                    "type": types[count]
                }
            else:

                row = {
                    "id": id,
                    "CAP": result['cap']['universal'],
                    "astroturf": result['display_scores']['universal']['astroturf'],
                    "fake_follower": result['display_scores']['universal']['fake_follower'],
                    "financial": result['display_scores']['universal']['financial'],
                    "other": result['display_scores']['universal']['other'],
                    "overall": result['display_scores']['universal']['overall'],
                    "self-declared": result['display_scores']['universal']['self_declared'],
                    "spammer": result['display_scores']['universal']['spammer'],
                    "type": types[count]
                }

            # append to dataframe
            out_df = out_df.append(row, ignore_index=True)

            # if the count is mod 75
            if count % rate_limit == 0:
                # export the output df using the latest count as the output
                p = "{}/id_labels_with_cap_{}.csv".format(out_path, count)
                out_df.to_csv(p, index=False)

                # time out
                if count > 1:
                    print("is sleeping for {} seconds...".format(timeout))
                    time.sleep(timeout)


            print("{} has been processed. Number: {}".format(id, count))

            # increment the count
            count += 1

        except Exception as e:
            # skip if error
            print("{} Could not be fetched: {}".format(id, e))

            # if the count is mod 75
            if count % rate_limit == 0:
                # export the output df using the latest count as the output
                p = "{}/id_labels_with_cap_{}.csv".format(out_path, count)
                out_df.to_csv(p, index=False)

                # time out
                if count > 1:
                    print("is sleeping for {} seconds...".format(timeout))
                    time.sleep(timeout) # should be about 5 minutes

            # increment the count
            count += 1
            continue

    # send the info the the df
    p = "{}/id_labels_with_cap_{}.csv".format(out_path, count)
    out_df.to_csv(p, index=False)
def add_index_to_given_file(in_path, out_path):
    df = pd.read_csv(in_path)
    df.to_csv(out_path, index=True)

def add_column_to_file_inplace(in_path, col_name="type", val_to_be_added="ORGANIZATION"):
    df = pd.read_csv(in_path)
    df[col_name] = val_to_be_added
    df.to_csv(in_path, index=False)

# turn the labels
def label_conversion(row):
    if row["type"] == 'human':
        return 1
    elif row["type"] == 'ORGANIZATION':
        return 2
    else:
        return 0 # bot

def types_to_integers(in_path, out_path):
    df = pd.read_csv(in_path)
    df["labels"] = df.apply(lambda row: label_conversion(row), axis=1)
    df = df.drop(columns=['type', 'spammer'])  # drop the type column, not needed
    df = df.drop_duplicates(subset=["id"]) # drop any duplicate ids
    df.to_csv(out_path, index=False)

def remove_column_and_output_result(in_path, out_path, col_name):
    df = pd.read_csv(in_path)

    #drop
    df.drop(col_name,axis=1,inplace=True)
    df.to_csv(out_path, index=False)

def remove_indices_and_output(in_path, out_path):
    df = pd.read_csv(in_path)
    df.to_csv(out_path, index=False)

def get_twitter_handle_from_name(name):

    print(f'Looking up {name}...')

    # lookup
    try:
        info = luckysocial.lookup(name)

        # error handling?
        if not info["twitter"]:
            return "No twitter"

        handle = info["twitter"].split("/")

        print(f'Twitter handle found: {handle[-1]}')

        # handle
        return handle[-1]
    except Exception as e:
        print(f'There was an error: {e}')
        return "Not Found"


def get_twitter_handles(in_path, out_path):
    """
    Gets a list of company names and gets their twitter handle
    :param in_path: the in path of
    :param out_path: the output path of the last twitter handles
    :return:
    """

    df = pd.read_csv(in_path)
    names = df["name"].values

    handles = []
    for x in range(len(names)):

        if x % 300 == 0 and x > 1:
            # filter out the no twitter
            filtered = list(filter(lambda word: word != 'No twitter', handles))

            # shoot it out to a csv file
            new_df = pd.DataFrame(data=filtered, columns=['handle'])
            new_df.to_csv(out_path + "_{}.csv".format(x), index=False)

            print("sleeping for 180 seconds...")
            time.sleep(180)

        handles.append(get_twitter_handle_from_name(names[x]))

    # filter out the no twitter
    filtered = list(filter(lambda word: word != 'No twitter', handles))

    # shoot it out to a csv file
    new_df = pd.DataFrame(data=filtered, columns=['handle'])
    new_df.to_csv(out_path + "all.csv", index=False)
    return

def remove_outliers(in_path=None):
    mdf = pd.read_csv("data_bank/cleaning_data/master_training_data_id/master_train_one_hot_no_dup.csv")

    # separate them by category
    human_df = mdf[mdf["labels"] == 1]
    non_human_df = mdf[mdf["labels"] == 0]

    # filter out the outliers
    # CAP,astroturf,fake_follower,financial,other,overall,self-declared

    # for humans
    columns = ["astroturf", "fake_follower", "financial", "other", "overall", "self-declared"]
    human_df = human_df[(np.abs(stats.zscore(human_df[columns])) < 2.5).all(axis=1)]

    # for non_humans
    non_human_df = non_human_df[(np.abs(stats.zscore(non_human_df[columns])) < 2.5).all(axis=1)]

    #print(human_df.describe())
    print(non_human_df.describe())

    # combine both of the dataframe and export
    master = pd.concat([human_df, non_human_df])

    #print(master.describe())
    master.to_csv("data_bank/cleaning_data/master_training_data_id/master_train_one_hot_no_outliers_z_25.csv", index=False)

    pass

def get_tweets_from_sample(path):

    # create a dataframe
    df = pd.DataFrame(columns=['id', 'screen_name', 'follower_count'])

    # goal
    # open the file and put username data into a dataframe and create a file for it
    with open(path, "r") as f:

        index = 0

        for line in f:
            # skip lines that are empty space
            if line.isspace():
                continue

            try:
                # deserialize
                tweet_data = json.loads(line)
                print(tweet_data['text'])
                # vars
                id = tweet_data['user']['id_str']
                screen_name = tweet_data['user']['screen_name']
                followers = tweet_data['user']['followers_count']

                # insert info
                df.loc[index] = [id] + [screen_name] + [followers]

                # control
                if index > 3:
                    break

                index += 1


            except(json.decoder.JSONDecodeError, TypeError) as e:

                # skip lines that don't serialize correctly should only be 2% of the lines
                print(e)
                continue

        # send the info to a csv file
        #df.to_csv(output_path)
        print(df.head(6))



get_tweets_from_sample("vaccine-2020-07-09.txt")
#create_file_with_botometer_statistics(in_path="data_bank/cleaning_data/id-labels.tsv", out_path="data_bank/cleaning_data/sixth_batch")
#remove_column_and_output_result("data/prepared_data/organization-split/organization_scores.csv", "data/prepared_data/organization-split/organization_scores_no_index.csv", "index")
#types_to_integers("data_bank/cleaning_data/master_training_data_id/master_training_set.csv", "data_bank/cleaning_data/master_training_data_id/master_train_one_hot_no_dup.csv")

#print(get_twitter_handle_from_name("uc berkeley"))
#get_twitter_handles(company_names_path, "organization_officials_data/org_twitter_handles_2")

#add_column_to_file_inplace("organization_officials_data/org_twitter_handles_2all.csv")
#create_file_with_botometer_statistics_org("organization_officials_data/org_twitter_handles_2all.csv", out_path="data_bank/cleaning_data/org_clean")
#remove_outliers()