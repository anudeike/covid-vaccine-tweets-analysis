import pandas as pd
import botometer
from dotenv import load_dotenv
import os
import time

"""
This file contains all the utility functions that I will use throughout this repo

Its about time I had one.
"""

# GLOBAL VARS
path = "data/verified-human-usernames.csv"
outpath = "data_bank/cleaning_data/id_labels_humans.tsv"

path_to_id_labels = "data_bank/cleaning_data/id-labels.tsv"
batch_files_output = "data_bank/cleaning_data/output_batches" # this is just the folder

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
def create_file_with_botometer_statistics(in_path, out_path):
    """
    Takes a file that contains an account id and type and passes it to
    botometer's api to get a bunch of statistics on the account.
    :param in_path: The original file with the account id and type
    :param out_path: the path of the output
    :return:
    """
    in_df = pd.read_csv(in_path, sep="\t")

    print(in_df['type'].value_counts())

    # create a dataframe that will be the output
    out_df = pd.DataFrame(columns=["id", "CAP", "astroturf",
                                   "fake_follower", "financial",
                                   "other", "overall", "self-declared","spammer"])

    # get all the ids and the types
    ids = in_df["id"].values
    types = in_df["type"].values

    # this will be used to keep track of categories
    count = 502

    # this is the rate limit
    rate_limit = 100
    timeout = 240

    # check the accounts
    for id, result in bom.check_accounts_in(ids[count:1200]):

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


#create_file_with_id_and_type(path, outpath)

""" RUN FUNCTIONS HERE """
create_file_with_botometer_statistics(path_to_id_labels, out_path=batch_files_output)