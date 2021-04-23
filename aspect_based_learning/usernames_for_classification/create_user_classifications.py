import pickle
import botometer
from dotenv import load_dotenv
import os
import pandas as pd
import numpy as np
import time
from datetime import datetime
import sqlite3

# # sent analysis
# import aspect_based_sentiment_analysis as absa
# nlp = absa.load()

load_dotenv()

class AccountClassifier:
    def __init__(self, rapid_api_key, twitter_app_auth, model_path, data_file_path="", separator=',', isBatch=True, isPreprocessed=False, path_to_classified=None):

        # set whether you can use a batch
        self.isBatch = isBatch

        # init the botometer module
        self.bom = botometer.Botometer(wait_on_ratelimit=True,
                                       rapidapi_key=rapid_api_key,
                                       **twitter_app_auth)
        # load model
        self.model = self.load_model(model_path)

        # load the bank
        self.classification_bank = None
        if path_to_classified is not None:
            # this is the li_trial_master data
            self.classification_bank = pd.read_csv(path_to_classified)

        if isPreprocessed:
            print("using preprocessed tweets...")
            # set batch to false
            self.isBatch = False


            self.prep_df = self.get_preprocessed_tweets(data_file_path, separator)

        # will only be a batch if is batch is true
        if self.isBatch:
            self.account_ids = self.get_account_ids(data_file_path, separator)

        # set some stats vars
        self.missing_classified_accounts = []
        self.successful_analysis, self.failed_analysis = 0, 0
        self.tweet_index = 0





    def predict(self, data):

        try:
            r = self.model.predict(data)
            print(f'Prediction: {r}')
            return r
        except Exception as e:
            print(e)
            raise RuntimeError("Prediction Failed.")

    def classify(self):
        """
        Classifies the accounts given one at a time. Returns a dataframe containing the id and the predicted_label
        :return: Dataframe
        """

        # this function does not work on batches
        if self.isBatch:
            raise ValueError("Classifier set to work with Batch processing (isBatch = True). Consider using classify_batch() instead.")

        # set some of the dataframe parameters
        out = pd.DataFrame()
        row_list = []

        # for each of the names in the list -> this would work best on the muted list
        for id in self.account_ids:

            try:
                result = self.bom.check_account(id)

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
                        # "spammer": result['display_scores']['english']['spammer'],
                    }

                    # prepare to be read
                    reshaped_data = np.array(list(row.values())[1:]).reshape(1, -1)

                    # predict
                    classification = self.predict(reshaped_data) # make a prediction

                    # display what it is classified as
                    if classification[0] == 1:
                        print(f'{id} is classified as HUMAN')
                    else:
                        print(f'{id} is classified as NON-HUMAN')

                    # notify
                    print(f'{id} has been predicted.\n')

                    # append the row list
                    row_list.append({"id":id,
                                     "predicted_label": classification[0]})

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
                        # "spammer": result['display_scores']['universal']['spammer'],
                    }

                    # prepare to be read
                    reshaped_data = np.array(list(row.values())[1:]).reshape(1, -1)

                    classification = self.predict(reshaped_data)  # make a prediction

                    # display what it is classified as
                    if classification[0] == 1:
                        print(f'{id} is classified as HUMAN')
                    else:
                        print(f'{id} is classified as NON-HUMAN')

                    # notify
                    print(f'{id} has been predicted.\n')

                    # append the row list
                    row_list.append({"id": id,
                                     "predicted_label": classification[0]})

            except Exception as e:
                # skip if error
                print("{} Could not be fetched: {}".format(id, e))

        print(row_list)
        out = out.append(row_list)
        return out

    def classify_single_account(self, account_id):
        """
        Classifies a single account
        :param id: Either the screen_name of an account or an id.
        :return:
        """

        # this function does not work on batches
        if self.isBatch:
            raise ValueError("Classifier set to work with Batch processing (isBatch = True). Consider using classify_batch() instead.")


        # classify account
        try:
            result = self.bom.check_account(account_id)

            classification = []

            if (result["user"]["majority_lang"] == 'en'):
                # use the english results

                # for each row that'll be appended
                row = {
                    "id": account_id,
                    "CAP": result['cap']['english'],
                    "astroturf": result['display_scores']['english']['astroturf'],
                    "fake_follower": result['display_scores']['english']['fake_follower'],
                    "financial": result['display_scores']['english']['financial'],
                    "other": result['display_scores']['english']['other'],
                    "overall": result['display_scores']['english']['overall'],
                    "self-declared": result['display_scores']['english']['self_declared'],
                }

                # prepare to be read
                reshaped_data = np.array(list(row.values())[1:]).reshape(1, -1)

                # predict
                classification = self.predict(reshaped_data)  # make a prediction

                # notify
                print(f'{account_id} has been predicted.\n')

                # set the row
                row['classification'] = classification[0]

                # classification
                return row

            else:

                row = {
                    "id": account_id,
                    "CAP": result['cap']['universal'],
                    "astroturf": result['display_scores']['universal']['astroturf'],
                    "fake_follower": result['display_scores']['universal']['fake_follower'],
                    "financial": result['display_scores']['universal']['financial'],
                    "other": result['display_scores']['universal']['other'],
                    "overall": result['display_scores']['universal']['overall'],
                    "self-declared": result['display_scores']['universal']['self_declared'],
                }

                # prepare to be read
                reshaped_data = np.array(list(row.values())[1:]).reshape(1, -1)

                classification = self.predict(reshaped_data)  # make a prediction

                # notify
                print(f'{account_id} has been predicted.\n')

                row['classification'] = classification[0]

                return row



        except Exception as e:
            # raise so the external function can catch it
            raise ValueError("{} Could not be fetched: {}".format(account_id, e))


    def fetch_from_classification_bank(self, uid):

        if self.classification_bank is None:
            return None

        df = self.classification_bank

        # search
        res = df.loc[df['id'] == uid]

        if res.empty:
            print(f'\n{uid} could not be found in database.')
            return None

        print(f'\n{uid} fetched.')
        #print(res)

        # get the type
        pred = res.values

        return pred[0][1]

    def classify_preprocessed_by_row(self, row):

        # this function does not work on batches
        if self.isBatch:
            raise ValueError(
                "Classifier set to work with Batch processing (isBatch = True). Consider using classify_batch() instead.\n"
                "To use this function you must set (isPreproccessed = True) as well.")

        try:
            # find the account using botometer
            # del row['counts'] # don't need counts
            screen_name = row['id']
            print(f'Looking for {screen_name} in botometer.')

            # find the account
            res = self.classify_single_account(screen_name)
            row["prediction"] = res['classification']

            # # add to the end of the row
            # row['CAP'] = res['CAP']
            # row['astroturf'] = res['astroturf']
            # row['fake_follower'] = res['fake_follower']
            # row['financial'] = res['financial']
            # row['other'] = res['other']
            # row['self-declared'] = res['self-declared']

            # prediction
            print(f'Prediction for {screen_name}: {res["classification"]}')

            # success!
            self.successful_analysis += 1

            # increment the index
            print(f'Processed tweet # {self.tweet_index}')
            self.tweet_index += 1

            # add to the missing
            #self.missing_classified_accounts.append([screen_name, int(res)])
            return row

        except Exception as e:
            print(f"[top-level]: {repr(e)}\n")
            self.failed_analysis += 1
            return None
        pass


    def log_statistics(self, start, end):
        """
        Logs General Statistics for the Program
        :param start: Start time
        :param end: End time
        :param success: Amount of tweets successfully processed
        :param failed: Amound of tweets unsuccessfully processed
        :return: None
        """

        total = self.tweet_index
        elapsed = end - start
        amt_accts_added = len(self.missing_classified_accounts)

        elapsed_seconds = elapsed.total_seconds()

        # get the timer in the time in hours
        elapsed_hours = divmod(elapsed_seconds, 3600)
        elapsed_minutes = divmod(elapsed_hours[1], 60)

        # display the stats
        print("\n\n==========================STATISTICS============================\n")
        print(f'Time Elapsed: {elapsed_hours[0]} hours, {elapsed_minutes[0]} minutes, and {elapsed_minutes[1]} seconds.')
        print(f'Evaluated {total} Tweets. \n{self.successful_analysis} successful evaluations\n {self.failed_analysis} failed evaluations')
        print(f'{amt_accts_added} new accounts added to classification bank.')

    def get_account_ids(self, path, separ=','):
        """
        Reads the account ids from a file into a dataframe.
        :param path: Path to the csv, or tsv file
        :param separ: The separator of each value in the file. In CSV's this value is ','
        :return: a dataframe containing all the account ids
        """
        try:
            df = pd.read_csv(path, sep=separ)
            return df['id'].values # put into an array
        except Exception as e:
            raise ValueError(f'ERROR: {repr(e)}')

    def get_preprocessed_tweets(self, path, sepr=','):
        """
        Used to get the preprocessed tweets from the csv files.
        :param path: path to file
        :param sepr: separator
        :return: a dataframe
        """

        try:
            df = pd.read_csv(path, sep=sepr)
            return df
        except Exception as e:
            raise ValueError(f'ERROR: {repr(e)}')


    def load_model(self, path):
        """
        Loads the model at a certain path.
        :param path: Path to model created by pkl
        :return: Model
        """

        try:
            pl = pickle.load(open(path, 'rb'))
            return pl
        except Exception as e:
            print(f'ERROR: {repr(e)}')

    def update_classification_bank(self):
        # add the missing accounts to the bank and then output the bank
        mdf = pd.DataFrame(self.missing_classified_accounts, columns=["id", "prediction"])
        print(mdf)
        # send to csv
        mdf.to_csv("new_classified_accounts", index=False)




def insert_into_database(row_info, conn):

    # convert to a dataframe
    row_info = pd.DataFrame(row_info, index=[row_info['id']])

    # add stuff to the database and then commit
    # cur = conn.cursor()
    # cur.execute(sql_code, row_info)
    # cur.commit()
    row_info.to_sql(name="classified_accounts", con=conn, if_exists="append")
    conn.commit()

if __name__ == "__main__":


    """
    TWITTER_API_KEY=WG6dfDuyqgxQsbm2XkSCKzzuE
TWITTER_API_SECRET=os1fTHci5JSdfg1BWA98d1YrInvWG9fTc9eCOrZ2x9T7HgoVCV
TWITTER_ACCESS_TOKEN=1161101984688218113-OToVZdmMMV9euRKitHYAWhvS2A59kS
TWITTER_ACCESS_SECRET=wpwTBRYDcNhYHK0RALDL9LbWAPuBjf4HDwShosRpSOmvn
RAPID_FIRE_KEY=93e5b6e789msh21f8c0a25171494p1d5236jsnd91e3eb8aa6b
    """
    # rapid fire key
    rapidapi_key = "589e67089emsh9b790008204d6afp188707jsn9d24da4b417c"

    # authentication via app
    twitter_app_auth = {
        'consumer_key': "WG6dfDuyqgxQsbm2XkSCKzzuE",
        'consumer_secret': "os1fTHci5JSdfg1BWA98d1YrInvWG9fTc9eCOrZ2x9T7HgoVCV",
    }

    # model path
    path_models = "XGB_Default_Classifier.dat"

    path_to_clean = "not_classified_3M_4_1M.csv"

    df = pd.read_csv(path_to_clean)

    # make it managagebl
    #df = df[21165:21165 + 15000]

    df_dicts = df.to_dict(orient='records')

    # start the timer.
    start_time = datetime.now()

    current_time = start_time.strftime("%H:%M:%S")
    print("Script Started: ", current_time)

    # create the batch class
    bc = AccountClassifier(rapid_api_key=rapidapi_key, twitter_app_auth=twitter_app_auth,
                           model_path=path_models, data_file_path=path_to_clean, isBatch=False,
                           isPreprocessed=True)

    db_file = "newly_classified_4.db"

    # open the database connection.
    conn = sqlite3.connect(db_file)

    # for
    ind = 0
    for dict in df_dicts:
        result = bc.classify_preprocessed_by_row(dict)

        # if the result is empty, we don't need to add it
        if result is None:
            print(f'****======Completed Row: {ind} =======****')
            ind += 1
            continue

        # send the data into the sql
        insert_into_database(result, conn)
        print(f'****======Completed Row: {ind} =======****')
        ind += 1

    #r = df.apply(bc.classify_preprocessed_by_row, axis=1)

    end_time = datetime.now()

    # print the result
    # with pd.option_context('display.max_columns', None):  # more options can be specified also
    #     print(r)

    # stats
    bc.log_statistics(start_time, end_time)

    # add to classification bank
    #bc.update_classification_bank()

    #r.dropna(subset=["id"], inplace=True)
    # r.to_csv(f"classified_2_5M_3M_3.csv", index=False)

    # drop empty and then create csv

