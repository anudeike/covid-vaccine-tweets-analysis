import pickle
import botometer
from dotenv import load_dotenv
import os
import pandas as pd
import numpy as np
import time
from datetime import datetime
import sqlite3

# sent analysis
import aspect_based_sentiment_analysis as absa
nlp = absa.load()

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
        self.seen_tweets = {} # this holds the tweets that have been seen and classified





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

                # display what it is classified as
                if classification[0] == 1:
                    print(f'{account_id} is classified as HUMAN')
                    return {"prediction": classification[0], "type": "HUMAN"}
                else:
                    print(f'{account_id} is classified as NON-HUMAN')
                    return {"prediction": classification[0], "type": "NON-HUMAN"}



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

                # display what it is classified as
                if classification[0] == 1:
                    print(f'{account_id} is classified as HUMAN')
                    return {"prediction": classification[0], "type": "HUMAN"}
                else:
                    print(f'{account_id} is classified as NON-HUMAN')
                    return {"prediction": classification[0], "type": "NON-HUMAN"}



        except Exception as e:
            # raise so the external function can catch it
            raise ValueError("{} Could not be fetched: {}".format(account_id, e))


    def classify_batch(self, batchSize = 100, timeout = 120, outputFolder = None, outputFileName = "default", cap=100, base_count = 0):
        """
        Classifies accounts in batches.
        :param batchSize: The amount of accounts
        :param timeout: The amount of time the program waits between batches
        :param output_folder: The folder where all the batch files will be saved
        :return:
        """
        # this function does not work on batches
        if not self.isBatch:
            raise ValueError("Classifier set to work with single entry processing (isBatch = False). Consider using classify() instead.")

        out = pd.DataFrame(columns=['id', 'prediction'])
        botometer_data = []

        count = base_count

        batch_ids = []

        # for each of the names in the list -> this would work best on the muted list
        for id, result in self.bom.check_accounts_in(self.account_ids[count:cap]):

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
                        # "spammer": result['display_scores']['english']['spammer'],
                    }

                    # append to the batch_data
                    botometer_data.append(np.array(list(row.values())[1:]))

                    # append to the batch ids
                    batch_ids.append(id)
                    # notify
                    print(f'Account {count}: {id} has been processed.\n')

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

                    # append to the batch_data
                    botometer_data.append(np.array(list(row.values())[1:]))
                    batch_ids.append(id)
                    # notify
                    print(f'Account {count}: {id} has been processed.\n')


                # save the batch and then sleep if you hit batch size
                if (count % batchSize == 0):

                    # make predictions
                    print(f"Making predictions on {count}...")
                    predictions = self.predict(np.array(botometer_data))

                    # zip
                    zipped = list(zip(batch_ids, predictions))

                    df = pd.DataFrame(zipped, columns=["id", "prediction"])

                    out = out.append(df)

                    # send it out.
                    out.to_csv(f'{outputFolder}/{outputFileName}_{count}.csv', index=False)

                    # reset the array
                    botometer_data = []
                    batch_ids = []

                    # sleep
                    print("is sleeping for {} seconds...".format(timeout))
                    time.sleep(timeout)

                # increment by one every loop
                count += 1

            except Exception as e:
                # skip if error
                print("{} Could not be fetched: {}".format(id, e))

                count += 1
                continue


        # export the remaining
        # make predictions
        if len(botometer_data) == 0 :
            out.to_csv(f'{outputFolder}/{outputFileName}_{count+1}.csv', index=False)
            return out

        predictions = self.predict(np.array(botometer_data))

        # zip
        zipped = list(zip(batch_ids, predictions))
        df = pd.DataFrame(zipped, columns=["id", "prediction"])

        out = out.append(df)

        # send it out.
        out.to_csv(f'{outputFolder}/{outputFileName}_{count}.csv', index=False)

        return out

    def get_sentiment(self, text):

        # get the sentiment in all of the target words
        vaccine, virus, vaccines, vaccination = nlp(text, aspects=["vaccine", "virus", "vaccines", "vaccination"])

        return [{
            "vaccine_scores": vaccine.scores,
            "vaccine_overall_sent": vaccine.sentiment
        },{
            "virus_scores": virus.scores,
            "virus_overall_sent": virus.sentiment
        },{
            "vaccines_scores": vaccines.scores,
            "vaccines_overall_sent": vaccines.sentiment
        },{
            "vaccination_scores": vaccination.scores,
            "vaccination_overall_sent": vaccination.sentiment
        }]

        pass

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

            # classify the account
            # this fetches from the classification bank
            screen_name = row['User ID']
            fetched = self.fetch_from_classification_bank(screen_name)

            if fetched is None:

                # for now we just skip if it is not in the classification bank
                print(f'Since {screen_name} was not found in the classification bank, it will be skipped.')
                self.tweet_index += 1
                return None
                # THIS CODE ONLY HAPPENS IF WE ARE ANALYZING THE DATA THAT IS NOT FOUND IN THE CLASSIFICATION BANK
                # # if is not in the classification bank, then fetch from botometer
                # type = self.classify_single_account(screen_name)
                # row["prediction"] = type["prediction"]  # this is the numerical value
                # row["class_type"] = type["type"]
                #
                # # then add it to the list to be appended to the classification bank
                # self.missing_classified_accounts.append({"id": screen_name, "prediction": row['prediction']})
            else:
                # if it is in the bank

                print(f'Prediction for {screen_name}: {fetched}')

                # if human
                if int(fetched) == 0:
                    # not counting non humans
                    print(
                        f'{screen_name} has been classified as a non-human and therefore will not be counted in the final analysis.')
                    self.tweet_index += 1
                    return None

                else:
                    row["prediction"] = fetched # should only be humans

            proccessed_tweet_content = row["proccessed_tweet"]

            # should be valid in python 3.8
            if (match := self.seen_tweets.get(proccessed_tweet_content)):
                # if there is a match, then get the value from the seen_tweets
                print(f"**[{proccessed_tweet_content}]** has already been seen.")
                # vaccine scores
                row["vaccine_score_neutral"] = match[0]["vaccine_scores"][0]
                row["vaccine_score_negative"] = match[0]["vaccine_scores"][1]
                row["vaccine_score_positive"] = match[0]["vaccine_scores"][2]
                row["vaccine_overall_sentiment"] = match[0]["vaccine_overall_sent"]

                # virus scores
                row["virus_scores_neutral"] = match[1]["virus_scores"][0]
                row["virus_scores_negative"] = match[1]["virus_scores"][1]
                row["virus_scores_positive"] = match[1]["virus_scores"][2]
                row["virus_overall_sentiment"] = match[1]["virus_overall_sent"]

                # vaccines scores
                row["vaccines_scores_neutral"] = match[2]["vaccines_scores"][0]
                row["vaccines_scores_negative"] = match[2]["vaccines_scores"][1]
                row["vaccines_scores_positive"] = match[2]["vaccines_scores"][2]
                row["vaccines_overall_sentiment"] = match[2]["vaccines_overall_sent"]

                # vaccination scores
                row["vaccination_scores_neutral"] = match[3]["vaccination_scores"][0]
                row["vaccination_scores_negative"] = match[3]["vaccination_scores"][1]
                row["vaccination_scores_positive"] = match[3]["vaccination_scores"][2]
                row["vaccination_overall_sentiment"] = match[3]["vaccination_overall_sent"]

                # success!
                self.successful_analysis += 1

                # increment the index
                print(f'Processed tweet # {self.tweet_index}')
                self.tweet_index += 1
                return row

            else:
                # get the sentiment
                sent = self.get_sentiment(proccessed_tweet_content)
                print(f'Sentiment Analysis Result: {sent}')

                # vaccine scores
                row["vaccine_score_neutral"] = sent[0]["vaccine_scores"][0]
                row["vaccine_score_negative"] = sent[0]["vaccine_scores"][1]
                row["vaccine_score_positive"] = sent[0]["vaccine_scores"][2]
                row["vaccine_overall_sentiment"] = sent[0]["vaccine_overall_sent"]

                # virus scores
                row["virus_scores_neutral"] = sent[1]["virus_scores"][0]
                row["virus_scores_negative"] = sent[1]["virus_scores"][1]
                row["virus_scores_positive"] = sent[1]["virus_scores"][2]
                row["virus_overall_sentiment"] = sent[1]["virus_overall_sent"]

                # vaccines scores
                row["vaccines_scores_neutral"] = sent[2]["vaccines_scores"][0]
                row["vaccines_scores_negative"] = sent[2]["vaccines_scores"][1]
                row["vaccines_scores_positive"] = sent[2]["vaccines_scores"][2]
                row["vaccines_overall_sentiment"] = sent[2]["vaccines_overall_sent"]

                # vaccination scores
                row["vaccination_scores_neutral"] = sent[3]["vaccination_scores"][0]
                row["vaccination_scores_negative"] = sent[3]["vaccination_scores"][1]
                row["vaccination_scores_positive"] = sent[3]["vaccination_scores"][2]
                row["vaccination_overall_sentiment"] = sent[3]["vaccination_overall_sent"]

            # success!
            self.successful_analysis += 1

            # increment the index
            print(f'Processed tweet # {self.tweet_index}')
            self.tweet_index += 1

            # add the tweet to the the seen tweets
            self.seen_tweets[proccessed_tweet_content] = sent

            return row

        except Exception as e:
            print(f"[top-level]: {repr(e)}\n")
            self.failed_analysis += 1
            return None
        pass


    def log_statistics(self, start, end, success, failed, amt_accts_added):
        """
        Logs General Statistics for the Program
        :param start: Start time
        :param end: End time
        :param success: Amount of tweets successfully processed
        :param failed: Amound of tweets unsuccessfully processed
        :return: None
        """

        total = success + failed
        elapsed = end - start
        elapsed_seconds = elapsed.total_seconds()

        # get the timer in the time in hours
        elapsed_hours = divmod(elapsed_seconds, 3600)
        elapsed_minutes = divmod(elapsed_seconds, 60)

        # display the stats
        print("\n\n==========================STATISTICS============================\n")
        print(f'Time Elapsed: {0} hours, {elapsed_minutes[0]} minutes, and {elapsed_minutes[1]} seconds.')
        print(f'Evaluated {total} Tweets. \n{success} successful evaluations\n {failed} failed evaluations')
        print(f'{amt_accts_added} new accounts added to classification bank.')

    def log_statistics_new(self, start, end):
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
        mdf = pd.DataFrame(self.missing_classified_accounts)

        self.classification_bank = self.classification_bank.append(mdf)

        # send to csv
        self.classification_bank.to_csv("classification_bank.csv", index=False)


def insert_into_database(row_info, conn):

    # convert to a dataframe
    row_info = pd.DataFrame(row_info, index=[row_info['No.']])

    # add stuff to the database and then commit
    # cur = conn.cursor()
    # cur.execute(sql_code, row_info)
    # cur.commit()
    row_info.to_sql(name="tweet_information_second_batch", con=conn, if_exists="append")


if __name__ == "__main__":

    # rapid fire key
    rapidapi_key = os.getenv('RAPID_FIRE_KEY')

    # authentication via app
    twitter_app_auth = {
        'consumer_key': os.getenv('TWITTER_API_KEY'),
        'consumer_secret': os.getenv('TWITTER_API_SECRET'),
    }

    # model path
    path_models = "models/XGB_Default_Classifier.dat"

    path_to_clean = "preprocessed/2020-07_2020-09_preproccessed_5_1500000_to_2500000.csv"

    df = pd.read_csv(path_to_clean)

    # define the database:
    db_file = r"human_classified_sentiment_processed.db"

    # reduce the df to something managable
    start_rows = 700000
    num_rows = 800000

    df = df[start_rows:num_rows]

    # turn it into an array of dicts
    df_dicts = df.to_dict(orient='records')

    # start the timer.
    start_time = datetime.now()

    current_time = start_time.strftime("%H:%M:%S")
    print("Script Started: ", current_time)

    # create the batch class
    bc = AccountClassifier(rapid_api_key=rapidapi_key, twitter_app_auth=twitter_app_auth,
                           model_path=path_models, data_file_path=path_to_clean, isBatch=False,
                           isPreprocessed=True, path_to_classified="classification_bank.csv")


    #open the database connection.
    conn = sqlite3.connect(db_file)


    #go through each row
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

    # r = df.apply(bc.classify_preprocessed_by_row, axis=1)
    #
    # # print the result
    # with pd.option_context('display.max_columns', None):  # more options can be specified also
    #     print(r)

    end_time = datetime.now()

    # stats
    bc.log_statistics_new(start_time, end_time)

    # add to classification bank
    #bc.update_classification_bank()

    #r.to_csv(f"2020-07_2020-09_csvfiles/tweet_processing_test_{start_rows}-{num_rows}_second_batch.csv", index=False)

    # drop empty and then create csv
    #r.dropna(subset=["No."], inplace=True)
