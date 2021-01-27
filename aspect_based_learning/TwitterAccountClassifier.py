import pickle
import botometer
from dotenv import load_dotenv
import os
import pandas as pd
import numpy as np
import time
from datetime import datetime

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
        vaccine, virus = nlp(text, aspects=["vaccine", "virus"])

        return [{
            "vaccine_scores": vaccine.scores,
            "vaccine_overall_sent": vaccine.sentiment
        },{
            "virus_scores": virus.scores,
            "virus_overall_sent": virus.sentiment
        }]

        pass

    def fetch_from_classification_bank(self, uid):

        if self.classification_bank is None:
            return None

        df = self.classification_bank

        # search
        res = df.loc[df['id'] == uid]

        if res.empty:
            print(f'{uid} could not be found in database.')
            return None

        print(f'{uid} fetched.\n')
        print(res)
        # get the type
        pred = res.values

        if pred[0][1] == 0:
            return pred[0][1], 'NON-HUMAN'

        return pred[0][1], 'HUMAN'

    def classify_preprocessed(self):
        """
        Classifies the accounts given one at a time. Returns a dataframe containing the id and the predicted_label
        :return: Dataframe
        """

        # this function does not work on batches
        if self.isBatch:
            raise ValueError("Classifier set to work with Batch processing (isBatch = True). Consider using classify_batch() instead.\n"
                             "To use this function you must set (isPreproccessed = True) as well.")

        # place an if statement here that checks in the separate data frame
        # create a blank_dataframe
        out = pd.DataFrame()

        row_list = []

        # put into a 2d array
        p_data = self.prep_df.values

        # to be inserted data
        row_data = {}

        # this data will be inserted into the classification bank
        # these are the accounts that were classified by botometer that were not in the classification
        missing_classified_accounts = []


        # for each row
        for entry in p_data:

            row_data["id"] = entry[1]
            row_data["tweet_text"] = entry[0]

            # get the type of account
            try:
                # get the type

                # this fetches from the classification bank
                fetched = self.fetch_from_classification_bank(row_data["id"])

                if fetched is None:

                    # if is not in the classification bank, then fetch from botometer
                    type = self.classify_single_account(row_data["id"])
                    row_data["prediction"] = type["prediction"] # this is the numerical value
                    row_data["class_type"] = type["type"]

                    # then add it to the list to be appended to the classification bank
                    missing_classified_accounts.append([row_data['id'], row_data['prediction']])
                else:
                    row_data["prediction"] = fetched[0]
                    row_data["class_type"] = fetched[1]


                # analyze sentiment
                sent = self.get_sentiment(row_data["tweet_text"])

                # put into the row
                row_data["vaccine_scores"] = [sent[0]["vaccine_scores"]]
                row_data["vaccine_overall"] = sent[0]["vaccine_overall_sent"]

                row_data["virus_scores"] = [sent[1]["virus_scores"]]
                row_data["virus_overall"] = sent[1]["virus_overall_sent"]

                #append the dictionary as a new row
                print(f"row: {row_data}")
                out = out.append(row_data, ignore_index=True)



            except Exception as e:
                print(f"[top-level]: {repr(e)}\n")
                continue # skip this one

        # show the time elapsed

        print("Missing Accounts: ")
        print(missing_classified_accounts)
        out.to_csv("analysis_data_test.csv", index=False)

        return 0

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



if __name__ == "__main__":

    # rapid fire key
    rapidapi_key = os.getenv('RAPID_FIRE_KEY')

    # authentication
    # twitter_app_auth = {
    #     'consumer_key': os.getenv('TWITTER_API_KEY'),
    #     'consumer_secret': os.getenv('TWITTER_API_SECRET'),
    #     'access_token': os.getenv('TWITTER_ACCESS_TOKEN'),
    #     'access_token_secret': os.getenv('TWITTER_ACCESS_SECRET'),
    # }

    # authentication via app
    twitter_app_auth = {
        'consumer_key': os.getenv('TWITTER_API_KEY'),
        'consumer_secret': os.getenv('TWITTER_API_SECRET'),
    }

    # model path
    path_models = "models/XGB_Default_Classifier.dat"

    # file path
    prep_path = "pre_processed_tweets.csv"

    # create the batch class
    bc = AccountClassifier(rapid_api_key=rapidapi_key, twitter_app_auth=twitter_app_auth,
                       model_path=path_models, data_file_path=prep_path, isBatch=False, isPreprocessed=True, path_to_classified="li_trial_master.csv")


    bc.classify_preprocessed()

