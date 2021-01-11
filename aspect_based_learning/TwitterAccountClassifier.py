import pickle
import botometer
from dotenv import load_dotenv
import os
import pandas as pd
import numpy as np
import time
from datetime import datetime

load_dotenv()

class AccountClassifier:
    def __init__(self, rapid_api_key, twitter_app_auth, model_path, data_file_path="", separator=',', isBatch=True):

        # set whether you can use a batch
        self.isBatch = isBatch

        # init the botometer module
        self.bom = botometer.Botometer(wait_on_ratelimit=True,
                                       rapidapi_key=rapid_api_key,
                                       **twitter_app_auth)
        # load model
        self.model = self.load_model(model_path)

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

                # display what it is classified as
                if classification[0] == 1:
                    print(f'{account_id} is classified as HUMAN')
                else:
                    print(f'{account_id} is classified as NON-HUMAN')

                # notify
                print(f'{account_id} has been predicted.\n')

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

                # display what it is classified as
                if classification[0] == 1:
                    print(f'{account_id} is classified as HUMAN')
                else:
                    print(f'{account_id} is classified as NON-HUMAN')

                # notify
                print(f'{account_id} has been predicted.\n')

        except Exception as e:
            # skip if error
            print("{} Could not be fetched: {}".format(account_id, e))


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

    # create the batch class
    bc = AccountClassifier(rapid_api_key=rapidapi_key, twitter_app_auth=twitter_app_auth,
                       model_path=path_models, isBatch=False)

    bc.classify_single_account(account_id="johanvinet")

