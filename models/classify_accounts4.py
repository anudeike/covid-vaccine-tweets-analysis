"""
This file classifies an account as a human or non-human.
The only input is the account name or id.
"""
import pickle
import botometer
#from dotenv import load_dotenv
#import os
import pandas as pd
import numpy as np
import time
from datetime import datetime

#load_dotenv()

# def get_botometer_stats_single_account(bom_instance, screen_name):
#     """
#     Gets statistics about the account from the botometer API
#     :param bom_instance: An authenticated instance of the botometer object
#     :param screen_name: The screen name of the account
#     :return: dict
#     """
#
#     result = bom_instance.check_account(screen_name)
#
#     try:
#         if (result["user"]["majority_lang"] == 'en'):
#             # use the english results
#
#             # for each row that'll be appended
#             row = {
#                 "id": screen_name,
#                 "CAP": result['cap']['english'],
#                 # "astroturf": result['display_scores']['english']['astroturf'],
#                 "fake_follower": result['display_scores']['english']['fake_follower'],
#                 "financial": result['display_scores']['english']['financial'],
#                 "other": result['display_scores']['english']['other'],
#                 "overall": result['display_scores']['english']['overall'],
#                 "self-declared": result['display_scores']['english']['self_declared'],
#                 # "spammer": result['display_scores']['english']['spammer'],
#             }
#         else:
#
#             row = {
#                 "id": screen_name,
#                 "CAP": result['cap']['universal'],
#                 # "astroturf": result['display_scores']['universal']['astroturf'],
#                 "fake_follower": result['display_scores']['universal']['fake_follower'],
#                 "financial": result['display_scores']['universal']['financial'],
#                 "other": result['display_scores']['universal']['other'],
#                 "overall": result['display_scores']['universal']['overall'],
#                 "self-declared": result['display_scores']['universal']['self_declared'],
#                 # "spammer": result['display_scores']['universal']['spammer'],
#             }
#
#         return row
#
#     except Exception as e:
#         # skip if error
#         print("{} Could not be fetched: {}".format(id, e))
#
#         return None
#
# def load_model(model_path = "XGB_Default_Classifier.dat"):
#     """
#     Returns an instance of the model. The model must be created using the pickle library
#     :param model_path: Path of the file that contains the model
#     :return: the loaded model
#     """
#     return pickle.load(open(model_path, 'rb'))
#
# def process_single_account(model, stats):
#
#     print(f'Statistics from Botometer: {stats}')
#     stats = list(stats.values())
#
#     #prediction = model.predict([stats[1:]])
#
#     # if prediction == 0:
#     #     print(f'We classify {stats[0]} as a non-human user.')
#     #
#     # if prediction == 1:
#     #     print(f'We classify {stats[0]} as a human user.')
#
#     pass



# create a class that will hold all the information that is needed and run the actual model

class BotClassifier:
    def __init__(self, rapid_api_key, twitter_app_auth, model_path, data_file_path, separator=',', isBatch=True):

        # set whether you can use a batch
        self.isBatch = isBatch

        # init the botometer module
        self.bom = botometer.Botometer(wait_on_ratelimit=True,
                                       rapidapi_key=rapid_api_key,
                                       **twitter_app_auth)
        # load model
        self.model = self.load_model(model_path)

        # will be a dataframe of the account ids
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

                # # save the batch and then sleep if you hit batch size
                # if (count % batchSize == 0):
                #     # make predictions
                #     predictions = self.predict(np.array(botometer_data))
                #
                #     # zip
                #     zipped = list(zip(self.account_ids, predictions))
                #     df = pd.DataFrame(zipped, columns=["id", "prediction"])
                #
                #     out = out.append(df)
                #
                #     # send it out.
                #     out.to_csv(f'{outputFolder}/{outputFileName}_{count}.csv', index=False)
                #
                #     # reset the array
                #     botometer_data = []
                #
                #     # sleep
                #     print("is sleeping for {} seconds...".format(timeout))
                #     time.sleep(timeout)

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
    rapidapi_key = "311a57929dmshcbcfdfe7bf3ad75p1f9660jsn4039e85dd182"

    # authentication
    # twitter_app_auth = {
    #     'consumer_key': os.getenv('TWITTER_API_KEY'),
    #     'consumer_secret': os.getenv('TWITTER_API_SECRET'),
    #     'access_token': os.getenv('TWITTER_ACCESS_TOKEN'),
    #     'access_token_secret': os.getenv('TWITTER_ACCESS_SECRET'),
    # }

    # authentication via app - this should be a third app
    twitter_app_auth = {
        'consumer_key': "Hm0TKKMoVmSUDKxuw4dyDXKsH",
        'consumer_secret': "rd8YArYeuc0rgR0FMjBS3ctyxMQfCP0PAJQ5OZSWrrz9D1cSA2",
    }

    # model path
    path_models = "XGB_Default_Classifier.dat"

    bc = BotClassifier(rapid_api_key=rapidapi_key, twitter_app_auth=twitter_app_auth,
                       model_path=path_models, data_file_path="uniq_user_list_2020_07_to_2020_09.csv")

    now = datetime.now()

    current_time = now.strftime("%H:%M:%S")
    print("Script Started: ", current_time)

    # total range 300k to 400k
    bc.classify_batch(batchSize=100, timeout=90, outputFolder="li_data/classifications/fourth", outputFileName="li_usn_trial", cap=1100000, base_count=1096602)


