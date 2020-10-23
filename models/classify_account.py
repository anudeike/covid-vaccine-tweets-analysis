"""
This file classifies an account as a human or non-human.
The only input is the account name or id.
"""
import pickle
import botometer
from dotenv import load_dotenv
import os
import pandas as pd

load_dotenv()


def get_botometer_stats_single_account(bom_instance, screen_name):
    """
    Gets statistics about the account from the botometer API
    :param bom_instance: An authenticated instance of the botometer object
    :param screen_name: The screen name of the account
    :return: dict
    """

    result = bom_instance.check_account(screen_name)

    try:
        if (result["user"]["majority_lang"] == 'en'):
            # use the english results

            # for each row that'll be appended
            row = {
                "id": screen_name,
                "CAP": result['cap']['english'],
                # "astroturf": result['display_scores']['english']['astroturf'],
                "fake_follower": result['display_scores']['english']['fake_follower'],
                "financial": result['display_scores']['english']['financial'],
                "other": result['display_scores']['english']['other'],
                "overall": result['display_scores']['english']['overall'],
                "self-declared": result['display_scores']['english']['self_declared'],
                # "spammer": result['display_scores']['english']['spammer'],
            }
        else:

            row = {
                "id": screen_name,
                "CAP": result['cap']['universal'],
                # "astroturf": result['display_scores']['universal']['astroturf'],
                "fake_follower": result['display_scores']['universal']['fake_follower'],
                "financial": result['display_scores']['universal']['financial'],
                "other": result['display_scores']['universal']['other'],
                "overall": result['display_scores']['universal']['overall'],
                "self-declared": result['display_scores']['universal']['self_declared'],
                # "spammer": result['display_scores']['universal']['spammer'],
            }

        return row

    except Exception as e:
        # skip if error
        print("{} Could not be fetched: {}".format(id, e))

        return None

def load_model(model_path = "XGB_Default_Classifier.dat"):
    """
    Returns an instance of the model. The model must be created using the pickle library
    :param model_path: Path of the file that contains the model
    :return: the loaded model
    """
    return pickle.load(open(model_path, 'rb'))

def process_single_account(model, stats):

    print(f'Statistics from Botometer: {stats}')
    stats = list(stats.values())

    #prediction = model.predict([stats[1:]])

    # if prediction == 0:
    #     print(f'We classify {stats[0]} as a non-human user.')
    #
    # if prediction == 1:
    #     print(f'We classify {stats[0]} as a human user.')

    pass



# create a class that will hold all the information that is needed and run the actual model
class BotClassifier:
    def __init__(self, rapid_api_key, twitter_app_auth, model_path, data_file_path, separator=','):

        # init the botometer module
        self.bom = botometer.Botometer(wait_on_ratelimit=True,
                                       rapidapi_key=rapid_api_key,
                                       **twitter_app_auth)
        # load model
        self.model = self.load_model(model_path)

        # will be a dataframe of the account ids
        self.account_ids = self.get_account_ids(data_file_path, separator)
        print(self.account_ids)

    def get_account_ids(self, path, separ=','):
        """
        Reads the account ids from a file into a dataframe.
        :param path: Path to the csv, or tsv file
        :param separ: The separator of each value in the file. In CSV's this value is ','
        :return: a dataframe containing all the account ids
        """
        try:
            return pd.read_csv(path, sep=separ).values # put into an array
        except Exception as e:
            print(f'ERROR: {repr(e)}')


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
    twitter_app_auth = {
        'consumer_key': os.getenv('TWITTER_API_KEY'),
        'consumer_secret': os.getenv('TWITTER_API_SECRET'),
        'access_token': os.getenv('TWITTER_ACCESS_TOKEN'),
        'access_token_secret': os.getenv('TWITTER_ACCESS_SECRET'),
    }

    # model path
    path_models = "XGB_Default_Classifier.dat"

    #print(twitter_app_auth)
    # set up botometer - auth
    # bom = botometer.Botometer(wait_on_ratelimit=True,
    #                           rapidapi_key=rapidapi_key,
    #                           **twitter_app_auth)

    bc = BotClassifier(rapid_api_key=rapidapi_key, twitter_app_auth=twitter_app_auth,
                       model_path=path_models, data_file_path="test_accounts.csv")

    #process_single_account(model=model, stats=res)


