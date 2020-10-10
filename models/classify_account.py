"""
This file classifies an account as a human or non-human.
The only input is the account name or id.
"""
import pickle
import botometer
from dotenv import load_dotenv
import os

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

def load_model(model_path = "MainModel_LogisticRegression.sav"):
    """
    Returns an instance of the model. The model must be created using the pickle library
    :param model_path: Path of the file that contains the model
    :return: the loaded model
    """
    return pickle.load(open(model_path, 'rb'))

def process_single_account(model, stats):

    stats = list(stats.values())

    prediction = model.predict([stats[1:]])

    if prediction == 0:
        print(f'We classify {stats[0]} as a non-human user.')

    if prediction == 1:
        print(f'We classify {stats[0]} as a human user.')

    pass



if __name__ == "__main__":

    rapidapi_key = os.getenv('RAPID_FIRE_KEY')

    # authentication
    twitter_app_auth = {
        'consumer_key': os.getenv('TWITTER_API_KEY'),
        'consumer_secret': os.getenv('TWITTER_API_SECRET'),
        'access_token': os.getenv('TWITTER_ACCESS_TOKEN'),
        'access_token_secret': os.getenv('TWITTER_ACCESS_SECRET'),
    }

    #print(twitter_app_auth)
    # set up botometer - auth
    bom = botometer.Botometer(wait_on_ratelimit=True,
                              rapidapi_key=rapidapi_key,
                              **twitter_app_auth)

    # TODO get id/name from the user
    # TODO decide whether the input is a whole stream of information of just one

    screen_name = "@clayadavis"

    res = get_botometer_stats_single_account(bom, screen_name)
    model = load_model()
    process_single_account(model=model, stats=res)


