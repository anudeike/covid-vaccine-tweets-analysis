import pandas as pd
import botometer
from dotenv import load_dotenv
import os

# FORMAT OF THE DATA
# ,id,screen_name,follower_count

# set globals
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

# path to the data from the earlier df
path = "organization_officials_data/org_twitter_handle_only.csv"
output_path = "data/prepared_data/organization-split/organization_scores_2_400-500.csv"

df = pd.read_csv(path)

#out_df = pd.DataFrame(columns=["screen_name", "CAP", "astroturf", "fake_follower", "financial", "other", "overall", "self-declared", "spammer"])

def main():
    ## TO DO: GET THE INFORMATION FROM THE BOTOMETER AND PUT INTO THE LIST
    out_df = pd.DataFrame(
        columns=["screen_name", "CAP", "astroturf", "fake_follower", "financial", "other", "overall", "self-declared",
                 "spammer"])

    screen_names = df['Handle'].values


    # check the accounts
    for screen_name, result in bom.check_accounts_in(screen_names[400:500]):

        # target the row with that particular screen name
        #rowIndex = df.loc[df['screen_name'] == screen_name]

        # this will be appended to the new dataframe
        row = {}

        try:
            if (result["user"]["majority_lang"] == 'en'):
                # use the english results

                # for each row that'll be appended
                row = {
                    "screen_name": screen_name,
                    "CAP": result['cap']['english'],
                    "astroturf": result['display_scores']['english']['astroturf'],
                    "fake_follower": result['display_scores']['english']['fake_follower'],
                    "financial": result['display_scores']['english']['financial'],
                    "other": result['display_scores']['english']['other'],
                    "overall": result['display_scores']['english']['overall'],
                    "self-declared": result['display_scores']['english']['self_declared'],
                    "spammer": result['display_scores']['english']['spammer'],
                    "type": 'ORGANIZATION'
                }
            else:

                row = {
                    "screen_name": screen_name,
                    "CAP": result['cap']['universal'],
                    "astroturf": result['display_scores']['universal']['astroturf'],
                    "fake_follower": result['display_scores']['universal']['fake_follower'],
                    "financial": result['display_scores']['universal']['financial'],
                    "other": result['display_scores']['universal']['other'],
                    "overall": result['display_scores']['universal']['overall'],
                    "self-declared": result['display_scores']['universal']['self_declared'],
                    "spammer": result['display_scores']['universal']['spammer'],
                    "type": 'ORGANIZATION'
                }

            # append to dataframe
            out_df = out_df.append(row, ignore_index=True)

            print("{} has been processed.".format(screen_name))

        except Exception as e:
            # skip if error
            print("{} Could not be fetched: {}".format(screen_name, e))
            continue

    # send the info the the df
    out_df.to_csv(output_path)

main()

