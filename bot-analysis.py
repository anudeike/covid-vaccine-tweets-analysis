import pandas as pd
import json
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
path = "data/basic_user_data.csv"

df = pd.read_csv(path)

print(df)
def main():
    ## TO DO: GET THE INFORMATION FROM THE BOTOMETER AND PUT INTO THE LIST
    pass

main()

