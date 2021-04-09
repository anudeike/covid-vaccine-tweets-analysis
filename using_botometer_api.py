import botometer
from dotenv import load_dotenv
import os
import json
import pandas as pd

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


#accounts = ['@clayadavis', '@onurvarol', '@aken_sir']
org_handles_df = pd.read_csv('organization_scores_2_0-100.csv')

# get the data in array form
accounts = org_handles_df['screen_name'][0:10].values


rows_arr = []

for screen_name, result in bom.check_accounts_in(accounts):

    # Do stuff with `screen_name` and `result`
    print(screen_name)
    print(result)

    if result['cap'] == None:
        continue # skip the entry

    english_cap = result["cap"]["english"]
    overall_display_scores = result["display_scores"]["english"]["overall"]

    row = {
        'screen_name': screen_name,
        'CAP': english_cap,
        'overall_score': overall_display_scores
    }

    rows_arr.append(row)


# send to dataframe to be exported to csv
df_export = pd.DataFrame(rows_arr)

#df_export.to_csv('some path name here.csv', index=False)
print(df_export)
