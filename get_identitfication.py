import pandas as pd
import json


path = "data/tar/verified-human/verified-2019_tweets.json"
output_path = "data/verified-human-usernames.csv"

def main():

    # create a dataframe
    df = pd.DataFrame(columns=['id', 'screen_name', 'follower_count', 'type'])

    # goal
    # open the file and put username data into a dataframe and create a file for it
    with open(path, "r") as f:
        tweets = json.load(f)

        index = 0
        for tweet in tweets:
            id = tweet['user']['id_str']
            screen_name = tweet['user']['screen_name']
            followers = tweet['user']['followers_count']

            # add the info to the dataframe
            df.loc[index] = [id] + [screen_name] + [followers] + ['human_verified']

            # inc
            index += 1

        df.to_csv(output_path)


main()