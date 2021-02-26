import pandas as pd
import json
import string
import re


# print(stop_words_en)
# exit(0)
path = "vaccine-2020-07-09.txt"

def tweet_to_json(limit=3):

    tweet = {}

    with open(path, "r") as f:
        for x in range(0, limit):
            f.readline()

        line = f.readline()

        tweet = json.loads(line)

    with open("test.json", "w") as j:
        json.dump(tweet, j)

def get_full_tweet_text(tweet_data: dict):
    """
    Gets the following data:
    - full tweet text
    - screen_name
    - location: if there is one

    should return a dictionary with these values
    :param tweet_data:
    :return: dict
    """

    tweet_info = {}

    # check if truncated
    if tweet_data["truncated"]:
        # get the extended tweet
        tweet_info["full_text"] = tweet_data["extended_tweet"]["full_text"]
        # get the username
        tweet_info["screen_name"] = tweet_data["user"]["screen_name"]

        return tweet_info
    else:
        # if not retweeted
        if "retweeted_status" not in tweet_data:
            tweet_info["full_text"] = tweet_data["text"]
            tweet_info["screen_name"] = tweet_data["user"]["screen_name"]
            return tweet_info

        # if retweet
        if tweet_data["retweeted_status"]:
            tweet_info["screen_name"] = tweet_data["retweeted_status"]["user"]["screen_name"]

            # and you're truncated
            if tweet_data["retweeted_status"]["truncated"]:
                tweet_info["full_text"] = tweet_data["retweeted_status"]["extended_tweet"]["full_text"]

                return tweet_info

            tweet_info["full_text"] = tweet_data["text"]
            return tweet_info


def preprocess_tweet(full_text):
    """
    Pre-Processes the tweet text to be used.
    1. Convert to Lowercase
    2. Remove URL
    3. remove retweets and cc
    4. Remove Hashtags
    5. Remove Emojis
    6. Remove html tags
    7. Remove extra spaces
    8. Remove Punctuation
    :param full_text: full tweet text
    :return: processed text
    """

    # convert to lower case
    full_text = full_text.lower()

    # url removes
    full_text = re.sub(r'(https|http)?:\/(\w|\.|\/|\?|\=|\&|\%)*\b', '', full_text)
    full_text = re.sub(r'www\.\S+\.com', '', full_text)

    # removes retweets & cc
    full_text = re.sub(r'rt| cc', '', full_text)

    # hashtags removes
    full_text = re.sub(r'#\S+', '', full_text)

    # user mention removes
    full_text = re.sub(r'@\S+', '', full_text)

    # emoji
    full_text = re.sub(r'[^\x00-\x7F]+', '', full_text)

    # html tags
    full_text = re.sub(r'<.*?>', '', full_text)

    # removes extra spaces
    full_text = re.sub(r' +', ' ', full_text)
    full_text = re.sub(r'/(\r\n)+|\r+|\n+|\t+/', " ", full_text)


    return full_text


def get_tweets(limit=3):

    # create a dataframe
    df = pd.DataFrame()


    # goal
    # open the file and put username data into a dataframe and create a file for it
    with open(path, "r") as f:

        index = 0
        for line in f:

            # skip lines that are empty space
            if line.isspace():
                continue

            try:
                # deserialize
                tweet_data = json.loads(line)

                # get the full text of the tweet
                try:
                    key_tweet_info = get_full_tweet_text(tweet_data)

                    key_tweet_info["full_text"] = preprocess_tweet(key_tweet_info["full_text"])

                    # print(f'{index}: TWEET TEXT: {key_tweet_info["full_text"]}')

                    # insert info
                    df = df.append(key_tweet_info, ignore_index=True)

                except Exception as e:
                    # in final code: will change to skipping this iteration and going to the next one.
                    print(f'ERROR: {e}')
                    continue # skip



                if index > limit:
                    break

                index += 1


            except(json.decoder.JSONDecodeError, TypeError) as e:

                # skip lines that don't serialize correctly should only be 2% of the lines
                print(e)
                continue

        # send the info to a csv file
        print(df)
        df.to_csv("pre_processed_tweets.csv", index=False)

    return


# this is the latest function
def fetch_tweets(limit = 5, path="2020-07_2020-09_reduced.csv"):

    df = pd.read_csv(path, error_bad_lines=False)

    #df = df.head(limit).values

    # for each line
    for row in df:
        print(row)
    pass

def main():

    #get_tweets(limit=50)
    path_to_clean = "reduced/2020-07_2020-09_reduced_1500000_to_2500000.csv"

    # get the tweets
    df = pd.read_csv(path_to_clean)

    # clean the tweet content column
    df['proccessed_tweet'] = df['Tweet Content'].apply(preprocess_tweet)

    with pd.option_context('display.max_columns', None):  # more options can be specified also
        print(df.head(5))

    df.to_csv("preprocessed/2020-07_2020-09_preproccessed_5_1500000_to_2500000.csv", index=False)
    # export the tweet out
    pass

main()