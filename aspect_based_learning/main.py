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

    # check if truncated
    if tweet_data["truncated"]:
        # get the extended tweet
        return tweet_data["extended_tweet"]["full_text"]
    else:
        # if not retweeted
        if "retweeted_status" not in tweet_data:
            return tweet_data["text"]

        # if retweet
        if tweet_data["retweeted_status"]:
            # and you're truncated
            if tweet_data["retweeted_status"]["truncated"]:
                return tweet_data["retweeted_status"]["extended_tweet"]["full_text"]

            return tweet_data["text"]


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
    full_text = re.sub(r'rt|cc', '', full_text)

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

    # leave punctuation out the model should be able to handle it
    # # punctuation
    # full_text = full_text.re.sub('[{}]'.format(string.punctuation), '')

    #print(full_text)

    return full_text

def get_tweets(limit=3):

    # create a dataframe
    #df = pd.DataFrame(columns=['tweets'])


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
                    full_tweet_text = get_full_tweet_text(tweet_data)

                    tweet_content = preprocess_tweet(full_tweet_text)

                    print(f'{index}: TWEET TEXT: {tweet_content}')

                    # insert info
                    # df.loc[index] = [tweet_content]
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
        #print(df)

    #return df




def main():
    t = get_tweets(limit=10)
    #pp = preprocess_tweets(t)

    #tweet_to_json(limit=4)

    #print(pp)

    # write to a csv
    #pp.to_csv('test.csv', index=False)

    pass

main()