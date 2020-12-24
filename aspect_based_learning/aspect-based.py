# # this file only runs with the 3.6.9 + intepreter (Python 3.8)
#
# import aspect_based_sentiment_analysis as absa
#
# nlp = absa.load()
#
# text = ("We are great fans of Slack, but we wish the subscriptions "
#         "were more accessible to small startups.")
#
# slack, price = nlp(text, aspects=["slack", "price"])
#
#
# # what does slack.scores mean??
# print(slack)
# # assert price.sentiment == absa.Sentiment.negative
# # assert slack.sentiment == absa.Sentiment.positive

import pandas as pd
import json
import string


# print(stop_words_en)
# exit(0)
path = "vaccine-2020-07-09.txt"

def tweet_to_json():

    tweet = {}

    with open(path, "r") as f:
        f.readline()
        f.readline()
        line = f.readline()

        tweet = json.loads(line)

    with open("test.json", "w") as j:
        json.dump(tweet, j)


def get_tweets():

    # create a dataframe
    df = pd.DataFrame(columns=['tweets'])


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

                tweet_content = tweet_data['text']

                # insert info
                df.loc[index] = [tweet_content]

                if index > 4:
                    break

                index += 1


            except(json.decoder.JSONDecodeError, TypeError) as e:

                # skip lines that don't serialize correctly should only be 2% of the lines
                print(e)
                continue

        # send the info to a csv file
        #print(df)

    return df


def preprocess_tweets(data):

    # convert to lower case
    data['preprocess_data'] = data['tweets'].str.lower()

    # url removes
    data['preprocess_data'] = data['preprocess_data'].str.replace(r'(https|http)?:\/(\w|\.|\/|\?|\=|\&|\%)*\b', '')
    data['preprocess_data'] = data['preprocess_data'].str.replace(r'www\.\S+\.com', '')
    #
    # removes retweets & cc
    data['preprocess_data'] = data['preprocess_data'].str.replace(r'rt|cc', '')
    #
    # hashtags removes
    data['preprocess_data'] = data['preprocess_data'].str.replace(r'#\S+', '')

    # user mention removes
    data['preprocess_data'] = data['preprocess_data'].str.replace(r'@\S+', '')
    #
    # emoji
    data['preprocess_data'] = data['preprocess_data'].str.replace(r'[^\x00-\x7F]+', '')
    #
    # html tags
    data['preprocess_data'] = data['preprocess_data'].str.replace(r'<.*?>', '')
    #
    # removes extra spaces
    data['preprocess_data'] = data['preprocess_data'].str.replace(r' +', ' ')
    #
    # punctuation
    data['preprocess_data'] = data['preprocess_data'].str.replace('[{}]'.format(string.punctuation), '')

    print(data['preprocess_data'])

    return data

def main():
    #t = get_tweets()
    #pp = preprocess_tweets(t)

    tweet_to_json()

    #print(pp)

    # write to a csv
    #pp.to_csv('test.csv', index=False)

    pass

main()