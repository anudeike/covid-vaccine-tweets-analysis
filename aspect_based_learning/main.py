import pandas as pd
import json
import string
from nltk.corpus import stopwords

stop_words_en = set(stopwords.words('english'))

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

def filter_stop_words(x):
    print(x)
    return x

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
    #
    # stop words removes
    data['preprocess_data'] = data['preprocess_data'].apply(lambda x: filter_stop_words(x))
    #
    # # convert preprocessed list words to string
    # data['preprocess_str'] = data['preprocess_data'].apply(' '.join)

    return data

def main():
    t = get_tweets()
    pp = preprocess_tweets(t)

    #tweet_to_json()

    print(pp)

    # write to a csv
    pp.to_csv('test.csv', index=False)

    pass

main()