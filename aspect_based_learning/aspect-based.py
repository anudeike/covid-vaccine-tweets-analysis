# this file only runs with the 3.6.9 + intepreter (Python 3.8)
import aspect_based_sentiment_analysis as absa
import pandas as pd

# get the tweets
tweets_df = pd.read_csv('pre_processed_tweets.csv')

# example tweet
tweet = tweets_df.values[2][0]

# Break
nlp = absa.load()

vaccine, virus = nlp(tweet, aspects=["vaccine", "virus"])


# what does slack.scores mean??
print(f'Virus: {virus.scores}')
print(f'Vaccine: {vaccine.scores}')
# assert price.sentiment == absa.Sentiment.negative
# assert slack.sentiment == absa.Sentiment.positive

