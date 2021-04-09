# this file only runs with the 3.6.9 + intepreter (Python 3.8)
import aspect_based_sentiment_analysis as absa
import pandas as pd
#
# # get the tweets
# tweets_df = pd.read_csv('pre_processed_tweets.csv')
#
# # example tweet
# tweet = tweets_df.values[2][0]
#
# # Break
text = ("We are great fans of Slack, but we wish the subscriptions "
        "were more accessible to small startups.")

recognizer = absa.aux_models.BasicPatternRecognizer()
nlp = absa.load(pattern_recognizer=recognizer)
completed_task = nlp(text=text, aspects=['slack', 'price'])
slack, price = completed_task.examples

absa.summary(slack)
absa.display(slack.review)

# # what does slack.scores mean??
# print(f'Virus: {virus.scores}')
# print(f'Vaccine: {vaccine.scores}')
# assert price.sentiment == absa.Sentiment.negative
# assert slack.sentiment == absa.Sentiment.positive

