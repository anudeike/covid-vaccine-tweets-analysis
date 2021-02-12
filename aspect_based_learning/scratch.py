import pandas as pd
import sqlite3

# THIS IS FOR TESTING THE DATA BASE
# paths = ['2020-07_2020-09_csvfiles/tweet_processing_test_1-1500_second_batch.csv',
#          '2020-07_2020-09_csvfiles/tweet_processing_test_1500-5500_second_batch.csv',
#          '2020-07_2020-09_csvfiles/tweet_processing_test_5500-8000_second_batch.csv',
#          '2020-07_2020-09_csvfiles/tweet_processing_test_8000-10500_second_batch.csv',
#          '2020-07_2020-09_csvfiles/tweet_processing_test_10500-15500_second_batch.csv',
#          '2020-07_2020-09_csvfiles/tweet_processing_test_15500-30000_second_batch.csv',
#          '2020-07_2020-09_csvfiles/tweet_processing_test_30000-40000_second_batch.csv',
#          '2020-07_2020-09_csvfiles/tweet_processing_test_40000-50000_second_batch.csv',
#          '2020-07_2020-09_csvfiles/tweet_processing_test_50000-60000_second_batch.csv',
#          '2020-07_2020-09_csvfiles/tweet_processing_test_60000-75000_second_batch.csv']
#
# df = pd.read_csv(paths[0])
#
# for i in range(1, len(paths)):
#     df = df.append(pd.read_csv(paths[i]), ignore_index=True)
#
#
# df.dropna(subset=["No."], inplace=True)
#
# # create a connection
# cnx = sqlite3.connect(r'human_classified_sentiment_processed.db')
#
# # send to sqlite
# df.to_sql(name="tweet_information_second_batch", con=cnx, if_exists="append")

# FOR REDUCING SIZE
hi = 500000
lo = 150000

df = pd.read_csv('2020-07_2020-09.csv', error_bad_lines=False)

df = df[lo:hi]

df.to_csv(f"2020-07_2020-09_reduced_{lo}_to_{hi}.csv")