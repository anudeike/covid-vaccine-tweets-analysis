import pandas as pd
import sqlite3
import json
from collections import Counter
import numpy as np

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

#======== FOR REDUCING SIZE
lo = 5500000
hi = 5500020

def skip_function(ind):
    if ind < hi and ind > lo:
        return False

    return True


# TO DO
df = pd.read_csv('2020-07_2020-09.csv', skiprows=lambda x: skip_function(x), error_bad_lines=False, names="No.,Is retweet?,Tweet ID,Post Date,User Display Name,User ID,User Info,User Location,Tweet Content,Number of Quotes,Number of Replies,Number of Likes,Number of Retweets,Raw link 1,Solved link 1,Raw link 2,Solved link 2,Raw link 3,Solved link 3,Raw link 4,Solved link 4,Raw link 5,Solved link 5".split(','))

#df = df[lo:hi]
print(df)
df.to_csv(f"reduced/2020-07_2020-09_reduced_{lo}_to_{hi}_full.csv")

# ====== for converting the sql to csv
#
# conn = sqlite3.connect('human_classified_sentiment_processed.db')
# df = pd.read_sql_query("SELECT * from tweet_information_second_batch", conn)
#
# df.to_csv("output_3_19_2021.csv", index=False)
#
# # turn to json
# df_dict = df[0:10].to_dict(orient="records")
#
# with open("test_json_example.json", "w+") as f:
#     json.dump(df_dict, f)




# ids_in_classification_bank = set(pd.read_csv("classification_bank.csv")["id"].values)
# #
# # #print("retrived from c_bank")
# #
# # """ USING COUNTER """
# user_ids_in_preprocessed = Counter(pd.read_csv('preprocessed/2020-07_2020-09_preproccessed_4100000_to_4800000_full.csv', error_bad_lines=False)["User ID"].values)
#
# print("retrived user_ids")
#
# names_values_separated = list(zip(*user_ids_in_preprocessed.most_common()))
#
# top_n = 110000
# print(f"Getting most common {top_n} users: ")
# #print(f'length of the most common users: {most_common_users}')
# #print(f'ALL USERS: {all_most_common_users[0:100]}')
#
# print(f'Top {top_n} users:')
# print(f'Number of tweets they made: {sum(names_values_separated[1][0:top_n])}.\nThis is {np.round(sum(names_values_separated[1][0:top_n])/700000, 4)}% of the total tweets')
#
# unqiue_most_common_usernames = set(names_values_separated[0][0:top_n])
# difference = unqiue_most_common_usernames - ids_in_classification_bank
#
# print(f'Number of {top_n} top users not in classification bank: {len(difference)}')
#
#
# diff_uniq_df = pd.DataFrame(data=list(difference), columns=["id"])
# diff_uniq_df.to_csv("usernames_for_classification/not_classified_4_8M_5_5M.csv",index=False)