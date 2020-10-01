import json
from demographer import process_tweet
from demographer.indorg import IndividualOrgDemographer
from demographer.gender import CensusGenderDemographer
from demographer.indorg_neural import NeuralOrganizationDemographer

demographer_list = [ IndividualOrgDemographer(setup="balanced"),
                     CensusGenderDemographer(use_classifier=True, use_name_dictionary=True)]

#print("DEMOGRAPHER: {}".format(demographer_list[0]))

# NeuralOrganizationDemographer() doesn't seem to work even with itn th e file
# """
# Traceback (most recent call last):
#   File "C:/Users/Ikechukwu Anude/Documents/vaccine-data-processing/demography_example.py", line 9, in <module>
#     NeuralOrganizationDemographer()]
#   File "C:\Users\Ikechukwu Anude\AppData\Roaming\Python\Python36\site-packages\demographer\indorg_neural.py", line 45, in __init__
#     clear_devices=True)
#   File "C:\Users\Ikechukwu Anude\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\training\saver.py", line 1960, in import_meta_graph
#     **kwargs)
#   File "C:\Users\Ikechukwu Anude\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\framework\meta_graph.py", line 744, in import_scoped_meta_graph
#     producer_op_list=producer_op_list)
#   File "C:\Users\Ikechukwu Anude\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\util\deprecation.py", line 432, in new_func
#     return func(*args, **kwargs)
#   File "C:\Users\Ikechukwu Anude\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\framework\importer.py", line 391, in import_graph_def
#     _RemoveDefaultAttrs(op_dict, producer_op_list, graph_def)
#   File "C:\Users\Ikechukwu Anude\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\framework\importer.py", line 158, in _RemoveDefaultAttrs
#     op_def = op_dict[node.op]
# KeyError: 'IteratorV2'
# """

# the goal is to get demographic information on each of the bot/human data sets and make sure that organizations are sorted into bots

vaccine_tweets_path = "vaccine-2020-07-09.txt"

# this has about 20983 tweets
celebrity_tweets = "demographer_testing_data/celebrity-2019_tweets.json"

# verified human tweets
human_tweets = "demographer_testing_data/verified-2019_tweets.json"

# verified bot tweets
bot_tweets = "demographer_testing_data/botwiki-2019_tweets.json"

# get the json file
def open_tweets_file(json_path, num_tweets = 10):
    data = []
    with open(json_path) as f:
        data = json.loads(f.read())

    return data

def process_tweet_array(arr, correct_answer='org'):

    count = 0
    correct = 0
    for tweet in arr:
        res = process_tweet(tweet, demographer_list)
        #print(f'Processed Tweet {count}\nResult: {json.dumps(res)}')

        # if it predicts correctly then
        if(res["indorg_balanced"]["value"] == correct_answer):
            correct += 1

        count += 1

    print(f'Percentage Correct: {(correct/len(arr)) * 100}%')


def main():
    tweets = open_tweets_file(bot_tweets, num_tweets=1000)
    process_tweet_array(tweets)

    # process all the tweets


main()