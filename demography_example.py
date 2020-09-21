import json
from demographer import process_tweet
from demographer.indorg import IndividualOrgDemographer

demographer_list = [ IndividualOrgDemographer(setup="balanced")]

print("DEMOGRAPHER: {}".format(demographer_list[0]))

with open('vaccine-2020-07-09.txt', 'r') as f:
    tweet = json.loads(f.readline())
    print(process_tweet(tweet, demographer_list))