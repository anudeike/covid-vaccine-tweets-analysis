import json
from demographer import process_tweet
from demographer.indorg import IndividualOrgDemographer
from demographer.gender import CensusGenderDemographer

demographer_list = [ IndividualOrgDemographer(setup="balanced"),
                     CensusGenderDemographer(use_classifier=True, use_name_dictionary=True)]

#print("DEMOGRAPHER: {}".format(demographer_list[0]))

with open('vaccine-2020-07-09.txt', 'r') as f:
    tweet = json.loads(f.readline())

    json_obj = json.dumps(process_tweet(tweet, demographer_list))
    print(json_obj)