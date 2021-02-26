import pandas as pd
from collections import Counter
import json

#df = pd.read_csv('2020-07_2020-09_preproccessed_2.csv')

inp_dict = {'Python': "A", 'Java': "B", 'Ruby': "C", 'Kotlin': "D"}

if (m := inp_dict.get('Python')):
    print(m)
    print("The key is present.\n")


else:
    print("The key does not exist in the dictionary.")

# print(df[df["Tweet ID"] == 1281264233590804480])
# lets check whether the retweets have the same id