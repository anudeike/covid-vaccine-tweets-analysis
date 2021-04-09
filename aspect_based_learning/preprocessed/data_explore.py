import pandas as pd
from collections import Counter
import json

df = pd.read_csv('2020-07_2020-09_preproccessed_5_2500000_to_3500000.csv')



# print(df[df["Tweet ID"] == 1281264233590804480])
# lets check whether the retweets have the same id