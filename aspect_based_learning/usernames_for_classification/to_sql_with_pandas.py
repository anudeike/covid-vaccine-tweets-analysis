import csv
import sqlite3
import pandas as pd

"""
This is specifically for the conversation of the newlyclassified usernames 
"""
from glob import glob; from os.path import expanduser
conn = sqlite3.connect(r"newly_classified_3.db")

# turn to a pandas dataframe
df = pd.read_sql("select * from classified_accounts", conn)

# exclude the first index
df = df[["id", "prediction"]]

df.to_csv("out.csv", index=False)