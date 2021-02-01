import time
import botometer
from dotenv import load_dotenv
import os
import pandas as pd
import numpy as np
import time
from multiprocessing import Process

load_dotenv()

df = pd.read_csv("2020-07_2020-09.csv", error_bad_lines=False)

out_df = df.iloc[:20000]

print("processed: ")
out_df.to_csv("2020-07_2020-09_clean.csv", index=False)
print("done")