import pandas as pd

paths = ['tweet_processing_test_1200.csv',
         'tweet_processing_test_4000.csv',
         'tweet_processing_test_4000-8000.csv',
         'tweet_processing_test_8000-12000.csv',
         'tweet_processing_test_12000-18000.csv',
         'tweet_processing_test_18000-20000.csv']

df = pd.read_csv(paths[0])

for _ in range(1, len(paths)):
    paths