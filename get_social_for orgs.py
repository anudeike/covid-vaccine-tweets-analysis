import luckysocial # can use to get social media provided a name
import pandas as pd

path = "organization_officials_data/fortune_500_data_set.csv"

df = pd.read_csv(path)

#print(df["twitter"])

twitter_links = df["twitter"].dropna()


for link in twitter_links:

    if "https" in link:
        print(link.split('/')[-1])
    else:
        print(link)
