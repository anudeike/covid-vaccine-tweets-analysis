import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# get the data needed
org_path = "data/prepared_data/organization-split/organization_scores.csv"
bot_path = "data/prepared_data/bot-split/bot_stats_all.csv"
human_path = "data/prepared_data/human-split/human_all_prepared"

# turn into dataframes
org_df = pd.read_csv(org_path)
bot_df = pd.read_csv(bot_path)
human_df = pd.read_csv(human_path)

# create the model with 100 trees
model = RandomForestClassifier(n_estimators=100,
                               bootstrap=True,
                               max_features='sqrt')

