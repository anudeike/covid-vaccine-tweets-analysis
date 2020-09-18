import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# this file will plot the information out from the relative categories
org_path = "data/prepared_data/organization-split/organization_scores.csv"
bot_path = "data/prepared_data/bot-split/bot_stats_all.csv"
human_path = "data/prepared_data/human-split/human_all_prepared"

#print("done")
# turn into dataframes
org_df = pd.read_csv(org_path)
bot_df = pd.read_csv(bot_path)
human_df = pd.read_csv(human_path)

#print("done")
# get the the CAP for each of them
org_cap = org_df['CAP'].values
bot_cap = bot_df['CAP'].values
human_cap = human_df['CAP'].values

#print("done")
# set the bins
bins = np.linspace(0, 1, 100)

# print some of the average and std
print("Average CAP of Humans: {}, Standard Dev = {}".format(np.mean(human_cap), np.std(human_cap)))
print("Average CAP of Bot: {}, Standard Dev = {}".format(np.mean(bot_cap), np.std(bot_cap)))
print("Average CAP of Organizations: {}, Standard Dev = {}".format(np.mean(org_cap), np.std(org_cap)))

#print("done")
plt.hist(org_cap, bins, alpha=0.5, label='CAP_ORG')
plt.hist(bot_cap, bins, alpha=0.5, label='CAP_BOT')
plt.hist(human_cap, bins, alpha=0.5, label='CAP_HUMAN')
plt.legend(loc='upper right')
plt.title("Distribution of Complete Automation Probability (CAP) \nof Humans, Bots and Organizations")
plt.show()