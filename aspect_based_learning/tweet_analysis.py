import pandas as pd



def main():
    df = pd.read_csv("li_trial_master.csv")

    o = df.loc[df["id"] == "JusLorsdfa"]
    b = df.loc[df["id"] == "NewsThisSecond"]
    print(int(b.head(1)["prediction"]))
    pass

main()