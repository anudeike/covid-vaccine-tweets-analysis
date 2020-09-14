import pandas as pd
import json


path = "vaccine-2020-07-09.txt"
output_path = "data/basic_user_data.csv"

def main():

    # create a dataframe
    df = pd.DataFrame(columns=['id', 'screen_name', 'follower_count'])

    # goal
    # open the file and put username data into a dataframe and create a file for it
    with open(path, "r") as f:

        index = 0

        for line in f:
            # skip lines that are empty space
            if line.isspace():
                continue

            try:
                # deserialize
                tweet_data = json.loads(line)

                # vars
                id = tweet_data['user']['id_str']
                screen_name = tweet_data['user']['screen_name']
                followers = tweet_data['user']['followers_count']

                # insert info
                df.loc[index] = [id] + [screen_name] + [followers]

                if index > 2500:
                    break

                index += 1


            except(json.decoder.JSONDecodeError, TypeError) as e:

                # skip lines that don't serialize correctly should only be 2% of the lines
                print(e)
                continue

        # send the info to a csv file
        df.to_csv(output_path)
        #print(df.head(6))



    pass

main()