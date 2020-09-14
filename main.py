import numpy as np
import pandas as pd
from azure.cosmosdb.table.tableservice import TableService
from azure.cosmosdb.table.tablebatch import TableBatch
from azure.cosmosdb.table.models import Entity
from dotenv import load_dotenv
import os
import json
import uuid

load_dotenv()


# only able to get about 630 items into the database

"""
Goal is to use this to store the sample vaccine data 
in the azure storage so that it is a lot easier to retrieve
and process
"""

def get_table_service():
        """
        Gets the Table Service Provided by the information
        :return:
        """
        table_service = TableService(account_name='vaccinedatatraining', account_key=os.getenv("CONNECTION_KEY"))
        return table_service

def insert_into_table(table_service, table_name, data):
        """
        Insert information into a table. Must have a partition and row key.
        :param table_service: the service generated
        :param table_name: the name of the table
        :param data: the data that must be inserted.
        :return: None
        """
        table_service.insert_entity(table_name, data)

def main():

    # set up the table
    service = get_table_service()
    #service.create_table("master")

    # create a batch
    batch = TableBatch()

    # read in the file
    f = open("vaccine-2020-07-09.txt", "r")
    with open("vaccine-2020-07-09.txt", "r") as f:
        count = 1
        entry_error = 0
        for line in f:
            # skip lines that are empty space
            if line.isspace():
                continue

            # jerry rigged code to insert more than 630 entries (is a limit for some reason)
            if (count < 3930):
                count += 1
                continue


            try:
                # serialize the json string into a dictionary
                tweet_data = json.loads(line)

                # get the important information -- this will be the info that is posted to database
                batch_data = {
                    'PartitionKey': "master_batch",
                    'RowKey': uuid.uuid4().hex,
                    'user_name': tweet_data['user']['name'],
                    'user_screen_name': tweet_data['user']['screen_name'],
                    'followers': tweet_data['user']['followers_count'],
                    'user_description': tweet_data['user']['description'],
                    'body': line # should be the json body that we can post
                }

                # add it to the batch
                batch.insert_entity(batch_data)

                # for every 75 entries, make sure to post to azure
                if (count % 70 == 0):
                    print("posting....")
                    service.commit_batch("master", batch)
                    print("posted")

                    print("clearing batch...")
                    batch = TableBatch()
                    print("batch cleared.")


                print("completed entry {}.".format(count))

            except (RuntimeError, TypeError, json.decoder.JSONDecodeError) as e:
                print(e)

                # if there's an error just skip it.
                print("seems to be an error parsing this file at entry: " + str(count))
                entry_error += 1
                continue

            count += 1

        lost_rate = 100 * (entry_error / count)
        print("Total Entries Parsed: {}".format(count))
        print("Total Entries Lost: {}".format(entry_error))
        print("Percent Lost: {} %".format(lost_rate))

        # post the remainder
        service.commit_batch("master", batch)

main()



