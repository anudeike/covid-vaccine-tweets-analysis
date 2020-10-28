# Welcome to BotClassifier

## Introduction:
This model is aimed at classifying twitter accounts as individuals or non-individuals. Non-individuals include both bots and organizations. 

The model uses the [botometer library](https://github.com/IUNetSci/botometer-python) to gather key statistics about a twitter account and passes the information into a model that will classify whether the account represents and individual or not. 

This model uses the [XGBoost Library](https://xgboost.readthedocs.io/en/latest/) to provided a boosted Random Forest Model to classifiy the accounts.

## Preparation

### Before Install
To be able to use this library, you will need three things:

 - A Twitter Account: You can create a new twitter account or use an existing one. After getting a twitter account, you will need to [obtain a consumer key, consumer secret, access token and access token secret.](https://developer.twitter.com/en/docs/authentication/oauth-1-0a/obtaining-user-access-tokens)

- A Rapid API key for the Botometer API. Follow the instructions at [this link](https://docs.rapidapi.com/docs/keys) to get an API key for the botometer endpoint. 
- A csv or tsv of accounts that you would like to classify

For more detailed instructions, take a look at the 'Prior to Utilizing Botometer' section of the [botometer-python repo.](https://github.com/IUNetSci/botometer-python)

### Installation
You can install the library from pip (not implemented yet)

    pip install twitterbotclassifer

If you prefer to use this as a command line utility feel free to download this repo (not implemented yet).

## Usage
Using this library is easy. If you installed with pip, you can import the class at the top of the script

    from twitterBotClassifier import BotClassifier
   From there, you'll need to set up the credentials from RapidAPI and Twitter.
   

    rapid_api_key = os.getenv('RAPID_FIRE_KEY')
    twitter_app_auth = {  
    'consumer_key': os.getenv('TWITTER_API_KEY'),  
    'consumer_secret': os.getenv('TWITTER_API_SECRET'),  
    'access_token': os.getenv('TWITTER_ACCESS_TOKEN'),  
    'access_token_secret': os.getenv('TWITTER_ACCESS_SECRET'),  }

** Note,  this method of installing the .env variable requires third-party modules.**

Create a BotClassifer object with the rapid_api_key, twitter_app_auth and the path to the data file

    bc = BotClassifier(rapid_api_key=rapidapi_key, twitter_app_auth=twitter_app_auth,  
                   model_path=path_models, data_file_path="test_accounts.csv")
 
 **Personal Note: There will not need to be a model_path**

Lastly, call classify_batch() to run the classifier and get a result.
    
    output = bc.classify_batch()

The output will be a dataframe that looks like this:
| index | id           | labels |
| ----- | ------------ | ------ |
| 0     | exxonmobil   | 0      |
| 1     | claydavis    | 1      |
| 2     | investopedia | 0      |
| 3     | aken\_sir    | 1      |

- **Id**: The name (or id) of the twitter account
- **labels**: the label assigned by the classifier. **'0' means the model classified the account as non-human, while '1' means that model classified the account as human.**
- **index:** a number assigned to the row.

You can choose to export the results to a csv file by adding the path to the file as a parameter.

    output = bc.classify_path(output_path = "example.csv")

## Important Notes
### Console Messages
When you run the model, you should see a line of console message outputs that looks something like this:

    Account 0: exxonmobil has been processed.
    Account 1: clayadavis has been processed.
    Account 2: investopedia has been processed.
    Account 3: generalelectric has been processed.

The format of these messages is: ```Account{count}:{id} has been processed```. This helps you keep track of what has processed. 

You can also set a batch_size and timeout (more on this later), and when the batch size has been hit, there will be a console message ```Batch Size {batch_size} has been reached. Saving batch to file at {file_path}``` followed by the another console message: ``` Sleeping for {timeout} seconds.```

### Timeout and Batch Size
Under the hood, this classifer gets the required information from the botometer endpoint for every account and then classifies all the accounts at once. 

For small lists of accounts (75 or less accounts), this is practical and works well. If you have more than 150 accounts, they the program will run into some issues. This is because for every account, the program must ping the Twitter API and because of the twitter rate limitations, the program can be denied service for a set amount of time (usually 10-15 minutes) before it can make requests again. Botometer has a built in way of dealing with this, but in our experience, it was a little slower than we would've liked. 

Therfore we decided to create a way circumvent this limitation (its not perfect).

To workaround this limitation, you can set the **batch_size**, **timeout**, and **folder_path** parameters when you call the ```classify_batch()``` function. 

- **batch_size**: This is the size of every "batch" of accounts. When a multiple of this number of accounts has been processed, the program will predict all the values of the current batch and store the result in a dataframe that will be written the folder specified by the **folder_path** variable. Default value is 100 (it is also the size we recommend)
- **timeout**: This is the amount of seconds that the program will wait before it begins to process a new batch. Default value is 80 (this is recommended). 
- **folder_path**: Every time a new batch is finished, the resulting dataframe (along with the results from other batches) will be written to a file that will be stored in this folder. 

For example: If the ```batch_size = 100```, the ```timeout = 80```, the program will get information for the next 100 accounts. When it hits the 100th account in the batch, it will then make predictions on all the accounts, put the result in a dataframe and then export the result to a csv file. After that, it will sleep for 80 seconds. When it wakes up, it will continue to process the next 100 accounts, adding the predictions from those accounts to the dataframe. 

**NOTE** : Even though this system significiantly improves the speed at which things are classified, it is not perfect and will sometimes lead to stalls but only after 1000+ accounts. If you have a suggestion for a smarter/faster way of dealing with this problem, we're all ears in the github issues section.

[There should be a video to go along with this to help explain it] 