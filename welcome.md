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
    
    bc.classify_batch()