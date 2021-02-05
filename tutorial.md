# How to classify twitter accounts as human or non-human from using Machine Learning  
This tutorial is based on a module called ```bot classifier```. It can be found at this [link](https://github.com/anudeike/covid-vaccine-tweets-analysis).  
  
## Introduction  
Millions of tweets and interactions per second, each one containing a viewpoint or opinion that influence the other; researchers have been drawn to Twitter as a way of studying interpersonal interactions and understanding how information spreads.   
  
Its no secret that in such a vast and rich social space there exists bots/organizations and other automated accounts that are not geniune individuals. These accounts tend to represent large corporations, non-profit organizations, political movements, and other large non-indivudals. Their actions on the platform are usually automated and their tweets are markedly similar, all in the goal of advertising a certain brand or organization.  
  
Any effective study into how information on Twitter affects interpersonal interaction needs to be able to separate geniune individual accounts from automated accounts.   
  
In this tutorial, we're going to explore how we could do that with a little help from machine learning and python.   
  
## Requirements  
To be able to follow along with this tutorial, you'll need a couple of things:  
  
 - A Twitter Account  
 - Python (pip and anaconda distributions will work fine)  
 - An IDE, Text Editor or an enviornment to use Python in  
 - A Rapid API account (don't worry, we'll walk through this one together)  
 - Basic knowledge of programming in Python  
  
### Creating a Rapid API account  
In this tutorial, we'll be using a module called [Botometer](https://github.com/IUNetSci/botometer-python). Botometer will gather relevant statstics about each account for us.   
  
 1. **Create a [Rapid API](https://rapidapi.com/marketplace) account**: Its completely free!  
 2. **Subscribe to the [Botometer Pro API](https://rapidapi.com/OSoMe/api/botometer-pro)**: There two other tiers that you can subscribe to but we recommend using the free tier which gives you 2000 requests per day for free. ($0.001 for every request after than).   
 3. **Look for the RapidAPI key**: You can find it under "Header Parameters" > "X-RapidAPI-Key". It should be a long hexdecimal string. **Save this value for later**, it is important.  
  
### Setting up Your Twitter Account  
To be able to use Botometer, you will need to set your account up to authorize requests.   
  
 1. **Create a Twitter Application [here](https://developer.twitter.com/en)**:   
 2. **Login into your Twitter Account**  
 3. **Follow the Steps to create a new twitter application [here](https://developer.twitter.com/en/apps)**: It'll ask for things like your application name, description and a website address (you can put your github link here if you'd like). You can also leave the "Callback URL" field empty.  
 4. **Accept the TOS**  
 5. **Navigate to the Keys and Access Tokens Tab**: It should be right of the settings tab. Your **Consumer Key** and **Consumer Secret** should be at the top of the page under "Application Settings".  
 6. **Click the Create my Access Token button.**: This'll generate an **access token** and an **access token secret**.   
 7. **This **[full tutorial](https://www.slickremix.com/docs/how-to-get-api-keys-and-tokens-for-twitter/)** shows you how to test these values to make sure that they work.**: For now we will assume that they work.  
 8. **Copy the Consumer Key, Consumer Secret, Access Token and the Access Token Secret**: Put them in a save place. We'll be using a .env file to store this information. If you want to learn how to use dotenv files, you can take a look at the [python-dotenv](https://pypi.org/project/python-dotenv/) module homepage.   
  
### Modules Required  
In this tutorial, we'll be using a lot of different modules to retrieve data, process the data, and use the data to train our model to make predictions.   
  
 - **Pandas  ```pip install pandas```:** This module is used to be read data from csv/tsv files. It is also useful for data exploration/wrangling  
 - **Matplotlib ```pip install matplotlib```:** This module is used for displaying information about the data in the form of graphs, charts and other graphics.  
 - **Numpy ```pip install numpy```:** Used for faster than built-in math calculations.  
 - **Sklearn ```pip install sklearn```:** This is where we will create our models/test them. There are a lot of useful metrics here  
 - **XgBoost ```pip install xgboost```:** This library contains faster and more robust models complete with a lot of validation and metrics.  
  
Whew! That was a lot of preparation, but each step is critical to building a model and making this pipeline work!  

## Getting Data  
The first step in creating model is getting data! The data that we need is going to come from the botometer module processing each account's statistics.   
Lets jump directly into the code!  
  
At the very top of the file import the pandas and botometer libraries. If you're using .env files, you'll need to import ```load_dotenv```.  
  
 import pandas as pd import botometer from dotenv import load_dotenv load_dotenv() # this loads the variables defined in the nearest .env file  Next we need to define the path to our datasets that contain a list of usernames (or ids) and a label that indicated whether they are automated or not. [Here](https://drive.google.com/file/d/18nd0UR8iL6g-aMWvn4ImN563bKSRzZKO/view?usp=sharing) is a sample dataset that only contains verified human accounts. Right below that, we'll define an output path for later.   
  
 path = "verified_humans.csv" # you can also use tsv output_path = "output_data.csv"  Now its time to use the keys that we got from the Twitter and RapidAPI API's. Define them as variables and then pass them into the botometer object to be able to create an authenticated Botometer instance that is ready to process some accounts!  
    
  
 rapidapi_key = os.getenv('RAPID_FIRE_KEY')     # authentication    
 twitter_app_auth = {    
     'consumer_key': os.getenv('TWITTER_API_KEY'),    
     'consumer_secret': os.getenv('TWITTER_API_SECRET'),    
     'access_token': os.getenv('TWITTER_ACCESS_TOKEN'),    
     'access_token_secret': os.getenv('TWITTER_ACCESS_SECRET'),    
 }    
        
    # you can copy this    
    bom = botometer.Botometer(wait_on_ratelimit=True,    
                              rapidapi_key=rapidapi_key,    
                              **twitter_app_auth)  
  From here, getting information on an account is pretty simple. Most of this example will be pulled from the botometer documentation.   
  
You can get information on a single account using one line of code  
  
 result = bom.check_account('@clayadavis')  
if you do not have the username you can use the ID as well. result = bom.check_account(1548959833)  
To check an array of accounts, you can use a simple for loop.  
  
 accounts = ['@clayadavis', '@onurvarol', '@jabawack'] for screen_name, result in bom.check_accounts_in(accounts): # Do stuff with `screen_name` and `result` print(result)  
The result (for each account) will look like this.  
  

     { "cap": { "english": 0.8018818614025648, "universal": 0.5557322218336633 }, "display_scores": { "english": { "astroturf": 0.0, "fake_follower": 4.1, "financial": 1.5, "other": 4.7, "overall": 4.7, "self_declared": 3.2, "spammer": 2.8 }, "universal": { "astroturf": 0.3, "fake_follower": 3.2, "financial": 1.6, "other": 3.8, "overall": 3.8, "self_declared": 3.7, "spammer": 2.3 } }, "raw_scores": { "english": { "astroturf": 0.0, "fake_follower": 0.81, "financial": 0.3, "other": 0.94, "overall": 0.94, "self_declared": 0.63, "spammer": 0.57 }, "universal": { "astroturf": 0.06, "fake_follower": 0.64, "financial": 0.3133333333333333, "other": 0.76, "overall": 0.76, "self_declared": 0.74, "spammer": 0.47 } }, "user": { "majority_lang": "en", "user_data": { "id_str": "11330", "screen_name": "test_screen_name" } } }  

  (This is copied directly from the github repo)  
 Here are the meaning of some of the vital fields  
 * **user**: Twitter user object (from the user) plus the language inferred from majority of tweets  
* **raw scores**: bot score in the [0,1] range, both using English (all features) and Universal (language-independent) features; in each case we have the overall score and the sub-scores for each bot class (see below for subclass names and definitions)  
* **display scores**: same as raw scores, but in the [0,5] range  
* **cap**: conditional probability that accounts with a score **equal to or greater than this** are automated; based on inferred language  
  
Meanings of the bot type scores:  
  
* `fake_follower`: bots purchased to increase follower counts   
* `self_declared`: bots from botwiki.org  
* `astroturf`: manually labeled political bots and accounts involved in follow trains that systematically delete content  
* `spammer`: accounts labeled as spambots from several datasets  
* `financial `: bots that post using cashtags  
 1. `other`: miscellaneous other bots obtained from manual annotation, user feedback, etc.  
  
For more information on the response object, consult the [API Overview](https://rapidapi.com/OSoMe/api/botometer-pro/details) on RapidAPI.  
  
### Getting the Data We Need  
The result of the botometer processing is quite large and contains many fields, most we don't need. For clarity, here are the fields that we need.  
  
 2. The Id/Username of the Account  
 3. The CAP: Complete Automation Probability. See the [Botometer FAQ](https://botometer.osome.iu.edu/faq) for more.  
 4. Astroturf  
 5. fake_follower  
 6. financial  
 7. other  
 8. overall  
 9. self_declared  
  
All of these values (except the CAP) can be take from the display_scores dictionary.  
The resulting code should look something like this:  
  

     # check the accounts    ids = df["ids"] # this is just a list of all of the account ids  
     for id, result in bom.check_accounts_in(ids):        
          # this will be appended to the new dataframe    
          row = {}    
            
          # we use a try-catch because we do not want it to stop execution if botometer fails to get stats on an account.  
     try:      if (result["user"]["majority_lang"] == 'en'):  
           # use the english results    
            row = {    
                        "id": id,    
                        "CAP": result['cap']['english'],    
                        "astroturf": result['display_scores']['english']['astroturf'],    
                        "fake_follower": result['display_scores']['english']['fake_follower'],    
                        "financial": result['display_scores']['english']['financial'],    
                        "other": result['display_scores']['english']['other'],    
                        "overall": result['display_scores']['english']['overall'],    
                        "self-declared": result['display_scores']['english']['self_declared'],    
                        "spammer": result['display_scores']['english']['spammer'],    
                        "type": types[count]    
                    }    
           else:    
             row = {    
                        "id": id,    
                        "CAP": result['cap']['universal'],    
                        "astroturf": result['display_scores']['universal']['astroturf'],    
                        "fake_follower": result['display_scores']['universal']['fake_follower'],    
                        "financial": result['display_scores']['universal']['financial'],    
                        "other": result['display_scores']['universal']['other'],    
                        "overall": result['display_scores']['universal']['overall'],    
                        "self-declared": result['display_scores']['universal']['self_declared'],    
                        "spammer": result['display_scores']['universal']['spammer'],    
                        "type": types[count]    
                    }    
                       
        # notify that we are done processing  
     print(f'{id} has been processed.') # you can then add it to a dataframe or do whatever you want to here with the row       except Exception as e:    
          # skip if error    
             print("{} Could not be fetched: {}".format(id, e))  
     continue  

 
### A Short Note about getting Data from large datasets  
Getting data using this library for large datasets is ... tricky, to say the least, because of Twitter's rate limiting and how Botometer chooses to handle this edge case. A larger discussion and a possible solution to this problem is can be found [here](https://github.com/anudeike/covid-vaccine-tweets-analysis/blob/master/welcome.md), but the gist of the issue is that Twitter's rate limiting only allows a certain amount of requests within a certain time period. This cap is very very low and processing large datasets (anything about 150 rows), can take multiple hours.   
  
If you find a better solution, feel free to reach out to use at ikechukwuanude@gmail.com or leave an issue in the github repo.

**Update 11/12:** Recently, I have found a couple of resources that are helpful for dealing with this rate limiting issue. I'll summarize them below

 1. **Using application authentication:** When you authenticate with Twitter, you can choose to authenticate with your account, or authenticate with an app. If you are processing large amounts of data, you should defintely choose the app authentication. You can do this by only including the `consumer_key` and `consumer_secret` keys in the `twitter_app_auth`. This should give you a limit of 450 requests per 15 minute window instead of the 250 you'd get if you use your account. As far as I know, there is no limit to the amount of apps you can create in an account, but you are limited to creating 3 in a 24 hour period. [You can find more information about the details here.](https://github.com/IUNetSci/botometer-python#rapidapi-and-twitter-access-details)
 2. **Use application Auth with RapidAPI:** Similar to Twitter, RapidAPI also has applicaiton authentication options. The rate limits are about the same per application with RapidAPI, but all of the requests will be counted towards the total for the entire account. This means that you will not be billed for each application separately but instead you will be billed for the usage across the account.

## Cleaning and Understanding Our Data
If you've reached this part, you should have a spreadsheet filled to the brim with data points.
Each row should look something like this:

    screen_name,CAP,astroturf,fake_follower,financial,other,overall,self-declared,spammer,type
    PatriciaMazzei,0.30618807248975083,1.3,0.2,0.0,0.8,0.2,0.1,0.0,HUMAN

The first line of the example data are the headers and the second line is an example row of data. 
**NOTE**: If you were not able to get enough data, or there simply isn't enough time to make thousands of requests, there will be a folder filled with example datasets that'll help you speedrun this tutorial. 

The first problem with this data is that we need to replace the `type` column values with an integer so that our model will be able to train with it. Here's a function that can get the job done:

    # row is a row from the dataframe that contains the data we collected.
    def label_conversion(row):  
      if row["type"] == 'human':  
	      return 1  
      elif row["type"] == 'ORGANIZATION':  
	      return 2  
      else:  
	      return 0 # bot
We also could've implemented this logic while we were fetching the data as well.

### Data Exploration 
Data exploration is a fundmental part of understanding trends in your data and improving your dataset. Since this tutorial is aimed at learning how to gather data and then train a model on it, I won't cover Data Exploration with the depth that it deserves but I will cover some of the highlights.

The first thing to do is to make sure that you separate our your datasets. 