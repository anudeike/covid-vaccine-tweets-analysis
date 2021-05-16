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
      
    
 ```rapidapi_key = os.getenv('RAPID_FIRE_KEY')     # authentication      
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
```
 From here, getting information on an account is pretty simple. Most of this example will be pulled from the botometer documentation.     
    
You can get information on a single account using one line of code    
    
 result = bom.check_account('@clayadavis')    
if you do not have the username you can use the ID as well. result = bom.check_account(1548959833)    
To check an array of accounts, you can use a simple for loop.    
    
 accounts = ['@clayadavis', '@onurvarol', '@jabawack'] for screen_name, result in bom.check_accounts_in(accounts): # Do stuff with `screen_name` and `result` print(result)    
The result (for each account) will look like this.    
    
  
````{ "cap": { "english": 0.8018818614025648, "universal": 0.5557322218336633 }, "display_scores": { "english": { "astroturf": 0.0, "fake_follower": 4.1, "financial": 1.5, "other": 4.7, "overall": 4.7, "self_declared": 3.2, "spammer": 2.8 }, "universal": { "astroturf": 0.3, "fake_follower": 3.2, "financial": 1.6, "other": 3.8, "overall": 3.8, "self_declared": 3.7, "spammer": 2.3 } }, "raw_scores": { "english": { "astroturf": 0.0, "fake_follower": 0.81, "financial": 0.3, "other": 0.94, "overall": 0.94, "self_declared": 0.63, "spammer": 0.57 }, "universal": { "astroturf": 0.06, "fake_follower": 0.64, "financial": 0.3133333333333333, "other": 0.76, "overall": 0.76, "self_declared": 0.74, "spammer": 0.47 } }, "user": { "majority_lang": "en", "user_data": { "id_str": "11330", "screen_name": "test_screen_name" } } }
````   
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
    
  
 # check the accounts    ids = df["ids"] # this is just a list of all of the account ids     for id, result in bom.check_accounts_in(ids):          
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
  
` screen_name,CAP,astroturf,fake_follower,financial,other,overall,self-declared,spammer,type PatriciaMazzei,0.30618807248975083,1.3,0.2,0.0,0.8,0.2,0.1,0.0,HUMAN`
The first line of the example data are the headers and the second line is an example row of data.   
**NOTE**: If you were not able to get enough data, or there simply isn't enough time to make thousands of requests, there will be a folder filled with example datasets that'll help you speedrun this tutorial.   
  
The first problem with this data is that we need to replace the `type` column values with an integer so that our model will be able to train with it. Here's a function that can get the job done:  
  
 row is a row from the dataframe that contains the data we collected. 
 ``` 
 def label_conversion(row):      
	 if row["type"] == 'human':    
         return 1    
      elif row["type"] == 'ORGANIZATION':    
         return 2    
      else:    
         return 0 # bot 
 ``` 
We also could've implemented this logic while we were fetching the data as well.  

## Data Exploration   
Data exploration is a fundmental part of understanding trends in your data and improving your dataset. Since this tutorial is aimed at learning how to gather data and then train a model on it, I won't cover Data Exploration with the depth that it deserves but I will cover some of the highlights.  
  
The first thing to do is to make sure that you separate our your datasets.
You should have a separate dataset for each one of the treatment groups. In this case, our treatment groups are bot, individuals, and organizations. They'll be stored in `bot_account_names.csv`, `individual_account_names.csv` and `organization_account_names.csv` (the file names don't have to match these, they can be whatever you want). 

### Exploring Each of the Datasets
There's a lot of ways to do data exploration, but one of the best/first places to start is basic summary analysis of the mean and standard deviation of the distribution. 

The first step to calculating the mean and standard deviation for your datasets is to first get the dataset from a csv into dataframe.

This can be done with the following code
``df = pandas.read_csv("path_to_csv.csv")`` 
 From there, we can use `print(df.describe())` to print out a couple of summary statistics.
 Your output should look something like this:
 ```
        Unnamed: 0        No.  ...  Raw link 5  Solved link 5
count   11.000000  11.000000  ...         0.0            0.0
mean     5.000000   7.000000  ...         NaN            NaN
std      3.316625   3.316625  ...         NaN            NaN
min      0.000000   2.000000  ...         NaN            NaN
25%      2.500000   4.500000  ...         NaN            NaN
50%      5.000000   7.000000  ...         NaN            NaN
75%      7.500000   9.500000  ...         NaN            NaN
max     10.000000  12.000000  ...         NaN            NaN

 ```
 (It might not look *exactly* like this, but the left hand labels (count, mean, std etc) should be there.)
 
 Can't see the columns you want? Don't worry, `df.describe()` returns a dataframe and like any other dataframe you can simply select the column you want by first assigning the result to a variable and then using this syntax: `describe_df["column_name"]`. 
 If you want to be able to see the entire output you can simply export it to a csv file using the `to_csv()` functions like this: `describe_df.to_csv("path_to_output_csv.csv")`. 

You can read more about this function in the documentation [here](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.describe.html)

### Plotting Historgrams 
Though numbers are good and all, chances are, the person who will look at the data would like to see some kind of visual aid to contextualize the mean and standard deviation. A perfect tool for this task is called a histogram.
A histogram is a series of bar charts that  shows the distribution of values in a sample. They tend to look something like this. 

 ![an example of a histogram](https://rb.gy/fxkgp0)
 Without much context, this might be a bit frustrating to read, but from a glance this histogram and tell us a couple of things
 * The Gross Monthly Salary is skewed to the left
 * The median is likely lower than the mean of the distribution
 * And most of the values fall between $800 and $1000

There is more you could take from it, but for now, those are the most important observations.
***
**Creating a Histogram in Python**
Creating a histogram in Python is relatively easy.
 
First make sure you have matplotlib installed.

Then, at the top of your file, include the following line: `import matplotlib.pyplot as plt` to include and use the matplotlib package

Next, Dataframes have a helper function called `df.hist()` that can generate a histogram for us which makes it easy to display a histogram of your data.

**Don't forget to include plt.show() at the end of the program so you can see the graph**
Your output should look something like this:
![And example output of df.hist()](https://res.cloudinary.com/cheezitromansh/image/upload/v1619139351/example_tkeghp.png)

For more control over the visual output (maybe you wanted it a different color, or all the graphs in a straight line), you should check out the base [matplotlib.pyplot](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.hist.html) documention and the [pandas.hist()](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.hist.html) documentation for more details.

Part of the fun here is playing around with the code and comparing different distributions to one another. You can perform certain statistics tests such as the t-test, and correlation tests but none of this will be covered in this tutorial.
### Intepreting the Data
We've already covered some of this with drawing conclusions from the histogram above, but this time, we'll take it a step further and compare features, such as CAP and overall score, between different classes of accounts (ie. Bot, human and organization). 

The first thing we need to do is to pick a feature that we will be comparing across all classes. For the sake of simplicity, I picked the CAP, or Complete Automation Score. This score ranges from 0-1, with 0 being the most automated and 1 being the most automated.

The way to get these specific columns from your dataframe is outlined below:
```
org_cap = org_df['CAP'].values  
bot_cap = bot_df['CAP'].values  
human_cap = human_df['CAP'].values
```

Now we have the values for the CAP feature for each class in their own array. 

To plot it, all we need to do is to call the `plt.hist()` function. You can read more about it [here.](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.hist.html) 

Before we do this, we should establish how many bins we should have. This will dictate the thinkness of the bars in our histogram. Instead of the standard amount, I'll specifiy 100 bins with this bit of code:
`bins = np.linspace(0, 1, 100)`
**Note**: np.linspace comes from the numpy library. Read all about it [here.](https://numpy.org/doc/stable/reference/generated/numpy.linspace.html)

The last thing we have to do now is to simply plot all the histograms!
```
plt.hist(org_cap, bins, alpha=0.5, label='CAP_ORG')  
plt.hist(bot_cap, bins, alpha=0.5, label='CAP_BOT')  
plt.hist(human_cap, bins, alpha=0.5, label='CAP_HUMAN')  
plt.legend(loc='upper right')  
plt.title("Distribution of Complete Automation Probability (CAP) \nof Humans, Bots and Organizations")
plt.show()
```
When you run this code, you should get something that looks a bit like this:
![enter image description here](https://res.cloudinary.com/cheezitromansh/image/upload/v1619154342/example_pwhdid.png)

As you can see, the bars are labeled, colored coded (human - green, bot - orange, organization - blue). For the sake of time and sanity, I'll leave it up to you to figure out how to create a plot for each one of the features.

But just looking at the plot, we can see some **massive** differences in the distributions between the 3 classes. Here are a couple of things you could notice at a glance:
* The Org and Bot distributions are a lot further right, clustering closer to 1 than the human distributions
* The Bot distribution appears to have the most right skewed frequency, followed by the Organization frequency
* The human distribution is a lot more spead out than both the Org and Both Distributions
* Though the Bot and Org distributions are close to each other on the graph, they have relatively small spread ( standard deviations) and are still very distinct from each other.
* The Bot distribution is heavily skewed to the right with most of its values occuring very close to 1
* The Org distribution seems to be centered around 0.8

Looking at the means and standard deviations for the dataset, it seems like these observations coroborate:
```
Average CAP of Humans: 0.46322320956704793, Standard Dev = 0.2427303756982515
Average CAP of Bot: 0.9497471030462905, Standard Dev = 0.0579888920629633
Average CAP of Organizations: 0.7755106538863092, Standard Dev = 0.08051632412876976
```
Feel free to do the same for the rest of the features.

But after you do, its a good idea to **pause** and take a second to carefully intepret these findings.

Since our goal is to predict whether an account is made by a individual (human) or non-individual (everyone else), try asking the question **how can I use this information to tell individuals from non-individuals?**

Seriously, do it. Write some of the ideas down!

Done?

Here's a couple of my answers: 
* We could use a statstical test to be able to discriminate between the two
	* Pros: Would be fast to train (not much training in all honesty), the botometer website recommends a statistical test
	* Cons: Could only be so accurate and might not work across multiple features (or at least its hard to do that -- not sure, never tried it)
* We could create a machine learning algorithm to be able to tell the two groups apart!
	* Pros: Its machine learning, its fun, it can work very well across multiple different feature dimensions
	* Cons: We don't have much data (less than 60,000 data points) and it is type of resource heavy to train
 
It wouldn't be much fun if I didn't choose the machine learning route, but if you want to try the statistical test and compare the performance of the two systems, go right on a ahead.

# Building a Machine Learning Model

Here's the fun part! In this step, we'll build a machine learning model, train it, and validate it. Despite the complexity typically involved in building, training and testing, we can simpliffy the process massively using `sci-kit learn`, `numpy` and `xgboost` libraries.

## Deciding what models to Use
There is a vast array of different machine learning algorithms. Each algorithm has their own use-cases, pros and cons and it's up to us to decide on the algorithm that is the most useful for our test case. 

### Our Test Case
Our test case looks like this:
* We have bunch of features (CAP, astroturf, overall etc) that describe a particular account
* We also need a binary output (individual or non-individual - organizations count as non-individual)

[This is a good article to help get familiar with some of the classification algorithms out there.](https://serokell.io/blog/classification-algorithms) 

But, for the sake of simplicity, we'll consider two separate types of algorithms: Random Forest and Logistic Regression.
### Logistic Regression
Logistic Regression is a type of classification algorithm that uses a logistic curve to classify data. It is mainly used for binary classification but cannot be adapted for multi-class or multi-label classification. [Here's](https://holypython.com/log-reg/logistic-regression-pros-cons/) a good article on some of the advantages and disadvantages of logisitic regression but the most important advantages and disadantages are as follows:
* **Unlikely to Overfit**: Since its a linear model, it won't over fit the data and will still maintain generalizability between datasets.
* **Efficient**: Not very resource heavy and easy to compute
* **Linearity**: [disadvantage] This is a **huge** disadvantage for our use case as we can't be certain that this is a linear problem. 

### Random Forest (XGBoost)
Random Forest is an algorithm based on the **decision tree**. It is a lot easier to explain that logistic regression because it works very similarly to how we might think of classifying objects. 

**How a Decision Tree Works**
For example, say we wanted to classify an object as an apple or not an apple (an apple being defined as a red, round, fruit). We might as questions such as: "Is the object a fruit?". If it is not a fruit, then it cannot be an apple, so we categorize it as not an apple. But if it is a fruit, we can ask another question: "Is it round?". If the fruit is not round, then we can say that it is not an apple. However if the fruit is round then we can ask our third and final question: "Is the round fruit red?". If the answer is yes, then we can classify the object as an apple, otherwise, it is not an apple.

**From Decision Tree to Decision Forest**
Similarly to how a forest is consisted of a bunch of different trees, a Random Forest is consisted of a bunch of different **uncorrelated** decision trees that act as an ensemble. Each individual decision tree returns a class prediction and the prediction with the most amount of votes becomes the result of the Random Forest. 

There's a little bit more to the story than this, including what it means for trees to be **uncorrelated** and what it means for trees to work in an **ensemble**. 

[You can read more about Random Forest in this blog post.](https://towardsdatascience.com/understanding-random-forest-58381e0602d2)   

## Importing the Necessary Libraries
Before we start writing functional code, we should install the necessary libraries. Here's the list of modules that you'll need to import
```
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import xgboost
from sklearn import preprocessing
import pickle
from sklearn.metrics import plot_confusion_matrix, accuracy_score
import pandas as pd # you should already have this installed
```
Lets's breakdown what each import does:
* `from sklearn import model_selection`: This line imports a class from the sklearn module that allows us to search from the best model out of a group of models. If this sounds confusing right now, it'll make more sense when we test multiple models in order to get the one with the highest performance.
* `from sklearn.linear_model import LogisticRegression`: This is one of the types of models that we will be using to discern non-individuals from individuals.
* `import xgboost`: This provides another type of model called a "Random Forest Classifier". [You can learn more about Random Forest Classifiers here](https://towardsdatascience.com/understanding-random-forest-58381e0602d2). **Note**: XGBoost is not a typical Random Forest Classifier and features a lot of enhancements compared to the more traditional versions. [You can read more about it's modifications here.](https://towardsdatascience.com/a-beginners-guide-to-xgboost-87f5d4c30ed7)
* `from sklearn import preprocessing`: This will help us prepare our data for the training and testing phases of the model. 
*  `import pickle`: This module will allow us to save the parameters, weights and biases of the machine learning model so we can use it on other datasets. 
* `from sklearn.metrics import plot_confusion_matrix, accuracy_score`: This will help us display the accuracu of the models in clearly readable formats.

**Note:** If you don't have any of these libraries installed you can install them using pip. (`pip install sklean`, `pip install pickle`, `pip install xgboost`)

## Getting the Data And Making It Easy to Use
By now you should have a csv file that contains all the data for training and testing the machine learning model. It should look something like this (I've called this file `train_test_data.csv`)

|id                 |CAP               |astroturf|fake_follower|financial|other|overall|self-declared|labels|
|-------------------|------------------|---------|-------------|---------|-----|-------|-------------|------|
|3039154799         |0.7828265255249504|0.1      |1.8          |1.4      |3.2  |1.4    |0.4          |1     |
|390617262          |1.0               |0.8      |1.4          |1.0      |5.0  |5.0    |0.2          |0     |
|4611389296         |0.7334998320027682|0.2      |0.6          |0.1      |1.8  |1.1    |0.0          |1     |
|734396807745286145 |0.2603678379426137|0.0      |0.0          |0.0      |0.2  |0.1    |0.0          |1     |
|1010978324569640960|0.9077975241648624|1.2      |2.7          |2.8      |4.8  |4.8    |0.3          |0     |

You can read from this csv file using this line of code: `master_df = pd.read_csv("train_test_data.csv")`
### Converting Organizations to Non-Individuals
In getting data for the model creation, there was a small hang up: we didn't have enough information on organizations to reasonably train a model on. To train and test a model, we need a lot of data for each category that we want to classify. To rectify this, we'll just lump the organizations in with the bots to make one general category called "non-individuals". 

Practically speaking, all we have to do is to make sure that every instance of a '2' in the `labels` column is changed to a '0'. This can be accomplished with a few lines of code:
```
def turn_orgs_to_bots(df):  
  df[df['labels'] == 2] = 0  
  return df

# call the function with this line
df = turn_orgs_to_bots(master_df)
```
### Creating the Feature and Labels Datasets
A feature is a characteristic of an entity while the label is the classification of that entity. In this case, the features of the twitter accounts are the CAP, astroturf, fake_follower, financial, other, overall, and self_declared scores. The labels are the 0s and 1s that indicate whether an account is an individual or non-individual. 

In short: features are **input** and labels are **output**.

At this point, your program should look something like this:
```
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import xgboost
from sklearn import preprocessing
import pickle
from sklearn.metrics import plot_confusion_matrix, accuracy_score
import pandas as pd 

def turn_orgs_to_bots(df):  
  df[df['labels'] == 2] = 0  
  return df
  
def main():
	
	master_df = pd.read_csv("train_test_data.csv")
	
	df = turn_orgs_to_bots(master_df)

main()
```

To be able to train and test the machine learning model, we'll need to separate the labels from the features. We can do this pretty easily with pandas in a couple lines of code
```
# features 
x1 = bots_humans.drop(['labels', 'id'], axis=1).values

# labels
y1 = bots_humans['labels'].values
```
In the features matrix, we drop both the `labels` and the `id` columns. We drop the `labels` columns because it would be essentially giving the model the "answers" and we drop the `id` column because we don't expect it to be correlated with the output. 

### Preparing the Train and Test Data
Next, we'll need to split our dataset into 4 more datasets: `X_train`,`X_test`,`Y_train`, `Y_test`

We can do this with one line of code:
`X_train, X_test, Y_train, Y_test = model_selection.train_test_split(x1, y1, test_size=0.30, random_state=100)`

* The `test_size` parameter is the proportion of the dataset that will be included in the test datasets.
* The `random_state` parameter controls the shuffling of the dataset before it is split. This is useful for reproducibility

[You can read more about the function here.](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)

After spliting the dataset, we'll normalize the features so that the machine learning algorithm will have an easier time training on the data. To do this, we can use the `preprocessing` class from `sklearn`

```
X_train_scaled = preprocessing.scale(X_train)
X_test_scaled = preprocessing.scale(X_test)
```
## Training and Testing
### Setting up out model
Before we can train our model, we first have to define it. In this first iteration, we'll focus on implementing the `LogisticRegession` classifier. 

We can set it up with one line of code
`model = LogisticRegression()`
There are many parameters that can be passed into model, but the default settings should work for us here.

[LogisticRegression docs](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)

### Training
Training our classifier is also relatively simple. We can do that in one line of code too:
`model.fit(X_train_scaled, Y_train)`

To check the accuracy of our model, we can use the built-in score function
`result = model.score(X_test_scaled, Y_test)`

Print the accuracy of the model
`print("Accuracy: %.2f%%" % (result * 100.0))`

## Resulting Code
At this point, your code should look something like this
```
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import xgboost
from sklearn import preprocessing
import pickle
from sklearn.metrics import plot_confusion_matrix, accuracy_score
import pandas as pd 

def turn_orgs_to_bots(df):  
  df[df['labels'] == 2] = 0  
  return df
  
def logisticRegression():
	
	master_df = pd.read_csv("train_test_data.csv")
	
	df = turn_orgs_to_bots(master_df)
	# features 
	x1 = bots_humans.drop(['labels', 'id'], axis=1).values

	# labels
	y1 = bots_humans['labels'].values
	X_train, X_test, Y_train, Y_test = model_selection.train_test_split(x1, y1, test_size=0.30, random_state=100)

	X_train_scaled = preprocessing.scale(X_train)
	X_test_scaled = preprocessing.scale(X_test)
	model = LogisticRegression()
	model.fit(X_train_scaled, Y_train)
	result = model.score(X_test_scaled, Y_test)
	print("Accuracy: %.2f%%" % (result * 100.0))

logisticRegression()
```
If everything when smoothly, your output might look something like this:
```
C:\Users\_____\AppData\Roaming\Python\Python38\site-packages\sklearn\preprocessing\_data.py:174: UserWarning: Numerical issues were encountered when centering the data and might not be solved. Dataset may contain too large values. You may need to prescale your features.
  warnings.warn("Numerical issues were encountered "
C:\Users\_________\AppData\Roaming\Python\Python38\site-packages\sklearn\preprocessing\_data.py:174: UserWarning: Numerical issues were encountered when centering the data and might not be solved. Dataset may contain too large values. You may need to prescale your features.
  warnings.warn("Numerical issues were encountered "
Accuracy: 83.65%
```
You can ignore the warnings for the time being, but take a look at the Accuracy Rating. In this case, it is 83.65%.

## Improving Your Model
Lets cut to the chase, 83.65% accuracy is a bit abymsal for a machine learning model (this depends on the context, but I'm sure we can do a lot better). It is better than chance, but if our goal is the accurately categorize thousands or even millions of accounts, we can't have 1 out of every 5 accounts being incorrectly classified. Let's jump into some of the steps we can take to improve the accuracy of our model

### Outing the Outliers
One of the quickest and easiest ways to improve the accuracy of your model is to change the data that it is trained on. Even though it sounds like cheating (Its not, I'm 83.65% sure), removing outliers is a pretty safe way to manage your dataset and decrease noise in a distribution. 

The problem with outlier is that they aren't representative of the distribution. They represent a bit of noise that clouds the overall picture of the distribution. Machine learning models are supposed spot general patterns in the over data that should carry over to data that it hasn't seen. Outliers harm this process of generalization because they represent noise that tweaks the assumptions that the machine learning model makes so that the assumptions aren't as generalizable anymore. Logistic Regression is especially sensitive to this, but this is a general rule for all machine learning models. 

You can read more about what is considered an outlier [here](https://www.itl.nist.gov/div898/handbook/prc/section1/prc16.htm#:~:text=An%20outlier%20is%20an%20observation,random%20sample%20from%20a%20population.&text=Examination%20of%20the%20data%20for,often%20referred%20to%20as%20outliers.) but for the pruposes of this tutorial, I'll demonstrate a couple of simple ways you can remove outlier from your dataset.

The way we'll remove outliers in this dataset is using something called the *z-score*. What is the z-score: ([from here](https://www.statisticshowto.com/probability-and-statistics/z-score/))

> **Simply put, a z-score (also called a  _standard score_) gives you an idea of how far from the 
> [mean](https://www.statisticshowto.com/probability-and-statistics/statistics-definitions/mean-median-mode/) a data point is.** But more technically it’s a measure of how many[standard deviations](https://www.statisticshowto.com/probability-and-statistics/standard-deviation/) below or above the [population mean](https://www.statisticshowto.com/population-mean/) a [raw score](https://www.statisticshowto.com/raw-score/) is.

That's pretty much all you need to know in order to get rid of outlier from our training dataset. We'll just calculate the z-score for each datapoint and if it is higher than a certain threshold, we'll remove it from the dataset. 

Note: this is not the only way to classify outliers, we can use the IQR (Inter-Quartile Range) as well. 

**How to Remove Outliers with Pandas and Numpy**
Luckily for us, removing outliers is really really easy with the pandas and numpy libraries.

First, lets grab the training data:
```
mdf = pd.read_csv("raw_data_set.csv")  
```

Then separate them into their respective categories
```
# separate them by category  
human_df = mdf[mdf["labels"] == 1]  
non_human_df = mdf[mdf["labels"] == 0]  
```
Here's the code to calculate the z score for each value with respect to the mean and standard deviation of the column that they are in. There might be a faster and simply way to do it, but it can be done all in one line with this piece of code.
```
human_df = human_df[(np.abs(stats.zscore(human_df[columns])) < 2.5).all(axis=1)] 
```
The 2.5 value is our *z-score threshold*. This means that if the value is more than 2.5 standard deviations from the mean (in either direction) then we discard it. You can change this value as much as you want, and tweak it depending on the accuracy of the model.

You should do this for both the human and non-human  categories and then concantenate the two dataframes like so:
`master = pd.concat([human_df, non_human_df])`

Here's what the full code looks like:
```
def remove_outliers(in_path=None):  
  mdf = pd.read_csv("data_bank/cleaning_data/master_training_data_id/master_train_one_hot_no_dup.csv")  
  
    # separate them by category  
  human_df = mdf[mdf["labels"] == 1]  
    non_human_df = mdf[mdf["labels"] == 0]  
  
    # filter out the outliers  
 # CAP,astroturf,fake_follower,financial,other,overall,self-declared  
 # for humans  columns = ["astroturf", "fake_follower", "financial", "other", "overall", "self-declared"]  
    human_df = human_df[(np.abs(stats.zscore(human_df[columns])) < 2.5).all(axis=1)]  
  
    # for non_humans  
  non_human_df = non_human_df[(np.abs(stats.zscore(non_human_df[columns])) < 2.5).all(axis=1)]  
  
    #print(human_df.describe())  
  print(non_human_df.describe())  
  
    # combine both of the dataframe and export  
  master = pd.concat([human_df, non_human_df])  
  
    #print(master.describe())  
  master.to_csv("data_bank/cleaning_data/master_training_data_id/master_train_one_hot_no_outliers_z_25.csv", index=False)  
  
    pass
```
Once you've removed the outliers, run the model training program again. Your accuracy should be substantially better.
### Validating Logistic Regression with Different Validation Schemes
Validation is an important part of the process of creating a machine learning model. There are many different types of validation beyond your typical 70-30, train/test split:
- k-Fold Cross-Validation
-   Leave-one-out Cross-Validation
-   Leave-one-group-out Cross-Validation
-   Nested Cross-Validation
-   Time-series Cross-Validation
-   Wilcoxon signed-rank test
-   McNemar’s test
-   5x2CV paired t-test
-   5x2CV combined F test

For this tutorial, it's not important to go through all of these different types of validation (if you would like to read more about them, [start here](https://towardsdatascience.com/validating-your-machine-learning-model-25b4c8643fb7)), but it is helpful to know for gaining more insights about the dataset and the model's performance.

### Creating a Confusion Matrix
Creating a confusion matrix is one of the most useful diagnostic tools you can use. A confusion matrix is a 2x2 matrix (in the case of binary output - more general case is NxN) that logs 4 different types of outcomes:
* **True Positive**: The predicted value matches the actual value. The actual value was positive and the model predicted a positive value
* **True Negative**: The predicted value matches the actual value. The actual value was negative and the model predicted a negative value
* **False Positive (Type 1 Error)**: The predicted value was falsely predicted. The actual value was negative but the model predicted a positive value.
* **False Negative (Type 2 Error)**: The predicted value was falsely predicted. The actual value was positive but the model predicted a negative value

Below is a visual representation of the matrix:
![Visual Representation of a Confusion Matrix](https://res.cloudinary.com/cheezitromansh/image/upload/v1621180639/Confusionmatrix-example_nizscq.webp)
([Source for definitions and image](https://www.analyticsvidhya.com/blog/2020/04/confusion-matrix-machine-learning/))
The negative/positive value here can be replaced with human/non-human, but the conceptually, the model stays the same. 

Adding a confusion matrix to our pipeline is pretty easy. All that needs to be done is to add a couple of lines of code to the end of the function that trains the logisticRegression model.

First, import the `plot_confusion_matrix` class with this line of code:
`from sklearn.metrics import plot_confusion_matrix`

Then in your function, set title options like so:
`title_options = [("Confusion Matrix, without normalization", None),  
                 ("Normalized Confusion Matrix", "true")]`

Lastly, for each aspect of the title options, display a confusion matrix with the model, X_test, Y_test and your display_label parameters. You can set the color and choose the 'normalize' option (highly recommended).
```
for title, normalize in title_options:  
  disp = plot_confusion_matrix(model, X_test_scaled, Y_test, display_labels=["bot", "human"],  
                                 cmap=plt.cm.Blues, normalize=normalize)  
    disp.ax_.set_title(title)  
    print(title)  
    print(disp.confusion_matrix)
```
Use `plt.show()` to display it to the screen. 
The result should look something like this:
![Example Confusion Matrix Without Normalization](https://res.cloudinary.com/cheezitromansh/image/upload/v1621181279/confusion_matrix_bvs0vr.png)

## Creating a Random Forest Model with XGBoost
We've spent quite a bit of time creating a model with Logistic Regression and while LogReg is a decent starter algorithm for binary classification, there is a more powerful option that we have available: The Random Forest Model. (There is an explanation above if you're curious about how it works, but we'll jump straight into the code here)

`sklearn` has a Random Forest Model built into it and while it is usable, it is considerably more unweildy and not as powerful as the XGBoost. XGBoost is meant to be a boosted version of the Random Forest Model allowing for GridSearch (for the best hyperparamters) and effortless customization. It's also super portable and easy to transfer from one system to another which'll make our lives easier. You can read more about the model and the project [here](https://xgboost.readthedocs.io/en/latest/). 

Don't forget to `import xgboost`
### Setting up the data
Setting up the data for the XGBoost model looks exactly the same as setting up the data for LogisticRegression.
```
# this one prunes outlier and then throws them back together  
master_df = pd.read_csv(master_path)  
  
bots_humans = turn_orgs_to_bots(master_df)  
#bots_humans = get_rid_of_orgs(master_df)  
  
  
  
# split into array for the features and resp vars  
removed = ['labels', 'financial', 'self-declared', 'fake_follower', 'CAP']  
x1 = bots_humans.drop(['labels', 'id'], axis=1).values  
# print(bots_humans.drop(['labels', 'id'], axis=1).head(5))  
# return  
y1 = bots_humans['labels'].values  
  
# train using Logistic Regression  
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(x1, y1, test_size=0.30, random_state=100,stratify=y1)  
  
# preprocess the data  
X_scaled = preprocessing.scale(X_train)  
```

### Choosing the Optimal XGBoost Model with Grid Search
After we've set up the data, we need to instance the model:
`model = xgboost.XGBClassifier()`

Then we need to create an `optimization_dict`. This dictionary will contain the set of hyperparameters that our program will set in the models that it will test and compare against each other to produce the best set of hyperparameters. The main 3 hyperparameters in an XGBoost model are:
* **max_depth**: The depth of each decision tree created 
* **n_estimators**: The amount of decision trees in each model
* **learning_rate**: Affects the cost incurred while training and validating a model.

Here's an example of an optimization dict:
```
optimization_dict = {  
    'max_depth': [2, 4, 6],  
    'n_estimators': [50, 200, 500],  
    'learning_rate': [0.1, 0.01, 1],  
}
```
To implement the Grid Search (this searches all the possible combinations of hyperparameters for the best combination) you can use this line of code:
```
model = model_selection.GridSearchCV(model, optimization_dict,  
                      scoring='accuracy', verbose=1)
```
Finally, fit the model using `model.fit(X_scaled, Y_train)` and print the `model.best_score_` and `model.best_params_` in order to get the optimal hyperparameters.

**Note**: You can also use the line `model.feature_importances_` to get a list of features and a probability that shows their influence on the decision making of the model. This is useful for feature pruning or getting rid of less than helpful features and speeding up computation.

Once you've done that, you should have a completely functioning model!
## Saving Your Model
Now you've built your model! Congulations and pat yourself on the back. But now you need to share your model with the world (or anyone else that you're working with). You can do this in numberous ways but the simplest is simply creating a pickle/data file that contains the information on the weights and biases of the model.
### Saving the model using pickle
Make sure you have the `pickle` model imported:
`import pickle`

Then save your model all in one line:
`pickle.dump(model, open('file_path_here.dat', 'wb'))`

This will save your model at the location that you specified as a `.dat` file. Technically, you can use any file extension you want, but most of the tutorials online on the subject use the `.dat` file extension.
### Loading the Model using Pickle
Once you've save it, you've got to open it again so you can run it. This process is just as simple but this time you need to make sure you have both `pickle` and `xgboost` imported. 

Then using a couple of lines of code, you can read the file from the previous step and load it in as a model.
```
try:  
  model = pickle.load(open(path, 'rb'))  
    return model  
except Exception as e:  
  print(f'ERROR: {repr(e)}')
```
### Running a saved model
Running a saved model is just as easy and requires one line of code:
`result = model.predict(data)`
The data variable should be valid input of whatever dimensions the model was trained on.

# Afterwords
And there you go! We've successfully created a model from scratch using data we've gathered ourselves. Pat yourself on the back! 