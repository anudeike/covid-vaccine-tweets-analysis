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

 - Pandas  ```pip install pandas```: This module is used to be read data from csv/tsv files. It is also useful for data exploration/wrangling
 - Matplotlib ```pip install matplotlib```: This module is used for displaying information about the data in the form of graphs, charts and other graphics.
 - Numpy ```pip install numpy```: Used for faster than built-in math calculations.
 - Sklearn ```pip install sklearn```: This is where we will create our models/test them. There are a lot of useful metrics here
 - XgBoost ```pip install xgboost```: This library contains faster and more robust models complete with a lot of validation and metrics.