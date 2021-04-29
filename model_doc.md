# Model Documentation
_This model is to be used to classify bots as either an individual or a non-individual. Non individuals include bots and corporations/organizations_

## Introduction
This model is aimed at classifying twitter accounts as individuals or non-individuals. Non-individuals include both bots and organizations. 

### How it works

The model uses the [botometer library](https://github.com/IUNetSci/botometer-python) to gather key statistics about a twitter account and passes the information into a model that will classify whether the account represents and individual or not. 

This model uses the [XGBoost Library](https://xgboost.readthedocs.io/en/latest/) to provided a boosted Random Forest Model to classifiy the accounts.

To authenticate yourself with the Botometer endpoint, you will need both TwitterAPI and RapidAPI credentials. Twitter API credentials comes from activation of an existing Twitter account. Learn more about this process [here](https://developer.twitter.com/en/docs/authentication/guides/authentication-best-practices).
You will also need a RapidAPI Subscription. Learn more about getting credentials [here](https://docs.rapidapi.com/docs/keys).

To learn more about how botometer works and try it out for yourself, take a look at [their website](https://botometer.osome.iu.edu/). Check out [datasets](https://botometer.osome.iu.edu/bot-repository/datasets.html) and the [papers](https://botometer.osome.iu.edu/publications) that inspired and codified their research.

