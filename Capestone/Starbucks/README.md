# Starbucks Capstone Challenge
Project in Data Scientist Nanodegree of Udacity
## Python libarieis needed
- pandas
- numpy
- json
- datatime
- matplotlib
- seaborn
- pickle
- Sci- learn
## Project Motivation

It is the Starbuck's Capstone Challenge of the Data Scientist Nanodegree in Udacity. We get the dataset from the program that creates the data simulates how people make purchasing decisions and how those decisions are influenced by promotional offers. We want to make a recommendation engine that recommends Starbucks which offer should be sent to a particular customer.

We are interested to answer the following question:
1- For each age group what is the most common offer type ?

2-For each gender what is the most common offer type ?

3-For each gender what is the most common income status?

4-What is the event status for each offer type?

5-For each offer id what is the most common event occurring?

6- Are we able to predict if the user will finish the offer he just viewed or not?
The data is contained in three files:
- `portfolio.json` - containing offer ids and meta data about each offer (duration, type, etc.)
- `profile.json` - demographic data for each customer
- `transcript.json` - records for transactions, offers received, offers viewed, and offers completed (Could not be added here because of large file size)

Here is the schema and explanation of each variable in the files:

`portfolio.json`
- id (string) - offer id
- offer_type (string) - the type of offer ie BOGO, discount, informational
- difficulty (int) - the minimum required to spend to complete an offer
- reward (int) - the reward is given for completing an offer
- duration (int) - time for the offer to be open, in days
- channels (list of strings)

`profile.json`
- age (int) - age of the customer
- became_member_on (int) - the date when customer created an app account
- gender (str) - gender of the customer (note some entries contain 'O' for other rather than M or F)
- id (str) - customer id
- income (float) - customer's income

`transcript.json`
- event (str) - record description (ie transaction, offer received, offer viewed, etc.)
- person (str) - customer id
- time (int) - time in hours since the start of the test. The data begins at time t=0
- value - (dict of strings) - eith
## Conclusion
We tried in this project to analyze the dataset and build a model that can predict whether a customer would complete the offer or just check it
I had different numerical variables that i had to turn into categorical by splitting the continuous data into a ranges with each range has specific categorical name.
I did this with the age, salary, events,gender, offer type and the day since being a member.
I cleaned the data to remove any null value or duplicates from the dataset.
I used 3 different machine learning model to compare and choose which model is the best to accurately predict if the customer will complete the offer or not.
## Results

The main findings of the code can be found at the post available https://mostafahaggag.medium.com/the-effect-of-starbucks-offers-on-customer-purchase-2e3008a5f820

the git hub link is https://github.com/Mostafa-Haggag/Udacity_data_science_nano_degree_projects/tree/main/Capestone/Starbucks