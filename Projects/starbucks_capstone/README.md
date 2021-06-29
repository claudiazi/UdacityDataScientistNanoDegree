# StarbucksCapstoneProject
# Introduction
As part of the final project, the simplified Starbucks data is given. 
The dataset contains simulated data that mimics customers' behavior after they received Starbucks offers.
The data is collected via Starbucks rewards mobile apps and the offers were sent out once every few days to the users of the mobile app.
The medium article can be seen: https://claudiazi1244.medium.com/starbucks-next-best-ffer-efee9da254a9.

# Goals of the analysis
Through this project, I would like to know 
1. What kind of customers will tend more to complete an offer?
2. What offers can be completed more?
3. How well canandomForest, XGBoost, and DecisionTree algorithm can learn from the data and predict whether a customer with specific demographics would respond to a specific offer?

## File Description
~~~~~~~
        starbucks_capstone
          |-- data
                |-- portfolio.json
                |-- profile.json
                |-- transcript.json
          |-- Starbucks_Capstone_notebook.ipynb
          |-- README
~~~~~~~


# Data Sets
The data is contained in three files:

* portfolio.json - containing offer ids and meta data about each offer (duration, type, etc.)
* profile.json - demographic data for each customer
* transcript.json - records for transactions, offers received, offers viewed, and offers completed

Here is the schema and explanation of each variable in the files:

**portfolio.json**
* id (string) - offer id
* offer_type (string) - type of offer ie BOGO, discount, informational
* difficulty (int) - minimum required spend to complete an offer
* reward (int) - reward given for completing an offer
* duration (int) - time for offer to be open, in days
* channels (list of strings)

**profile.json**
* age (int) - age of the customer 
* became_member_on (int) - date when customer created an app account
* gender (str) - gender of the customer (note some entries contain 'O' for other rather than M or F)
* id (str) - customer id
* income (float) - customer's income

**transcript.json**
* event (str) - record description (ie transaction, offer received, offer viewed, etc.)
* person (str) - customer id
* time (int) - time in hours since start of test. The data begins at time t=0
* value - (dict of strings) - either an offer id or transaction amount depending on the record

#Required Libraries
*  `pandas`
* `numpy`
* `seaborn`
* `sklearn`
* `xgboost`
* `json`

