# udacity_dataengineering
Udacity Data Scientist Nanodegree Data Engineering Portfolio

# Disaster Response Pipeline Project

### Software Requirements:

- Anaconda 3, Python 3.7.3
- nltk : https://www.nltk.org/install.html
- sqlalchemy : https://pypi.org/project/SQLAlchemy/
- plotly : https://plot.ly/python/getting-started/
- flask : https://pypi.org/project/Flask/


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/



### Web App :

![Alt text](https://github.com/elifinspace/udacity_dataengineering/blob/master/data/newplot.png?raw=true)


### Author:
Elif Surmeli
