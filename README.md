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
    
    Command:
    
    `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
        
    Sample Output:     
    
    ```
       
        Loading data...
            MESSAGES: data/disaster_messages.csv
            CATEGORIES: data/disaster_categories.csv
        Messages and Categories DataFrames are read and merged.
        Cleaning data...
        Record Count : 26386
        170 duplicates found to be removed
        Final Record Count : 26216
        Saving data...
            DATABASE: data/DisasterResponse.db
        messages Table is saved to : data/DisasterResponse.db
        Cleaned data saved to database!
     ```
        
    - To run ML pipeline that trains classifier and saves
     
     Command:
     
     `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
        
     Sample Output:
     
     ```
       
        Loading data...
        DATABASE: data/DisasterResponse.db
        Building model...
        Training model...
        Evaluating model...
                          category  f1_score  precision    recall
        0                  related  0.812166   0.812166  0.812166
        1                  request  0.901983   0.901983  0.901983
        2                    offer  0.995042   0.995042  0.995042
        3              aid_related  0.783181   0.783181  0.783181
        4             medical_help  0.922197   0.922197  0.922197
        5         medical_products  0.950229   0.950229  0.950229
        6        search_and_rescue  0.974828   0.974828  0.974828
        7                 security  0.980549   0.980549  0.980549
        8                 military  0.965484   0.965484  0.965484
        ...
        35           direct_report  0.860793   0.860793  0.860793
        Saving model...
            MODEL: models/classifier.pkl
        Trained model saved!
      ```


2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/



### Web App :

![Alt text](https://github.com/elifinspace/udacity_dataengineering/blob/master/data/genres_hist.png?raw=true)

![Alt text](https://github.com/elifinspace/udacity_dataengineering/blob/master/data/categories_hist.png?raw=true)


### Author:
Elif Surmeli
