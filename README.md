Rainfall Prediction — Machine Learning Project
Predicts whether it will rain today based on atmospheric conditions like humidity, pressure, cloud cover, and sunshine hours — deployed as a live web app anyone can use

Live Demo: Click here  https://rainfallpredictor-by-sourabh.streamlit.app/

Problem Statement :-
Given daily weather readings, can we predict rainfall accurately enough to be useful? This project answers that by comparing three classification models and deploying the best one as an interactive web application.

Workflow :-
Data Cleaning -> EDA -> Feature Selection -> Model Training > Evaluation > Deployment

Step	What I Did :-
Data Cleaning	Stripped extra spaces from column names, dropped 1 null row in winddirection and windspeed
EDA	Histograms, correlation heatmap, boxplots — found heavy multicollinearity in temperature features
Feature Selection	Dropped maxtemp, temperature, mintemp — kept dewpoint as the most rainfall-relevant feature. Also dropped day (not a time series problem)
Encoding	Converted rainfall target from yes/no to 1/0
Scaling	Applied StandardScaler before training
Modelling	Trained KNN, Naive Bayes, Decision Tree — compared all three
Deployment	Built Streamlit web app with confidence score and weather advisory
Model Comparison
Model	Test Accuracy	F1 Score	Overfit?
KNN	72.6%	0.80	Mild
Naive Bayes	73.9%	0.83	No
Decision Tree	73.9%	0.83	Yes (Train = 100%)
Chosen model: Gaussian Naive Bayes — Decision Tree scored the same accuracy but memorised the training data perfectly (100%), which means it would fail on real unseen data. Naive Bayes was more honest and generalised better.

Features Used
pressure · dewpoint · humidity · cloud · sunshine · winddirection · windspeed

Tech Stack
Python , Pandas ,

NumPy , Scikit-learn ,

Matplotlib  , Seaborn ,

Streamlit , Joblib

Key Learning
The Decision Tree had 100% training accuracy — which sounds great but is actually a red flag. It had memorised every row instead of learning patterns. This project taught me that a lower, consistent score across train and test is always better than a perfect training score that collapses on new data.

Contact 
LinkedIn - http://www.linkedin.com/in/sourabh9098
GitHub - https://github.com/sourabh9098
Email - www.sourabh555@gmail.com

Three models trained One chosen One reason — it generalised.
