# Predicting an Individual’s Risk of Heart Disease
Authors: Phuong Vu, Yunwei Liang, Trinh Nguyen

## Summary of Research Questions

Which machine learning algorithm can best predict the risk of heart disease?
We will compare the accuracy of different supervised machine learning algorithms for predicting the risk of heart disease and find the algorithm with the highest accuracy. We will be using the Decision Tree Classifier, the Random Forest Classifier, and the Gaussian Naive Bayes Classifier. We will be including all patients’ characteristics in the features of the machine learning model. Our models will predict the risk of a patient having heart disease. We will compare our models with the accuracy score of each model.

## **Which certain sets of features are better indicators of the high risk of heart disease than the other sets of features?**

We will implement the most optimal algorithm found in Part 1 of our research for Part 2. In Part 2, we are trying to find the impact specific set of features have on predicting the risk of heart disease. We will compute the correlation of different sets of features and risk of heart disease. We will examine each trial’s accuracy, and find the set of features with the highest accuracy. 

### Motivation and Background:

Through the research article “The accuracy of prediction of heart disease risk based on Machine Learning Classification Techniques”, we are inspired to find the correlation between patients’ characteristics and their risk of heart disease. 

The researchers in research article tested the accuracy of different Machine Learning algorithms when predicting the risk of heart disease of a participant. We will use their dataset of participants’ information to make a simplified version of their investigation. 

We will test and compare the accuracy of different learning machine models based on their accuracy score. We will then use the most accurate model to determine how specific set of features will affect the model’s prediction of health risk. 

For future exploration/expansion, we could develop machine learning models most fitting for common medical diagnosis to reduce the rate at which people are being turned away from hospitals during a crisis. This advance could also potentially lower medical costs, which is one of the biggest factors in one’s decision to even seek out a doctor.  

### Dataset:
[Prior Research Article on Machine Learning to Predict Heart Disease](https://www.sciencedirect.com/science/article/pii/S235291481830217X#bib22)

[Full Heart Disease Data Sets from UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Heart+Disease)

[Cleaned Cleveland Data Set: processed.cleveland.data -> cleveland_processed.csv](https://docs.google.com/spreadsheets/d/1RYEbU88sNvnPz-wQ7vI03xP251S6zems8A9-hpajaXo/edit?usp=sharing)

We will use the dataset in “processed.cleveland.data" file in the Data Folder of the Full Heart Disease Data Sets. This dataset was collected from Cleveland Clinic Foundation and accessed in the UCI Machine Learning Repository website. We converted the comma-separated-file into excel and added column names according to their metadata.

There are 14 attributes describing each of 303 studies in the dataset. The data shows the relationship among people’s different traits including sex, chest pain type, resting blood pressure, etc., and risks of heart disease.

For our machine learning models, we will use the column “class att” as the label. Note that “class att” indicates a healthy individual with “H” and at-risk individual with “S1”, “S2”, “S3”, and  “S4”.

## Methodology:
1. Determine the best algorithm

    1. Transform data from a mod file into an accessible format. Filter missing data out of the set. One-hot encoding the dataset if necessary.
    2. Select all columns as features except “class att”--this is the health risk--to be the model’s label. Include all attributes as the features.
    3. Set up and train a Decision Tree Classifier Model 
        - Follow the steps from lecture:
            - Unpack train_test_split into training and testing sets with a 8:2 ratio relatively.
            - Use DecisionTreeClassifier to build the model.
            - Fit the model with the training set.
            - Test the model with the test set and record the accuracy score.
    4. Set up and train a Random Forest Classifier Model
        - Import numpy and convert the label from the dataframe to a Numpy array. Remove the identifying names for the features from the dataframe, then turn the features into a Numpy array.
        - Splitting the data into 80% training set and 20% testing set using train_test_split() from sklearn’s model selection.
        - Train and fit the model with training data using model.fit()
        - Record the accuracy score. 
        - Follow this [link](https://towardsdatascience.com/random-forest-in-python-24d0893d51c0) for a more informative method 
    5. Set up and train a Gaussian Naive Bayes Model
        - Getting the dummy encoding for the features using get_dummies() from pandas. 
        - Splitting the data into 80% training set and 20% testing set using train_test_split() from Skicit-learn’s model selection. Establish a baseline for the model so that it can optimize its classification approach.
        - Create a Gaussian Naive Bayes Model using GaussianNB() from the Skicit-learn’s naive_bayes. 
        - Train and fit the model with training data using model.fit()
        - Test and record the accuracy with accuracy_score() using the testing set.
        - Follow this [link](https://www.datacamp.com/community/tutorials/naive-bayes-scikit-learn) for a more informative method
2. Using the best algorithm to determine the best set of features
    1. Use the prepared data from Part 1 and the most accurate machine learning model(will be determined by ).
    2. Using matplotlib, create a bar plot of the feature importances. This will help us visualize the impact of each feature on the model’s prediction. Based on the results, come up with different combinations of features from the dataset. 
    3. Use the relative instruction from Part 1 for the most accurate method to test the different sets of features.
    4. Calculate and record each accuracy score.
    5. Plot a scatter chart based on the model’s prediction to compare the predicted values to the actual values.

## Work Plan:
- Set up the starter files using Jupyter Notebook and share access
    - Starter files include main.py, cleveland.mod (finish by February 28th )
    - Break down the main.py into two main parts to address two main research questions
- Responsibilities: All contributors in meeting: first machine learning model in part 1 finish by March 2nd
    - Finish transforming data into  accessible format and filtering the data
    - Split out the feature and label columns
    - Set up and train a Decision Tree Classifier Model
    - Record the accuracy score
- Trinh: Part 1
    - Finish setting up the Random Forest Classifier model by March 3rd
    - Finish splitting the data and training the model by March 5th
    - Finish recording the accuracy score of the model by March 6th
- Yunwei: Part 1
    - Finish setting up the Gaussian Naive Bayes Classifier model by March 4th
    - Finish splitting the data and training the model by March 6th
    - Finish recording the accuracy score of the model by March 7th
- Phuong: Part 2
    - Finish making bar plot of the feature importances and prediction by March 7th
    - Finish testing different sets of features using the result of part 1 by March  9th
    - Finish calculate, record accuracy scores and make chart to compare the predicted and actual values by March 10th
- Time estimates
    - Finish writing all the code for transforming data, plotting graphs and building machine learning model by March 10th
    - Finish writing report of the result by March 13th
    - Finish preparing materials and practicing for in-class presentation by March 19th

### Questions(optional):
How can we convert the .mod or .data file into .csv one?
