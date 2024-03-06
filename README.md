# Lung Cancer Prediction using ML

> This repository contains the code for EDA, training and inferencing using trained models for predicting whether a person is prone to have lung cancer.

## Personal Details

> Name: Yatharth Mahesh Sant
> Email: ysant77@gmail.com


## Project Structure

* `EDA.ipynb` - IPython notebook for performing exploratory data analysis (EDA).
* `train.py` - Python file for performing model training and hyper-parameter tuning. Stores various evaluation metrics
* `inference.py` - Python file for performing model inferencing using the pre-trained models.
* `config.json` - Stores the various parameters required for hyper-parameter tuning for the different models.
* `run.sh` - Script to run the inference.py file.
  
```Lung_Cancer
├── README.md
├── EDA.ipynb
├── train.py
├── inference.py
├──config.json
└── run.sh
```
## Instructions to modify and fine-tune the models hyper-parameters

> The config.json file can be edited to modify the hyper-parameters for the different models. For every new model to be added, same structure should be followed as that of depicted in the config.json
> And add a new mapping of model name to the model_factory.py class in the src/train directory.

## Steps followed for creating the MLP

1. A config.json file was first defined for different models that captures the hyper-parameters for each one of them. Two additional params added to indicate if OHE (one hot encoding), and data standardization is needed for the model or not.
2. Separate methods defined for performing OHE, data standardization, and the creation of train-test split of data.
3. Mapping of model name to the model object is defined in models dictionary.
4. A 10-fold cross validation is performed using GridSearch, once completed the binary file is saved for best estimator found.
5. Metrics like accuracy, classification report, confusion matrix are computed and stored.
6. Metrics of all the models indexed by their name are stored in the model_evaluation_results.json file.

## Models selected for processing

> Six models viz RandomForest, KNN, LogisticRegression, NaiveBayes and DecisionTree are selected due to the well suited application of these models for the binary classification task. <br/>
> Hyper-parameter tuning and k-fold cross validation is performed to avoid overfitting and underfitting.  <br/>

## Evaluation Metrics computed

> The standard evaluation metrics for multi-class classification like test accuracy, confusion matrix, and classification report were computed.

## Steps to execute the inference script

> A CSV file is expected by the inference.py file. It should have all columns as that of the dataset.csv in this repository.  <br/>
> A sample CSV file is provided in this repository.  <br/> 
> The file used for training which is merged_df.csv is also provided in the repository.  <br/>
> The user is expected to input one integer from 0 to 6, 0 to run all models.  <br/>




