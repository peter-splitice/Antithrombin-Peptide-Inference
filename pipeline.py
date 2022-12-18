## Package Imports
# Unpack Files
import json
import pickle

# Data Analysis
import pandas as pd
import numpy as np 

## Pipeline for multi-stage model:
def model_pipeline(allFeaturesData, ensemble=bool):
    """
    Function for the multistage pipeline.  The following stages are applied:
        - Classification of the peptides.  Positive classification means efficacy in antithrombin response.
        - Classification of the positive peptides into buckets.  0 = small, 1 = medium, 2 = large.
        - Regression on peptides that are set to buckets 0 and 1, to predict "KI (nM)" values for the peptides.

    Parameters
    ----------
    allFeaturesData: Pandas DataFrame containing the peptides with extracted features.

    ensemble: Boolean Flag to determine use of an ennsemble method for the initial peptide classification.

    Returns:
    --------
    result: Pandas DataFrame containing the results with columns ['Name','Seq','Predicted','KI (nM) Predicted']
    """
    ### MODEL IMPORTS
    # ------------------------------------
    ## Classification Model Imports:
    # Scaler
    with open('Classification Dependencies/Scaler Classification.pkl', 'rb') as fh:
        scaler_for_classification = pickle.load(fh)

    # First Model
    with open('Classification Dependencies/SVM Linear Classification trained.pkl', 'rb') as fh:
        classification_model_1 = pickle.load(fh)
    
    # Second Model
    with open('Classification Dependencies/SVM RBF Classification trained.pkl', 'rb') as fh:
        classification_model_2 = pickle.load(fh)

    # Selected features Classification
    with open('Classification Dependencies/Features for Classification Model.json') as fh:
        classification_features = json.loads(fh.read())

    ## Regression Model Imports:
    # Scaler
    with open('Regression Dependencies/Scaler Regression.pkl', 'rb') as fh:
        scaler_for_regression = pickle.load(fh)

    # Classification with ki
    with open('Regression Dependencies/SVC RBF bucket Classification trained.pkl', 'rb') as fh:
        classification_model_for_buckets = pickle.load(fh)

    # Regression for Medium Bucket
    with open('Regression Dependencies/SVR RBF medium bucket Regression trained.pkl', 'rb') as fh:
        regression_model_medium_bucket = pickle.load(fh)

    # Regression for Small Bucket
    with open('Regression Dependencies/SVR RBF small bucket Regression trained.pkl', 'rb') as fh:
        regression_model_small_bucket = pickle.load(fh)

    # Selected features Regression
    with open('Regression Dependencies/Features for Regression Model.json') as fh:
        regression_features = json.loads(fh.read())

    ### CLASSIFICATION
    # ----------------------------------

    # get only necessary features
    clf_data = allFeaturesData[classification_features]

    # Apply preprocessing function
    clf_data = pd.DataFrame(scaler_for_classification.transform(clf_data),
                                                        columns = clf_data.columns)

    # Model 1 - Applying threshold on decision function
    decisionFuctionModel1 = classification_model_1.decision_function(clf_data)
    threshold = 0.25
    y_predict_model_1 = []
    for j in decisionFuctionModel1:
        if j > threshold:
            y_predict_model_1.append(1)
        else:
            y_predict_model_1.append(0)

    # Model 2
    y_predict_model_2 = classification_model_2.predict(clf_data)

    # Ensemble Model Prediction.  If the flag is set to 'True', we combine results of SVC with RBF + Linear Kernels.
    if ensemble==True:
        y_predict_ensemble = []
        for i in range(len(y_predict_model_1)):
            if (y_predict_model_1[i] == 1) & (y_predict_model_2[i] == 1):
                y_predict_ensemble.append('Positive')
            else:
                y_predict_ensemble.append('Negative')
        allFeaturesData = pd.concat([allFeaturesData, pd.DataFrame(y_predict_ensemble, columns=["Predicted"])], axis=1)

    elif ensemble==False:
        predictions_model_2 =  pd.DataFrame(y_predict_model_2, columns=["Predicted"])
        predictions_model_2.replace({1:'Positive',0:'Negative'}, inplace=True)
        allFeaturesData = pd.concat([allFeaturesData, predictions_model_2], axis=1)
    
    ### REGRESSION
    # ----------------------------------

    # get positively predicted peptides
    reg_data = pd.DataFrame(allFeaturesData[allFeaturesData['Predicted']=='Positive'], columns = allFeaturesData.columns)
    
    # Exception for negative peptides
    if len(reg_data) == 0:
         # save result in a new dataframe
        result = allFeaturesData[['Name','Seq','Predicted']]
        return result
        
    # Apply preprocessing function and select only necessary features
    reg_data_reduced = pd.DataFrame(scaler_for_regression.transform(reg_data.iloc[:,2:-1]), 
                                                        columns = reg_data.columns[2:-1])[regression_features]

    # Predict the buckets.
    buckets_pred = classification_model_for_buckets.predict(reg_data_reduced)
    reg_data_reduced['Bucket'] = buckets_pred

    # Fixed Ki range and Source Interval
    ki_range = (-11.330603908176274, 17.19207365866807)
    source_interval = (-5,5)

    # Make predictions for all of the buckets. The large bucket is predict as 0. Only make predictions if the arrays aren't empty.
    if reg_data_reduced[buckets_pred==0].size != 0:
        sml_pred = regression_model_small_bucket.predict(reg_data_reduced[buckets_pred==0].iloc[:,:-1])
        sml_pred = np.exp(np.interp(sml_pred, source_interval, ki_range))
    if reg_data_reduced[buckets_pred==1].size != 0:
        med_pred = regression_model_medium_bucket.predict(reg_data_reduced[buckets_pred==1].iloc[:,:-1])
        med_pred = np.exp(np.interp(med_pred, source_interval, ki_range))
    lrg_pred = np.zeros(np.count_nonzero(reg_data_reduced[buckets_pred==2]))

    # Put back the predictions in the original order.
    y_predict_regression = np.array([])
    for i in buckets_pred:
        if i == 0:
            y_predict_regression = np.append(y_predict_regression, sml_pred[0])
            sml_pred = np.delete(sml_pred, 0)
        elif i == 1:
            y_predict_regression = np.append(y_predict_regression, med_pred[0])
            med_pred = np.delete(med_pred, 0)
        elif i == 2:
            y_predict_regression = np.append(y_predict_regression, lrg_pred[0])
            lrg_pred = np.delete(lrg_pred, 0)

    reg_data['KI (nM) Predicted'] = y_predict_regression

    allFeaturesData = pd.merge(allFeaturesData,reg_data[['Seq','KI (nM) Predicted']],on='Seq', how='left')

    # save result in a new dataframe
    result = allFeaturesData[['Name','Seq','Predicted','KI (nM) Predicted']]

    return result