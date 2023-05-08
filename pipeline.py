## Package Imports
# Unpack Files
import json
import pickle

# Data Analysis
import pandas as pd
import numpy as np 

# Change this global variable depending on what variance we choose for PCA.  Set this to 'False' if we don't end up using PCA.
REGRESSION_ONLY_VARIANCE = False
BUCKET_VARIANCE = False
KI_RANGE = (-11.330603908176274, 17.19207365866807)
SOURCE_INTERVAL = (-5,5)

## Pipeline for multi-stage model:
def model_pipeline(allFeaturesData):
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

    # Trained Classification Model
    with open('Classification Dependencies/SVM Linear Classification trained.pkl', 'rb') as fh:
        classification_model_1 = pickle.load(fh)

    # Selected features Classification
    with open('Classification Dependencies/Features for Classification Model.json') as fh:
        classification_features = json.loads(fh.read())

    ### CLASSIFICATION
    # ----------------------------------

    # get only necessary features
    clf_data = allFeaturesData[classification_features]

    # Apply preprocessing function
    clf_data = pd.DataFrame(scaler_for_classification.transform(clf_data),
                                                        columns = clf_data.columns)

    # Model 1 - Applying threshold on decision function
    decisionFuctionModel1 = classification_model_1.decision_function(clf_data)
    threshold = 0.4

    y_predict = []
    for j in decisionFuctionModel1:
        if j > threshold:
            y_predict.append(1)
        else:
            y_predict.append(0)

    predictions =  pd.DataFrame(y_predict, columns=["Predicted"])
    predictions.replace({1:'Positive',0:'Negative'}, inplace=True)
    allFeaturesData = pd.concat([allFeaturesData, predictions], axis=1)
    
    ### REGRESSION
    # ----------------------------------

    # get positively predicted peptides
    reg_data = pd.DataFrame(allFeaturesData[allFeaturesData['Predicted']=='Positive'], columns = allFeaturesData.columns)
    
    # Exception if no peptides are predicted positive.
    if len(reg_data) == 0:
         # save result in a new dataframe
        result = allFeaturesData[['Name','Seq','Predicted']]
        return result

    y_predict_bucketed = bucket_regression_inference_pipeline(reg_data)
    y_predict_unbucketed = regression_only_inference_pipeline(reg_data)

    reg_data['KI (nM) Predicted (bucketized)'] = y_predict_bucketed
    reg_data['KI (nM) Predicted (non-bucketized)'] = y_predict_unbucketed


    allFeaturesData = pd.merge(allFeaturesData,reg_data[['Seq', 'KI (nM) Predicted (bucketized)',
                                                         'KI (nM) Predicted (non-bucketized)']],                                                     
                               on='Seq', how='left')

    # save result in a new dataframe
    result = allFeaturesData[['Name','Seq','Predicted', 'KI (nM) Predicted (bucketized)',
                              'KI (nM) Predicted (non-bucketized)']]

    return result

def bucket_regression_inference_pipeline(reg_data=pd.DataFrame()):
    # Scaler
    with open('Regression Dependencies/Scaler Regression.pkl', 'rb') as fh:
        regression_scaler = pickle.load(fh)

    # Classification with ki
    with open('Regression Dependencies/SVC RBF bucket classification trained.pkl', 'rb') as fh:
        bucket_classification_model = pickle.load(fh)

    # Regression for Medium Bucket
    with open('Regression Dependencies/SVR RBF medium bucket Regression trained.pkl', 'rb') as fh:
        regression_model_medium_bucket = pickle.load(fh)

    # Regression for Small Bucket
    with open('Regression Dependencies/SVR RBF small bucket Regression trained.pkl', 'rb') as fh:
        regression_model_small_bucket = pickle.load(fh)

    # Selected features Regression
    with open('Regression Dependencies/Features for Regression Model.json') as fh:      
        # Changed this back to sequential forward selection.
        regression_features = json.loads(fh.read())

    # Apply preprocessing function and select only necessary features
    reg_data = pd.DataFrame(regression_scaler.transform(reg_data.iloc[:,2:-1]), 
                                                columns = reg_data.columns[2:-1])[regression_features]
    
    if BUCKET_VARIANCE != False:
        with open('Regression Dependencies/SVC with Linear Kernel 10.00 rfe-pca.pkl', 'rb') as fh:
            pca = pickle.load(fh)
        reg_data = pd.DataFrame(pca.transform(reg_data))

        # Dimensionality Reduction based on accepted variance.
        ratios = np.array(pca.explained_variance_ratio_)
        ratios = ratios[ratios.cumsum() <= (BUCKET_VARIANCE/100)]

        # Readjust the dimensions of 'reg_data' based on the variance we want.
        length = len(ratios)
        if length > 0:
            reg_data = reg_data[reg_data.columns[0:length]]

    # Predict the buckets
    buckets_pred = bucket_classification_model.predict(reg_data)
    reg_data['Bucket'] = buckets_pred

    # Make predictions for all of the buckets.  The large bucket is predict as 0.  Only make predictions if the
    #   arrays aren't empty.
    if reg_data[buckets_pred==0].size != 0:
        sml_pred = regression_model_small_bucket.predict(reg_data[buckets_pred==0].iloc[:,:-1])
        sml_pred = np.exp(np.interp(sml_pred, SOURCE_INTERVAL, KI_RANGE))
    if reg_data[buckets_pred==1].size != 0:
        med_pred = regression_model_medium_bucket.predict(reg_data[buckets_pred==1].iloc[:,:-1])
        med_pred = np.exp(np.interp(med_pred, SOURCE_INTERVAL, KI_RANGE))
    lrg_pred = np.zeros(np.count_nonzero(reg_data[buckets_pred==2]))

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

    return y_predict_regression

def regression_only_inference_pipeline(reg_data=pd.DataFrame()):
    """
    This function handles the regression only portion of the inference pipeline if we never had a 
        classification stage in the pipeline.
        
    Parameters
    ----------
    reg_data: Pandas DataFrame containing the table of data for the various peptides in the positive
        dataset.

    Regurns
    -------
    result: A single column containing the predicted KI values 
    """

    ## MODEL IMPORTS
    # MinMaxScaler Transformation
    with open('Regression Only Dependencies/regression only scaler.pkl', 'rb') as fh:
        regression_scaler = pickle.load(fh)

    # sequential Forward Selection Features Selected
    with open('Regression Only Dependencies/regression_only_selected_features.json', 'rb') as fh:
        regression_features = json.loads(fh.read())

    # Regression model
    with open('Regression Only Dependencies/regression only SVR with RBF Kernel trained model.pkl', 'rb') as fh:
        regression_model = pickle.load(fh)

    # Scaler transformation and Feature Selection
    reg_data = pd.DataFrame(regression_scaler.transform(reg_data.iloc[:,2:-1]),
                                    columns = reg_data.columns[2:-1])[regression_features]
    # Apply PCA if needed.
    if REGRESSION_ONLY_VARIANCE != False:
        with open('Regression Only Dependencies/PCA for SVR with RBF Kernel.pkl', 'rb') as fh:
            pca = pickle.load(fh)
        reg_data = pd.DataFrame(pca.transform(reg_data))

        # Dimensionality Reduction based on accepted variance.
        ratios = np.array(pca.explained_variance_ratio_)
        ratios = ratios[ratios.cumsum() < (REGRESSION_ONLY_VARIANCE/100)]

        # Readjust the dimensions of 'reg_data' based on the bariance we get.
        length = len(ratios)
        if length > 0:
            reg_data = reg_data[reg_data.columns[0:length]]

    # Predictions
    y_pred = regression_model.predict(reg_data)
    y_pred = np.exp(np.interp(y_pred, SOURCE_INTERVAL, KI_RANGE))

    return y_pred