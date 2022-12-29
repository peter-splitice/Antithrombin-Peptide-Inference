## Dependency Imports:
# Compute protein descriptors
from propy import PyPro
from propy import AAComposition
from propy import CTD

# Build Sequence Object
from Bio.SeqUtils.ProtParam import ProteinAnalysis

# Read Fasta File
from pyfaidx import Fasta

# Grouping iterable
from itertools import chain

# Return file path
import glob

# Unpack Files
import json
import pickle

# Dataframes
import pandas as pd
import numpy as np

## Classification Model Imports:
# Scaler
file = open('Scaler Classification.pkl', 'rb')
scaler_for_classification = pickle.load(file)
file.close()

# First Model
file = open('SVM Linear Classification trained.pkl', 'rb')
classification_model_1 = pickle.load(file)
file.close()

# Second Model
file = open('SVM RBF Classification trained.pkl', 'rb')
classification_model_2 = pickle.load(file)
file.close()

# Selected features Classification
file = open("Features for Classification Model.json")
classification_features = json.loads(file.read())
file.close()

## Regression Model Imports:
# Scaler
file = open('Scaler Regression.pickle', 'rb')
scaler_for_regression = pickle.load(file)
file.close()

# Classification with ki
file = open('SVC RBF bucket Classification trained.pickle', 'rb')
classification_model_for_buckets = pickle.load(file)
file.close()

# Regression for Medium Bucket
file = open('SVR RBF medium bucket Regression trained.pickle', 'rb')
regression_model_medium_bucket = pickle.load(file)
file.close()

# Regression for Small Bucket
file = open('SVR RBF small bucket Regression trained.pickle', 'rb')
regression_model_small_bucket = pickle.load(file)
file.close()

# Selected features Regression
file = open("Features for Regression Model.json")
regression_features = json.loads(file.read())
file.close()

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
    # CLASSIFICATION
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
    
    # REGRESSION
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

    ## Inference Function for Fasta File:
def inferenceFasta(fastafile, ensemble=True):
    
    """ The inference function gets the protein sequence, trained model, preprocessing function and selected
    features as input. 

    The function read the sequence as string and extract the peptide features using appropriate packages into 
    the dataframe.

    The necessary features are selected from the extracted features which then undergoes preprocessing function, the
    target value is predicted using trained function and give out the results. """
    
    new_peptides = []
    for file in glob.glob(fastafile):
        new_peptides.append(file)
        
    for f in new_peptides:
        fa = Fasta(f)
        # empty list to save the features
        allFeaturesData = []
        for seq in fa:
            # Make sure the sequence is a string
            s = str(seq)
            
            # replace the unappropriate peptide sequence to A
            s = s.replace('X','A')
            s = s.replace('x','A')
            s = s.replace('U','A')
            s = s.replace('Z','A')
            s = s.replace('B','A')

            # Calculating primary features
            analysed_seq = ProteinAnalysis(s)
            wt = analysed_seq.molecular_weight()
            arm = analysed_seq.aromaticity()
            instab = analysed_seq.instability_index()
            flex = analysed_seq.flexibility()
            pI = analysed_seq.isoelectric_point()

            # create a list for the primary features
            pFeatures = [seq.name, s, len(seq), wt, arm, instab, pI]

            # Get Amino Acid Composition (AAC), Composition Transition Distribution (CTD) and Dipeptide Composition (DPC)
            resultAAC = AAComposition.CalculateAAComposition(s)
            resultCTD = CTD.CalculateCTD(s)
            resultDPC = AAComposition.CalculateDipeptideComposition(s)

            # Collect all the features into lists
            aacFeatures = [j for i,j in resultAAC.items()]
            ctdFeatures = [l for k,l in resultCTD.items()]
            dpcFeatures = [n for m,n in resultDPC.items()]
            allFeaturesData.append(pFeatures + aacFeatures + ctdFeatures + dpcFeatures)
        
        # Collect feature names
        pFeaturesName = ['Name','Seq' ,'SeqLength','Weight','Aromaticity','Instability','IsoelectricPoint']
        aacFeaturesData = [i for i,j in resultAAC.items()]
        ctdFeaturesData = [k for k,l in resultCTD.items()]
        dpcFeaturesData = [m for m,n in resultDPC.items()]
        
        featuresName  = []
        featuresName.append(pFeaturesName+aacFeaturesData+ctdFeaturesData+dpcFeaturesData)
        featuresFlattenList = list(chain.from_iterable(featuresName))
        
        # create dataframe using all extracted features and the names
        allFeaturesData = pd.DataFrame(allFeaturesData, columns = featuresFlattenList)
        
        result = model_pipeline(allFeaturesData, ensemble)

        return result

## Inference Function for Single Sequences:
def inferenceSingleSeqence(seq, ensemble=True):
    
    """ The inference function gets the protein sequence, trained model, preprocessing function and selected
    features as input. 
    
    The function read the sequence as string and extract the peptide features using appropriate packages into 
    the dataframe.
    
    The necessary features are selected from the extracted features which then undergoes preprocessing function, the
    target value is predicted using trained function and give out the results. """
    
    # empty list to save the features
    allFeaturesData = []
    
    # Make sure the sequence is a string
    s = str(seq)
    
    # replace the unappropriate peptide sequence to A
    s = s.replace('X','A')
    s = s.replace('x','A')
    s = s.replace('U','A')
    s = s.replace('Z','A')
    s = s.replace('B','A')
    
    # Calculating primary features
    analysed_seq = ProteinAnalysis(s)
    wt = analysed_seq.molecular_weight()
    arm = analysed_seq.aromaticity()
    instab = analysed_seq.instability_index()
    flex = analysed_seq.flexibility()
    pI = analysed_seq.isoelectric_point()
    
    # create a list for the primary features
    pFeatures = [seq, s, len(seq), wt, arm, instab, pI]
     
    # Get Amino Acid Composition (AAC), Composition Transition Distribution (CTD) and Dipeptide Composition (DPC)
    resultAAC = AAComposition.CalculateAAComposition(s)
    resultCTD = CTD.CalculateCTD(s)
    resultDPC = AAComposition.CalculateDipeptideComposition(s)
    
    # Collect all the features into lists
    aacFeatures = [j for i,j in resultAAC.items()]
    ctdFeatures = [l for k,l in resultCTD.items()]
    dpcFeatures = [n for m,n in resultDPC.items()]
    allFeaturesData.append(pFeatures + aacFeatures + ctdFeatures + dpcFeatures)
    
    # Collect feature names
    name1 = ['Name','Seq' ,'SeqLength','Weight','Aromaticity','Instability','IsoelectricPoint']
    name2 = [i for i,j in resultAAC.items()]
    name3 = [k for k,l in resultCTD.items()]
    name4 = [m for m,n in resultDPC.items()]
    name  = []
    name.append(name1+name2+name3+name4)
    flatten_list = list(chain.from_iterable(name))
    
    # create dataframe using all extracted features and the names
    allFeaturesData = pd.DataFrame(allFeaturesData, columns = flatten_list)

    result = model_pipeline(allFeaturesData, ensemble)

    return result

## Inference Function for .csv Files:
def inference_csv(csv, ensemble=True):
    sequences = pd.read_csv('combined_hits.csv')
    sequences = sequences.replace(r"^ +| +$", r"", regex=True)
    sequences = sequences['Seq']

    sequence_data = []
    for seq in sequences:
        # Make sure the sequence is a string
        s = str(seq)
        
        # replace the unappropriate peptide sequence to A
        s = s.replace('X','A')
        s = s.replace('x','A')
        s = s.replace('U','A')
        s = s.replace('Z','A')
        s = s.replace('B','A')

        # Calculating primary features
        analysed_seq = ProteinAnalysis(s)
        wt = analysed_seq.molecular_weight()
        arm = analysed_seq.aromaticity()
        instab = analysed_seq.instability_index()
        flex = analysed_seq.flexibility()
        pI = analysed_seq.isoelectric_point()

        # create a list for the primary features
        pFeatures = [seq, s, len(seq), wt, arm, instab, pI]

        # Get Amino Acid Composition (AAC), Composition Transition Distribution (CTD) and Dipeptide Composition (DPC)
        resultAAC = AAComposition.CalculateAAComposition(s)
        resultCTD = CTD.CalculateCTD(s)
        resultDPC = AAComposition.CalculateDipeptideComposition(s)

        # Collect all the features into lists
        aacFeatures = [j for i,j in resultAAC.items()]
        ctdFeatures = [l for k,l in resultCTD.items()]
        dpcFeatures = [n for m,n in resultDPC.items()]
        sequence_data.append(pFeatures + aacFeatures + ctdFeatures + dpcFeatures)

    # Collect feature names
    pFeaturesName = ['Name','Seq' ,'SeqLength','Weight','Aromaticity','Instability','IsoelectricPoint']
    aacFeaturesData = [i for i,j in resultAAC.items()]
    ctdFeaturesData = [k for k,l in resultCTD.items()]
    dpcFeaturesData = [m for m,n in resultDPC.items()]

    featuresName  = []
    featuresName.append(pFeaturesName+aacFeaturesData+ctdFeaturesData+dpcFeaturesData)
    featuresFlattenList = list(chain.from_iterable(featuresName))

    # create dataframe using all extracted features and the names
    sequence_data = pd.DataFrame(sequence_data, columns = featuresFlattenList)

    result = model_pipeline(sequence_data, ensemble)

    return result

## Implementation Function:
def inference(file=str, ensemble=True):
    if file.endswith('.fasta') == True:
        result = inferenceFasta(file, ensemble)
        name = file.split(sep='.')[0]
        result.to_csv('%s_fasta_results.csv' %(name))
    elif file.endswith('.csv') == True:
        result = inference_csv(file, ensemble)
        name = file.split(sep='.')[0]
        result.to_csv('%s_csv_results.csv' %(name))
    else:
        result = inferenceSingleSeqence(file, ensemble)
        result.to_csv('single_results.csv')
    return result

inference('combined_hits.csv', ensemble=True)

inference('APEADQTTPEEKPAEPEPVA', ensemble=True)

inference('APEADQTTPEEKPAEPEPVA', ensemble=False)

inference('QSPLPERQE', ensemble=True)

inference('QSPLPERQE', ensemble=False)

inference('HTLGYINDNEEGPR', ensemble=True)

inference('HTLGYINDNEEGPR', ensemble=False)

inference('smallhits.fasta', ensemble=True)

inference('smallhits.fasta', ensemble=False)

pd.set_option('display.max_rows',None)
inference('longhits.fasta', ensemble=True)

pd.set_option('display.max_rows',None)
inference('longhits.fasta', ensemble=False)