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

# Data Pipeline
from pipeline import *

## Inference Function for Fasta File:
def inferenceFasta(fastafile, ensemble=True):  
    """
    Performs inference on a single .fasta file.

    Parameters
    ----------
    fastafile: String with the name of a .fasta file requiring inference.

    ensemble: Boolean flag determing whether or not to use an ensemble method for classification.

    Returns
    -------
    result: Pandas DataFrame containing information in the format (Name, Peptide Sequence, Predicted Response, Predicted KI (nM)))
    """
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
    """ 
    Performs inference on a single sequence.
    
    Parameters
    ----------
    seq: String with the name of a single peptide sequence.

    ensemble: Boolean flag determing whether or not to use an ensemble method for classification.

    Returns
    -------
    result: Pandas DataFrame containing information in the format (Name, Peptide Sequence, Predicted Response, Predicted KI (nM))) """
    
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
    """
    Performs inference on every peptide within a csv file.

    Parameters
    ----------
    csv: String with the name of a .csv file requiring inference.

    ensemble: Boolean flag determing whether or not to use an ensemble method for classification.

    Returns
    -------
    result: Pandas DataFrame containing information in the format (Name, Peptide Sequence, Predicted Response, Predicted KI (nM)))
    """
    sequences = pd.read_csv(csv)
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
        if ensemble==True:
            result.to_csv('Results/%s_fasta_results_ensemble.csv' %(name))
        else:
            result.to_csv('Results/%s_fasta_results.csv' %(name))
    elif file.endswith('.csv') == True:
        result = inference_csv(file, ensemble)
        name = file.split(sep='.')[0]
        if ensemble==True:
            result.to_csv('Results/%s_csv_results_ensemble.csv' %(name))
        else:
            result.to_csv('Results/%s_csv_results.csv' %(name))
    else:
        result = inferenceSingleSeqence(file, ensemble)
        if ensemble==True:
            result.to_csv('Results/%s_results.csv' %(file))
        else:
            result.to_csv('Results/%s_results.csv' %(file))
    return result