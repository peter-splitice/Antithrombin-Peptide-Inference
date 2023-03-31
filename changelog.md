#### 11/20/2022 (Peter)
    1. Added functionality to handle .csv files.
    2. Added functionality to dynamically export results.
        - The names are the same as the imported names.
    3. Ran results on the "combined_hits.csv" file with ~581 extracted peptides.
    4. More general cleanup/commenting to code.
    
#### 11/21/2022 (Nivedha)
    1. Changed the file name to inference_with_hits
    2. Added Positive Hits ipynb file
    3. It takes the combined csv file gives out peptides sorted based on confidence score

#### 11/26/2022 (Peter)
    1. Recreated the regression models with scikit-learn version 1.1.2 properly installed.
    2. Renamed them to the proper format (.pkl).

#### 11/28/2022 (Peter)
    1. Split the "inference.py" script into three scripts:
        a. "inference.py" holds the inference functions.
        b. "pipeline.py" holds the model pipeline
        c. "main.py" calls our entire pipeline/inference
    2. More general clean-up.

#### 12/1/2022 (Peter)
    1. Cleaned up the file imports in pipeline.py
    2. pipeline.py changes migrated to inference.ipynb

#### 12/17/2022 (Peter)
    1. Placed the dependencies and generated files in their own folders to clean up the directory before exporting to github.
    2. Created a github repo for this project

#### 12/26/2022 (Peter)
    1. Updated "pipeline.py"
        a. Changed the features included and models used to be the RFE models (see below).
        b. Added functionality for PCA in the model pipeline.
        c. Added global variable "VARIANCE" that is used to determine how many principal components to select.
            - This current version uses 90% variance.
    2. Updated "Regression Dependencies" wit the following (did not remove old models and dependencies):
        a. Added "Lasso Regression trained model medium bucket (rfe).pkl"
        b. Added "Lasso Regression trained model small bucket (rfe).pkl"
        c. Added "rfe_selected_features.json"
        d. Added "SVC with Linear Kernel 10.00 rfe-pca.pkl"
        e. Added "SVC with Linear Kernel trained model (rfe).pkl"

#### 12/29/2022 (Peter)
    1. Updated "Regression Dependencies"
        a. Added "SVR with Linear Kernel trained model medium bucket (rfe).pkl"
        b. Added "SVR with Linear Kernel trained model small bucket (rfe).pkl"
        c. Added "SVR with RBF Kernel trained model medium bucket (rfe).pkl"
        d. Added "SVR with RBF Kernel trained model small bucket (rfe).pkl"
    2. Updated "pipeline.py"
        a. Added notes to the code.

#### 1/2/2023 (Peter)
    1. Fixed bug generating a Lasso model instead of SVR with RBF Kernel for the trained medium/small (rfe) bucket models.


#### 3/17/2023 (Peter)
    1. Created new branch 'reg_only_update' that has updates for the regression only pipeline.  Merge the branches once regression only section is done.

#### 3/31/2023 (Peter)
    1. Changed the variance on the regression only model to 85%.