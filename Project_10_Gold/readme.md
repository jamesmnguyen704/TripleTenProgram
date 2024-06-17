**Review**

Hi, my name is Dmitry and I will be reviewing your project.
  
You can find my comments in colored markdown cells:
  
<div class="alert alert-success">
  If everything is done successfully.
</div>
  
<div class="alert alert-warning">
  If I have some (optional) suggestions, or questions to think about, or general comments.
</div>
  
<div class="alert alert-danger">
  If a section requires some corrections. Work can't be accepted with red comments.
</div>
  
Please don't remove my comments, as it will make further review iterations much harder for me.
  
Feel free to reply to my comments or ask questions using the following template:
  
<div class="alert alert-info">
  For your comments and questions.
</div>
  
First of all, thank you for turning in the project! You did an excellent job! The project is accepted. Good luck on the next sprint!

# Zyfra Gold Recovery

## Introduction

Zyfra is a company that extracts gold concentrate from gold ore to generate profit. They aim to predict the amount of gold concentrate yielded from gold ore, optimizing their gold production process. The company has gathered data from every stage of the gold extraction process, including a training set, a test set, and a complete dataset. 

The complete dataset includes all observations and features/targets, including those calculated after extraction. The training set is a subset of this data intended for training the prediction model, while the test set contains only features available at the start of the extraction process, with no target data. This test set is crucial for making predictions desired by the company.

To make numerical predictions, a regression model will be trained. We will experiment with linear regression and random forest regression models, omitting decision tree regression as random forest tends to be more accurate. The success metric will be symmetric mean absolute error (sMAPE), based on predicted vs actual gold recovery. The trained models will predict rougher concentrate recovery and final concentrate recovery, and sMAPE will be calculated for each, combining the two scores for the final performance score. Both linear and random forest regression models can predict multiple targets from a single model.

The process will begin with data preprocessing to verify the correctness of recovery calculations and compare the training set with the test set. Missing values will be addressed, and targets selected. Then, the concentrations of each metal at different stages will be compared, identifying any observations that should be dropped. Additionally, feed size distributions between the training and test sets will be compared. 

Finally, models will be trained, combined sMAPE scores calculated, and the better-scoring model selected as the final model. This model will be trained on the full training set, and its final sMAPE will be determined on the test set. A comparison with a dummy model will be made, and the final model will be saved for further use by the company.


# Importing Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error
from sklearn.dummy import DummyRegressor
from joblib import dump

# Preprocessing

try:
    train = pd.read_csv('gold_recovery_train.csv')
except:
    train = pd.read_csv('/datasets/gold_recovery_train.csv')
    
try:    
    test = pd.read_csv('gold_recovery_test.csv')
except:
    test = pd.read_csv('/datasets/gold_recovery_test.csv')
    
try:
    full = pd.read_csv('gold_recovery_full.csv')
except:
    full = pd.read_csv('/datasets/gold_recovery_full.csv')

train.head()

train.info()

test.info()

full.info()

I can see that the full and training datasets contain 87 columns while the test set contains only 53

full.head(2)

<div class="alert alert-success">
<b>Reviewer's comment</b>

The data was loaded and inspected

</div>

# Verify recovery

def calc_recov(C, F, T):
    
    """
    This function calculates gold recovery.
    
    To calculate rougher recovery, this function takes three series as inputs:
    - C = rougher.output.concentrate_au
    - F = rougher.input.feed_au
    - T = rougher.output.tail_au
        and returns the percentage recovery of gold concentrate after the roughing stage.
    
    To calculate final recovery, this function takes three measurements as inputs:
    - C = final.output.concentrate_au
    - F = rougher.output.concentrate_au
    - T = final.tail_au
        and returns the percentage recovery of gold concentrate after the final purification stage.
    """
    
    recovery = (C*(F-T)) / (F*(C-T)) * 100
    
    return recovery

To calculate recover for a row in the full dataset using my function to ensure that it work correctly, and that the values make sense.  I need to drop the nans before I can caluculate mean absolute error. 

train_cut = train.dropna(subset=['rougher.output.recovery', 'rougher.output.concentrate_au', 'rougher.input.feed_au', \
                                 'rougher.output.tail_au']) 

recov = train_cut['rougher.output.recovery']
pred_recov = calc_recov(train_cut['rougher.output.concentrate_au'], train_cut['rougher.input.feed_au'], train_cut['rougher.output.tail_au'])
mean_absolute_error(recov, pred_recov)

This confirms that the recovery equation matches the recover measurement exactly

<div class="alert alert-success">
<b>Reviewer's comment</b>

Recovery data was validated successfully!

</div>

# Compare training set and test set

Let's see if there is differences in the columns betwe3en the training set and test set. The training set has more columns so we should see if anything is missing. 

test_columns = test.columns

drop_columns = train.drop(test_columns, axis=1)
drop_columns.info()

It seems that the missing values in the test set pertain to outputs and calculations, representing data that wouldn't be accessible at the beginning of the extraction process. This test set replicates the data Zyfra would have during real-world predictions. We need to determine which data we want to predict from this list.


<div class="alert alert-success">
<b>Reviewer's comment</b>

Yep, you are right!
    
</div>

# Preprocess Datasets

## Training Set

I will begin by refining the training dataset to include only the necessary features and targets, followed by dropping any NaN values. The table will then be divided into training features and training targets. Subsequently, I will handle the test set by dropping NaNs and obtaining targets from the full dataset. Additionally, I will compile a list of columns containing concentration data for later use, as well as a list of concentration columns that need to be dropped from the training set.

The company states that observations in the datasets are arranged chronologically, with similar observations usually occurring adjacently. Although I plan to employ cross-validation for model training later on, I believe it would be beneficial to shuffle the data once preprocessing is complete. This should not impact linear regression, but it may affect the random forest model.

targets = ['rougher.output.recovery', 'final.output.recovery']

conc = ['final.output.concentrate_au', 'rougher.output.concentrate_au', 'rougher.input.feed_au', \
           'final.output.concentrate_ag', 'rougher.output.concentrate_ag', 'rougher.input.feed_ag', \
           'final.output.concentrate_pb', 'rougher.output.concentrate_pb', 'rougher.input.feed_pb', \
           'final.output.concentrate_sol', 'rougher.output.concentrate_sol', 'rougher.input.feed_sol']

conc_drop = ['final.output.concentrate_sol', 'rougher.output.concentrate_sol', \
                 'final.output.concentrate_pb', 'rougher.output.concentrate_pb', \
                 'final.output.concentrate_ag', 'rougher.output.concentrate_ag', \
                 'final.output.concentrate_au', 'rougher.output.concentrate_au']

drop_columns = drop_columns.drop(targets, axis=1)
drop_columns = drop_columns.drop(conc_drop, axis=1)
trim_train = train.drop(drop_columns.columns, axis=1)

trim_train.info()

To handle missing values and preserve the temporal sequence of observations, I will employ the forward fill method in a DataFrame. This method will propagate non-null values forward, effectively filling missing values with the preceding non-null value. By doing so, we ensure that the data's sequential order is maintained while addressing missing values.


trim_train = trim_train.fillna(method='ffill')

Time to check to see if there is any duplicate observation

print(trim_train.duplicated().sum())

No duplicates
Time to split the training set into features and targets

target_train = trim_train[targets]
features_train = trim_train.drop(targets, axis=1).drop('date', axis=1).drop(conc_drop, axis=1)

final_conc = trim_train[conc]

features_train

Shuffling the data, using same random state for each 

target_train = target_train.sample(frac=1, random_state=0)
features_train = features_train.sample(frac=1, random_state=0)

conc_train = final_conc.sample(frac=1, random_state=0)

target_train.head(2)

features_train.head(2)

conc_train.head(2)

Done. Indexes match positions.

Certain groups of features exhibit similar scales, while others do not. For instance, concentrations of materials are on significantly different scales compared to air volume and fluid levels. Although some standardization might be necessary, I will initially run my models without standardizing the data. If required, I can revisit this later to experiment with standardization. While the linear regression model may benefit from standardization, the random forest regression model is unlikely to be affected.

Collect targets for test set
I will perform similar preprocessing on the test set, making similar decisions, as I did for the training set.

The test set does not come with target values, as the test set represents the data that will be available to the company before the extraction process. I want to be able to test the performance of the models by having a target available to me for comparison, if possible. I will create an extra dataset that contains the full data for the test observations, then I can collect the target values into a series for later comparison with the predictions.

print(test.date.duplicated().sum())
print(full.date.duplicated().sum())

It seems the date is unilque for testing

test_full = full[full.date.isin(test.date)]
test_full_trimmed = test_full.fillna(method='ffill')

Spliting the dataframe into features and target

features_test = test_full_trimmed.drop('date', axis=1).drop(drop_columns.columns, axis=1).drop(targets, axis=1).drop(conc_drop, axis=1).reset_index(drop=True)
target_test = test_full_trimmed[targets].reset_index(drop=True)

conc_test = test_full_trimmed[conc].reset_index(drop=True)

<div class="alert alert-success">
<b>Reviewer's comment</b>

Targets for the test set were identified correctly

</div>

features_test

features_test = features_test.sample(frac=1, random_state=0)
target_test = target_test.sample(frac=1, random_state=0)

conc_test = conc_test.sample(frac=1, random_state=0)


features_test.head(2)

target_test.head(2)

conc_test.head(2)

Featues and Targets match the indexes

<div class="alert alert-success">
<b>Reviewer's comment</b>

Good job on data preprocessing!

</div>

## Analyze Data: Check Concentrations of Metals

To examine the variations in metal concentrations across different stages of extraction, I will analyze the full dataset. My prediction is that the concentration of gold will increase, whereas the concentrations of silver and lead will decrease.

def metals_eda(metal):
    
    """
    This function takes a string as input that represents the metal to evaluate concentration distributions for each. 
    
    For each metal we will look at both the distributions for concentrations for the concentrates and the tails, 
        in the form of boxplots ordered from raw feed to final stage. 
    """
    
    conc_cols = ['rougher.input.feed_', 'rougher.output.concentrate_', 'primary_cleaner.output.concentrate_', \
                 'final.output.concentrate_']
    conc_cols = [column + metal for column in conc_cols]
    
    tail_cols = ['rougher.input.feed_', 'rougher.output.tail_', 'primary_cleaner.output.tail_', \
                 'secondary_cleaner.output.tail_', 'final.output.tail_']
    tail_cols = [column + metal for column in tail_cols]
    
    plt.figure(figsize=(14,4))
    sns.boxplot(data=full[conc_cols])
    plt.title(f"Concentration of concentrate at each stage for {metal.capitalize()}")
    
    plt.figure(figsize=(14,4))
    sns.boxplot(data=full[tail_cols])
    plt.title(f"Concentration of tail at each stage for {metal.capitalize()}")

AU first 

metals_eda('au')

It's indeed encouraging to observe that the concentration of gold concentrate increases steadily throughout the extraction process. Additionally, maintaining a consistently low concentration of gold in the tails aligns with the desired outcome, as ideally, there should be minimal to zero gold in the tails for an efficient gold extraction process. The concentration of gold starts at approximately 10 and rises to nearly 50 by the conclusion of the process.

However, it's concerning to note the presence of outliers where the final gold concentration is low in the concentrate but high in the tails. Such instances deviate from the ideal scenario and warrant further investigation to identify potential factors contributing to this discrepancy.

metals_eda('pb')

It's observed that the concentration of lead begins at a relatively low level around 3 and gradually increases to approximately 10 by the end of the extraction process. Although this concentration is considerably less than that of gold, it still undergoes a notable increment throughout the stages.

Furthermore, the concentration of lead in the tails exhibits some fluctuations but generally remains within the same range throughout the process. This stability in tail concentration suggests a consistent performance in managing lead throughout the extraction proces

metals_eda('ag')

It's noted that the concentration of silver in the concentrate increases following the rougher stage, but ultimately ends up lower than its initial concentration by the conclusion of the process. Despite this decrease, there remains a relatively high concentration of silver in the tails at each stage of the extraction process.

Interestingly, the purification process appears to be more effective at reducing the concentration of silver compared to lead. This observation underscores the varying efficiencies of the purification process in treating different metals, highlighting a potential area for optimization or further investigation.

<div class="alert alert-success">
<b>Reviewer's comment</b>

Conclusions make sense!

</div>

# Remove Flawed Observations

Upon reviewing the boxplots generated earlier, I observed that several final gold concentrations are unusually low. While such occurrences might be considered normal if the final concentrations exceed the original input concentration, I am concerned about the impact of these observations on the training and test datasets. Therefore, I intend to investigate the effect of removing observations where the final concentration of gold is lower than or equal to the concentration of gold in the input feed.

This scenario should be extremely uncommon in a gold extraction process. I will develop a function to compare these values and subsequently modify the datasets. Additionally, the function will offer the flexibility to compare and adjust other values as required.

def trim_rows(features_df, target_df, conc_df, features_col=0, target_col=0, condition='greater', indexes=0):
    
    """
    This function trims rows from a set of three dataframes (training or testing sets, plus concentrations set) based on 
        a certain measurement from one dataframe being greater or less than the other. 
    
    Takes each of the two dataframes as input, then the concentrations dataframe, then the columns that I wish to compare. 
        The "greater" toggle is set to True but can be set to False to change the direction of the comparison.
    
    The function prints the original length and the new length, and returns the three dataframes as a tuple.
    """
    
    print("Length of original dataframes:", features_df.shape[0])
    
    if condition == 'greater':
        bad_rows = features_df[features_df[features_col] >= conc_df[target_col]].index
    elif condition == 'less':
        bad_rows = features_df[features_df[features_col] <= conc_df[target_col]].index
    else:
        bad_rows = indexes

    features_df = features_df.drop(bad_rows, errors='ignore')
    target_df = target_df.drop(bad_rows, errors='ignore')
    conc_df = conc_df.drop(bad_rows, errors='ignore')
     
    print("Length of trimmed dataframes:", features_df.shape[0])
    
    return features_df, target_df, conc_df
    return features_df, target_df, conc_df

Trim Training Set: Remove Low Gold Concentration Observations

To begin, we will trim down the training set by removing observations where the concentration of gold at any of the advanced stages (e.g., rougher.output.concentrate_au, primary_cleaner.output.concentrate_au) is equal to or less than the original concentration (e.g., rougher.input.feed_au).

features_train, target_train, conc_train = trim_rows(
    features_train, target_train, conc_train, 'rougher.input.feed_au', 'rougher.output.concentrate_au')

features_train, target_train, conc_train = trim_rows(
    features_train, target_train, conc_train, 'rougher.input.feed_au', 'final.output.concentrate_au')

features_test, target_test, conc_test = trim_rows(
    features_test, target_test, conc_test, 'rougher.input.feed_au', 'rougher.output.concentrate_au')

features_test, target_test, conc_test = trim_rows(
    features_test, target_test, conc_test, 'rougher.input.feed_au', 'final.output.concentrate_au')

About 600 flawed observations were removed during the trimming process based on gold concentration values. It's noted that the relationships between the concentrations of silver or lead at different stages of the extraction process don't elicit strong feelings. From the boxplots, it's observed that the concentration of silver rises after the rougher stage but falls after the purification stages, while the concentration of lead increases consistently throughout the extraction process. With the removal of these obvious anomalies, we can be more confident that our model will not be affected by such flawed observations.

<div class="alert alert-success">
<b>Reviewer's comment</b>

Good idea to filter out outliers!

</div>

Check feed size distributions
Next we should compare the feed particle size between the training set and the test set to ensure that the distributions are similar.

plt.figure(figsize=(4,3))
plt.title('Feed size in train vs test sets')

mask = 'rougher.input.feed_size'
features_train[mask][features_train[mask] <= features_train[mask].quantile(0.985)].hist(alpha=0.6, density=True)
features_test[mask][features_test[mask] <= features_test[mask].quantile(0.985)].hist(alpha=0.6, density=True)

# Feed Particle Size Distribution Comparison

The distributions of feed particle size in both the training and test sets exhibit remarkable similarity. Both distributions are predominantly normal, with peaks around 50, and display a slight right-skewness. This consistency suggests that our model evaluation should proceed effectively.

It's worth noting that there are some significantly higher outliers, which have been omitted from the histogram by considering only the lower 98.5% of values.

<div class="alert alert-success">
<b>Reviewer's comment</b>

Feed size distributions were compared successfully!

</div>

## Total Concentrations Investigation

To better understand the total concentrations of concentrate at various stages of the extraction process (raw feed, rougher concentrate, and final concentrate), and to potentially identify and remove abnormal values, we will conduct an analysis using the training set data.

We will sum the concentrations at each stage and visualize the distributions by plotting histograms for each stage.

input_conc_sums = conc_train['rougher.input.feed_au'] + conc_train['rougher.input.feed_ag'] + conc_train['rougher.input.feed_pb'] + \
    conc_train['rougher.input.feed_sol']
input_drop_indexes = input_conc_sums[input_conc_sums < input_conc_sums.quantile(0.015)].index
input_conc_sums[input_conc_sums > input_conc_sums.quantile(0.015)].hist()
plt.title('Input stage raw feed - training')
plt.show()

rougher_conc_sums = conc_train['rougher.output.concentrate_au'] + conc_train['rougher.output.concentrate_ag'] + \
    conc_train['rougher.output.concentrate_pb'] + conc_train['rougher.output.concentrate_sol']
rougher_drop_indexes = [rougher_conc_sums < rougher_conc_sums.quantile(0.015)].index
rougher_conc_sums[rougher_conc_sums > rougher_conc_sums.quantile(0.015)].hist()
plt.title('Rougher stage output concentrate - training')
plt.show()

final_conc_sums = conc_train['final.output.concentrate_au'] + conc_train['final.output.concentrate_ag'] + \
    conc_train['final.output.concentrate_pb'] + conc_train['final.output.concentrate_sol']
final_drop_indexes = [final_conc_sums < final_conc_sums.quantile(0.025)].index
final_conc_sums[final_conc_sums > final_conc_sums.quantile(0.025)].hist()
plt.title('Final stage output concentrate - training')
plt.show()

I have omitted outliers from each histogram with as conservative a cutoff as possible to achieve greater normality. I will drop the outlier values from the training and testing sets using these quantile values.

features_train, target_train, conc_train = trim_rows(
    features_train, target_train, conc_train, condition='n', indexes=input_drop_indexes)

features_train, target_train, conc_train = trim_rows(
    features_train, target_train, conc_train, condition='n', indexes=rougher_drop_indexes)

features_train, target_train, conc_train = trim_rows(
    features_train, target_train, conc_train, condition='n', indexes=final_drop_indexes)

input_conc_sums = conc_test['rougher.input.feed_au'] + conc_test['rougher.input.feed_ag'] + conc_test['rougher.input.feed_pb'] + \
    conc_test['rougher.input.feed_sol']
input_drop_indexes = input_conc_sums[input_conc_sums < input_conc_sums.quantile(0.015)].index
input_conc_sums[input_conc_sums > input_conc_sums.quantile(0.02)].hist()
plt.title('Input stage raw feed - test')
plt.show()

rougher_conc_sums = conc_test['rougher.output.concentrate_au'] + conc_test['rougher.output.concentrate_ag'] + \
    conc_test['rougher.output.concentrate_pb'] + conc_test['rougher.output.concentrate_sol']
rougher_drop_indexes = [rougher_conc_sums < rougher_conc_sums.quantile(0.015)].index
rougher_conc_sums[rougher_conc_sums > rougher_conc_sums.quantile(0.02)].hist()
plt.title('Rougher stage output concentrate - test')
plt.show()

final_conc_sums = conc_test['final.output.concentrate_au'] + conc_test['final.output.concentrate_ag'] + \
    conc_test['final.output.concentrate_pb'] + conc_test['final.output.concentrate_sol']
final_drop_indexes = [final_conc_sums < final_conc_sums.quantile(0.025)].index
final_conc_sums[final_conc_sums > final_conc_sums.quantile(0.04)].hist()
plt.title('Final stage output concentrate - test')
plt.show()

To improve the normality of each histogram, I have removed outliers using a conservative cutoff. This was done to enhance the normality of the distributions.

Following this, I will eliminate outlier values from both the training and testing sets using the quantile values obtained from the histograms.

features_train, target_train, conc_train = trim_rows(
    features_train, target_train, conc_train, condition='n', indexes=input_drop_indexes)

features_train, target_train, conc_train = trim_rows(
    features_train, target_train, conc_train, condition='n', indexes=rougher_drop_indexes)

features_train, target_train, conc_train = trim_rows(
    features_train, target_train, conc_train, condition='n', indexes=final_drop_indexes)

Time to test set

input_conc_sums = conc_test['rougher.input.feed_au'] + conc_test['rougher.input.feed_ag'] + conc_test['rougher.input.feed_pb'] + \
    conc_test['rougher.input.feed_sol']
input_drop_indexes = input_conc_sums[input_conc_sums < input_conc_sums.quantile(0.015)].index
input_conc_sums[input_conc_sums > input_conc_sums.quantile(0.02)].hist()
plt.title('Input stage raw feed - test')
plt.show()

rougher_conc_sums = conc_test['rougher.output.concentrate_au'] + conc_test['rougher.output.concentrate_ag'] + \
    conc_test['rougher.output.concentrate_pb'] + conc_test['rougher.output.concentrate_sol']
rougher_drop_indexes = [rougher_conc_sums < rougher_conc_sums.quantile(0.015)].index
rougher_conc_sums[rougher_conc_sums > rougher_conc_sums.quantile(0.02)].hist()
plt.title('Rougher stage output concentrate - test')
plt.show()

final_conc_sums = conc_test['final.output.concentrate_au'] + conc_test['final.output.concentrate_ag'] + \
    conc_test['final.output.concentrate_pb'] + conc_test['final.output.concentrate_sol']
final_drop_indexes = [final_conc_sums < final_conc_sums.quantile(0.025)].index
final_conc_sums[final_conc_sums > final_conc_sums.quantile(0.04)].hist()
plt.title('Final stage output concentrate - test')
plt.show()

features_test, target_test, conc_test = trim_rows(
    features_test, target_test, conc_test, condition='n', indexes=input_drop_indexes)

features_test, target_test, conc_test = trim_rows(
    features_test, target_test, conc_test, condition='n', indexes=rougher_drop_indexes)

features_test, target_test, conc_test = trim_rows(
    features_test, target_test, conc_test, condition='n', indexes=final_drop_indexes)

We have removed the outliers

<div class="alert alert-success">
<b>Reviewer's comment</b>

Excellent!

</div>

# Finalizing Preprocessing

With all other preprocessing steps completed and the training/testing dataframes finalized, I will now reset the indexes.

features_train = features_train.reset_index(drop=True)
target_train = target_train.reset_index(drop=True)

features_test = features_test.reset_index(drop=True)
target_test = target_test.reset_index(drop=True)

# Model Building

It's time to construct the regression models. I'll experiment with a linear regression model and a random forest regressor model, and assess their performance using cross-validation. Evaluation will be based on symmetric mean absolute percent error (sMAPE). The model with the highest sMAPE score will be selected, and its performance will be evaluated on the test set to obtain a final sMAPE score.

# Calculate sMAPE

Next, I'll develop functions to compute sMAPE and to combine the rougher and final sMAPE scores into a final score.

def calc_smape(actual, pred):
    
    """
    This function takes a series of answers and of predictions, and calculates symmetric mean absolute percentage error (sMAPE).
        It returns the sMAPE value.
    """
    
    N = len(actual)
    sum = 0
    for i in range(N):
        numerator = abs(actual[i] - pred[i]) * 100
        denominator = (abs(actual[i]) + abs(pred[i])) / 2
        sum += (numerator/denominator)
    smape = sum / N
    
    return smape

def combine_smapes(rougher, final):
    
    """
    Calculates a combined sMAPE score using the rougher concentrate sMAPE and final concentrate sMAPE score as inputs.
    """
    
    return (0.25 * rougher) + (0.75 * final)

<div class="alert alert-success">
<b>Reviewer's comment</b>

The functions for SMAPE calculation are correct

</div>

# Training and Evaluating the Models

Typically, I would utilize the cross-validation function provided by sklearn. However, due to having multiple targets and a different evaluation metric, I'll need to employ a more customizable approach. Cross-validation will enable me to choose the model with the lower sMAPE score. Once selected, I will train that model type using the entire training dataset and then evaluate its performance on the test set to obtain a final tested sMAPE score.

def cross_val(features, targets, model_type, sets=5, n_estimators=10, max_depth=5):

    """ 
    This function cross-validates regression models, calculates sMAPE using a separate function, and prints the average combined
        sMAPE score.
    
    The features and targets dataframes are the first inputs, followed by 'LR' or 'RF' to train linear regression or random forest
        regression models, respectively. The function, by default, uses five rotating sets for cross-validation, but this value 
        can be changed.
    """
    
    scores = []
    sample_size = int(len(features)/sets)
    
    k = 0 # Create counter to count the number of for-loop iterations
    
    for i in range(0, len(features), sample_size):
        valid_indexes = list(range(i, i + sample_size))
        train_indexes = list(range(0, i)) + list(range(i + sample_size, len(features)))

#         Split variables features and target into samples features_train, target_train, features_valid, target_valid
        features_train = features.iloc[train_indexes].reset_index(drop=True)
#         display(features_train)
        features_valid = features.iloc[valid_indexes].reset_index(drop=True)
#         display(features_valid)
        target_train = targets.iloc[train_indexes].reset_index(drop=True)
        target_valid = targets.iloc[valid_indexes].reset_index(drop=True)

#         Build model and store predictions based on model type chosen
        if model_type == 'LR':
            model = LinearRegression()
            model.fit(features_train, target_train)
            pred = pd.DataFrame(model.predict(features_valid))
        if model_type == 'RF':
            model = RandomForestRegressor(max_features=1.0, n_estimators=n_estimators, max_depth=max_depth, random_state=0)
            model.fit(features_train, target_train)
            pred = pd.DataFrame(model.predict(features_valid))

#         Store answers and specific predictions
        rougher_recov_actual = target_valid['rougher.output.recovery']
        final_recov_actual = target_valid['final.output.recovery']
        rougher_recov_pred = pred.iloc[:,0]
        final_recov_pred = pred.iloc[:,1]    

#         Calculate sMAPE
        rougher_smape = calc_smape(rougher_recov_actual, rougher_recov_pred)
        final_smape = calc_smape(final_recov_actual, final_recov_pred)
        combined_smape = combine_smapes(rougher_smape, final_smape)
    
        scores.append(combined_smape)
        
#         Break loop when number of desired sets is reached
        k += 1
        if k == sets:
            break
            
    final_score = round(pd.Series(scores).mean(), 2)
    print(f'Average model symmetric mean percent error: {final_score}%')

<div class="alert alert-success">
<b>Reviewer's comment</b>

Good job on creating a custom cross-validation function using our target metric!

</div>

Now we are using the function to find the average combined sMAPE for for linear regression model 

%%time
cross_val(features_train, target_train, 'LR')

# Random Forest Regression Model Tuning
And now for the random forest regression model. I will first loop through max_depth values with a low, constant n_estimators value and then choose the max_depth that yields the lowest sMAPE. If the sMAPE for multiple options are very close, I may choose hyperparameters that require less processing power/runtime. After that I will loop through some n_estimators values with the constant max_depth that performed best in the first loop. Breaking this process into two independent loops should cut down on overall runtime.


for depth in range(15,21,1):
    print(f'max_depth: {depth}')
    cross_val(features_train, target_train, 'RF', n_estimators=20, max_depth=depth, sets=4)

## Model Tuning: Maximum Depth Selection

To avoid overfitting the model, I will limit the exploration of max_depth values to a maximum of 20. Beyond this threshold, the gains in performance tend to diminish, suggesting a risk of overfitting. Therefore, I will select 20 as my preferred max_depth.

for n_est in range(40,81,10):
    print(f'n_estimators: {n_est}')
    cross_val(features_train, target_train, 'RF', n_estimators=n_est, max_depth=20, sets=4)

<div class="alert alert-success">
<b>Reviewer's comment</b>

Great, you tried a couple of different models and did some hyperparameter tuning using cross-validation with our target metric

</div>

# Model Selection and Training

Similar to the search for the optimal max_depth earlier, the improvement in performance from additional trees appears to level off. Therefore, I will set the number of trees (n_estimators) to 80, along with the previously determined max_depth of 20.

The random forest regression model exhibits a significantly lower sMAPE compared to linear regression, at 4.95%. Therefore, I will proceed with the random forest regression model. I will train the model using the complete training set and evaluate its performance on the test set.

model_forest = RandomForestRegressor(max_features=1.0, n_estimators=80, max_depth=20, random_state=0)
model_forest.fit(features_train, target_train)
pred = pd.DataFrame(model_forest.predict(features_test))

rougher_recov_actual = target_test['rougher.output.recovery']
final_recov_actual = target_test['final.output.recovery']
rougher_recov_pred = pred.iloc[:,0]
final_recov_pred = pred.iloc[:,1]      
        
#         Calculate sMAPE
rougher_smape = calc_smape(rougher_recov_actual, rougher_recov_pred)
final_smape = calc_smape(final_recov_actual, final_recov_pred)
combined_smape = combine_smapes(rougher_smape, final_smape)

print(f'Combined sMAPE: {round(combined_smape, 2)}%')

<div class="alert alert-success">
<b>Reviewer's comment</b>

The final model was evaluated using the test set

</div>

## Model Evaluation

The tested combined sMAPE is slightly higher than it was during cross-validation, but it remains below 10%, which appears acceptable given that Zyfra did not specify any maximum allowable sMAPE for a functional model. Although the random forest model takes longer to train compared to the linear regression model, the runtime is reasonable and not excessive.

Next, let's calculate the sMAPE for a dummy model and compare its error with our tested error and the cross-validated error.

for strategy in ['mean', 'median']:    
    dummy_regr = DummyRegressor(strategy=strategy)
    dummy_regr.fit(features_train, target_train)
    pred = pd.DataFrame(dummy_regr.predict(features_test))

    rougher_recov_actual = target_test['rougher.output.recovery']
    final_recov_actual = target_test['final.output.recovery']
    rougher_recov_pred = pred.iloc[:,0]
    final_recov_pred = pred.iloc[:,1]      

    #         Calculate sMAPE
    rougher_smape = calc_smape(rougher_recov_actual, rougher_recov_pred)
    final_smape = calc_smape(final_recov_actual, final_recov_pred)
    combined_smape = combine_smapes(rougher_smape, final_smape)

    print(f'Combined sMAPE with {strategy}: {round(combined_smape, 2)}%')

# Model Performance Comparison

The tested sMAPE of 7.53% is marginally lower than the dummy sMAPE of 7.67% when using the mean, but slightly higher than the dummy sMAPE of 7.23% when using the median. In contrast, the cross-validated sMAPE of 4.95% is notably lower than either dummy error.

Based on these comparisons, the model appears to perform as well as or better than simply predicting using the mean or median.

<div class="alert alert-success">
<b>Reviewer's comment</b>

Good idea to compare the model to a simple baseline

</div>

# Save the model to a joblib file for further use
dump(model_forest, 'ZyfraGoldRecoveryPredictor.joblib')

# Conclusion

Zyfra provided three datasets: one full dataset and pre-split training and test sets. The test set comprised only features available at the start of the extraction process, necessitating adjustments to the training set accordingly. Our objective was to construct a model predicting gold concentrate recovery at both the rougher and final stages of extraction, utilizing symmetric mean absolute percent error (sMAPE) for evaluation and deriving a weighted average of the two recoveries. It was essential for the provided recovery data to align with expectations, thus we verified the calculated recovery using a provided equation against the dataset values. Additionally, we examined metal concentrations at each extraction stage and filtered out nonsensical observations. We also assessed feed size distributions in both the training and test sets to ensure similarity. Further preprocessing involved handling missing values after trimming datasets to retain as many observations as possible.

We cross-validated linear regression and random forest regression models, identifying the random forest model with a sMAPE of 4.95% as the superior performer. Subsequently, we trained this model using consistent hyperparameters on the complete training set and tested it, yielding a sMAPE of 7.53%. While Zyfra did not specify sMAPE requirements, we compared our model against dummy models using mean/median predictions. Our tested model performed slightly worse than the dummy model using the median; however, the cross-validated sMAPE was distinctly lower than either dummy model's sMAPE, indicating our model should perform similarly or better than a dummy model. Finally, we saved the random forest model to a .joblib file for future use by the company.

<div class="alert alert-success">
<b>Reviewer's comment</b>

Nice summary!

</div>