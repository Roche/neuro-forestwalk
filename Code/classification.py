"""
Classification

Copyright 2024 F. Hoffmann-La Roche AG

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
   
"""





# Author: Francesca Tozzi <francesca.tozzi@roche.com>

import argparse
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import os
from scipy import stats
from pathlib import Path
from ast import literal_eval
import ranky as rk
from utils import auxiliaryFunctions as af

# ******************************************************************************
# Main
# ******************************************************************************

parser = argparse.ArgumentParser(description="Classification based on selected features")
parser.add_argument("--df_data", required=True)
parser.add_argument("--folder_path", required=True)
parser.add_argument("--results_path", required=True)
parser.add_argument("--group1", required=True)
parser.add_argument("--group2", required=True)
parser.add_argument("--beam", required=True)
parser.add_argument("--shuffle", required=True)
parser.add_argument("--date", required=True)
parser.add_argument("--n_features", required=True)
parser.add_argument("--data", required=True)
parser.add_argument("--n_jobs", required=True)



args = parser.parse_args()

# Get the command-line arguments
df_data = args.df_data
results_path = args.results_path
folder_path = args.folder_path
group1 = args.group1
group2 = args.group2
beam = args.beam
shuffle = args.shuffle
date = args.date
n_features = int(args.n_features)
data = args.data
n_jobs = int(args.n_jobs)
    
############################## PREPARING THE DATASET FOR THE ANALYSIS #######################################
# Loading the feature dataframe
results_p = Path(results_path)

if shuffle == 'True':
    data_files = [i.stem for i in results_p.glob('results_' + beam + '_testIndex_*' + '_' + group1 + '_' + group2 + '_' + date + '_shuffle.csv')]
else:
    data_files = [i.stem for i in results_p.glob('results_' + beam + '_testIndex_*' + '_' + group1 + '_' + group2 + '_' + date + '.csv')]

data = pd.read_csv(os.path.join(results_path, data))
beam_treat_df = data[(data.beam == beam) & (data.group_name.isin([group1,group2]))].copy()

# Making dummies of the sex category and removing not useful information
beam_treat_sex_df = pd.get_dummies(beam_treat_df, columns=['sex'], drop_first=True)

# Adding a column containing the groups defined based on the animal and the treatment
beam_treat_sex_df['Id'] = beam_treat_sex_df.groupby(by=['animalID', 'group_name'], sort=False).ngroup().add(1)
complete_df = beam_treat_sex_df.reset_index(drop=True)
groups = np.unique(complete_df.Id)

# complete_df.to_csv(os.path.join(results_p, 'çomp.csv'))


# Creating a unique dataframe containing the hyperparameters and selected features for all the test and validation sets
for (i,file) in enumerate(data_files):
    if i == 0:
        df = pd.read_csv(os.path.join(results_p,file + '.csv'), header=[0]).copy()
    else:
        df = pd.concat([df, pd.read_csv(os.path.join(results_p,file + '.csv'), header=[0])], ignore_index=True)

column_list = ['test_videos', 'validation_videos', 'selected_features', 'feature_importance', 'all_ranked_features', 'all_features', 'final_feature_ranking' ]
ripr_df = af.ripristinate_lists(df, column_list)

hyperparameters = ripr_df.groupby('test_indx')[['n_estimators', 'min_sample_leaf', 'max_depth', 'max_features', 'bootstrap', 'random_state']].agg(af.mode_with_none)

file = 'hyper.csv'
hyperparameters.to_csv(os.path.join(results_p, file))

prediction_results = pd.DataFrame(columns = ['test_set', 'beam', 'animal', 'group','predictions', 'accuracy', 'selected_feat', 'all_ranked_features', 'n_estimators', 'min_sample_leaf', 'max_depth', 'max_features', 'bootstrap', 'random_state'])

for test_set in groups:
    print('test_set:', test_set)
    temp_df = ripr_df[ripr_df.test_indx == test_set]
    # temp_df.to_csv('temp_df.csv')

    sorted_df, shared_sorted_list = af.get_ranks(temp_df) #Obtaining the rank of each feature using ranky and comparing different lists from different experiments

    animal_group = complete_df.group_name[complete_df.Id == test_set].values[0]

    selected_features = shared_sorted_list[:n_features]
    n_estimators = hyperparameters.loc[test_set, 'n_estimators']
    min_leaf = hyperparameters.loc[test_set, 'min_sample_leaf']
    if hyperparameters.loc[test_set, 'max_depth'] == 'None':
        max_depth = None
    else:
        max_depth = int(hyperparameters.loc[test_set, 'max_depth'])

    if hyperparameters.loc[test_set, 'max_features'] == 'None':
        max_features = None
    else:
        max_features = hyperparameters.loc[test_set, 'max_features']
    bootstrap = hyperparameters.loc[test_set, 'bootstrap']
    random_seed = hyperparameters.loc[test_set, 'random_state']

    rf = RandomForestClassifier(n_estimators=n_estimators, min_samples_leaf=min_leaf, max_depth=max_depth, max_features=max_features, bootstrap=bootstrap, random_state=random_seed, n_jobs=n_jobs)

    new_df = complete_df.copy() #Copyting the complete data df

    test_i = new_df.index[new_df.Id == test_set].tolist() # test set videos

    prov = pd.get_dummies(new_df, columns=['group_name'], drop_first=False)
    y = prov['group_name_' + group2]

    animal = new_df.animalID[new_df.Id == test_set].values[0] # Animal number
    index_to_remove = new_df.index[new_df.animalID == animal]

    test_df = new_df[new_df.Id == test_set]
    train_df = new_df.drop(index_to_remove).reset_index(drop=True)

    y_test = y.iloc[test_i].values

    if shuffle == 'True':
        y_train = y.drop(index_to_remove).reset_index(drop=True)
        np.random.shuffle(y_train)
    else:
        y_train = y.drop(index_to_remove).reset_index(drop=True)

    print(animal)
    print('index_to_remove', index_to_remove)
    s = StandardScaler()

    X_train = train_df[selected_features]
    X_test = test_df[selected_features] 

    X_train_scaled = s.fit_transform(X_train)
    X_test_scaled = s.transform(X_test)

    X_test_df = pd.DataFrame(data=X_test_scaled, columns=X_test.columns)
    X_train_df = pd.DataFrame(data=X_train_scaled, columns=X_train.columns)

    trained_rf = rf.fit(X_train_df, y_train)
    pred = list(trained_rf.predict(X_test_df))
    probs = trained_rf.predict_proba(X_test_df)[:,1]
    acc = accuracy_score(y_test, pred)

    row = [test_set, beam, animal, animal_group, [pred], acc, selected_features, shared_sorted_list, n_estimators, min_leaf, max_depth, max_features, bootstrap, random_seed]
    prediction_results.loc[len(prediction_results)] = row


# Finding the consensus list for this beam in the whole experiment
shared_ranked_df, shared_selected_features = af.get_ranks(prediction_results)

prediction_results['shared_ranked_features'] = [shared_selected_features]*prediction_results.shape[0]

if shuffle == 'True':
    file_name = 'prediction_results_' + beam + '_' + group1 + '_' + group2 + '_' + date +'_shuffle.csv'
else:
    file_name = 'prediction_results_' + beam + '_' + group1 + '_' + group2 +'_' + date + '.csv'     
prediction_results.to_csv(os.path.join(folder_path,file_name))


