"""
Feature selection multiclass

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
from sklearn.feature_selection import RFE
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import csv
import os
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from pathlib import Path


# ******************************************************************************
# Main
# ******************************************************************************

# Given arguments
parser = argparse.ArgumentParser(description="Feature selection")
parser.add_argument("--dataset_info", required=True) 
parser.add_argument("--data_path", required=True)
parser.add_argument("--test_indx", required=True) # test index to consider
parser.add_argument("--beam", required=True) # beam 
parser.add_argument("--n_features", required=True)
parser.add_argument("--random_state", required=True)
parser.add_argument("--shuffle", required=True)
parser.add_argument("--date", required=True)
parser.add_argument("--rfe_step", required=True)
parser.add_argument("--n_jobs", required=True)


args = parser.parse_args()

# Get the command-line arguments
dataset_info = args.dataset_info
data_path = args.data_path
test_indx = args.test_indx
beam = args.beam
n_features = int(args.n_features)
random_state = int(args.random_state)
shuffle = args.shuffle
date = args.date
rfe_step = args.rfe_step
n_jobs = int(args.n_jobs)

results_path = os.path.join(data_path, beam) 
rfe_step = float(rfe_step)


# Printing the test index being processed
print('test index: ', test_indx)
test_indx = int(test_indx)
print('rfe_step', rfe_step)

############## PREPARING THE DATASET FOR THE ANALYSIS ####################

# Reading the dataset containing the features about that experiment
dataset = pd.DataFrame()


n_add = 0
treatment_list = [] # list containing all the treatments

dataset_info = pd.read_csv(dataset_info, header=[0])


data_paths = []
for dset in dataset_info.datasets:
    full_path = os.path.join(dset, 'Results', beam, 'feature_df.csv')
    data_paths.append(full_path)

for d_path in data_paths:
    d_path = Path(d_path)
    df = pd.read_csv(d_path, header=[0])

    df_b = df[df.beam.isin([beam])]

        
    groups = np.unique(df_b.group_name)

    for t in groups:
        if treatment_list.count(t) > 0:
            new_t = t + '_' + str(n_add)
            n_add = n_add + 1
            
        else: 
            new_t = t
        treatment_list.append(new_t)
        df_b.loc[df_b.group_name == t, "group_name"] = new_t
        
    dataset = pd.concat([dataset, df_b], ignore_index=True)

# At this point I have a dataset with the right treatments and beam
classes = treatment_list

# Adding a column containing the groups defined based on the animal and the treatment
dataset = pd.get_dummies(dataset, columns=['sex'], drop_first=True)
dataset['Id'] = dataset.groupby(by=['animalID', 'group_name'], sort=False).ngroup().add(1)
complete_df = dataset.reset_index(drop=True)
conditions = np.unique(np.array(dataset.Id))

file_name = 'data_df_transformed.csv'
dataset.to_csv(os.path.join(data_path, file_name))

############## LEAVE-ONE-OUT CV WITH DIFFERENT VALIDATION SETS ###############

results_df = pd.DataFrame(columns=[
    'test_indx', 
    'validation_indx', 
    'test_videos', 
    'validation_videos', 
    'beam',
    'animal_group', 
    'accuracy',
    'n_selected_features',
    'selected_features',
    'feature_importance',
    'n_estimators',
    'min_sample_leaf',
    'max_features',
    'max_depth',
    'random_state',
    'bootstrap',
    'all_ranked_features',
    'all_features',
    'feature_elimination_ranking',
    'final_feature_ranking'])


# For every validation set
for index in conditions:
    temp_df = complete_df.copy()
    test_animal = temp_df.animalID[temp_df.Id == test_indx].values[0]
    print('test_animal', test_animal)
    
    if index != test_indx:
        print('validation_index=', index)
        val_animal_group = temp_df.group_name[temp_df.Id == index].values[0]
        print('animal_treatment', val_animal_group)
        val_animal = temp_df.animalID[temp_df.Id == index].values[0]
        print('val_animal and indices:', [temp_df.animalID[temp_df.Id == index]])
        index_to_remove = temp_df.index[temp_df.animalID.isin([val_animal, test_animal])]
        
        temp_df['classes'] = temp_df.groupby(by=['group_name'], sort=False).ngroup().add(1)
        y = temp_df.classes
        
        X = temp_df.drop(['group_name', 'animalID', 'age', 'beam','Id', 'ID', 'ref_distance_name', 'ref_distance_value', 'session', 'classes'], axis=1)
        
        test_i = temp_df.index[temp_df.Id == test_indx].tolist()
        val_i = temp_df.index[temp_df.Id == index].tolist()
        y_test = y.iloc[val_i].values # selecting the validation set 

        if shuffle == 'True':
            y_train = y.drop(index_to_remove.tolist()).reset_index(drop=True)
            np.random.shuffle(y_train)
        else:
            y_train = y.drop(index_to_remove.tolist()).reset_index(drop=True)

        X_test = X.iloc[val_i,:]
        X_train = X.drop(index_to_remove.tolist()).reset_index(drop=True) # Removing both the videos from the same animal on the validation set and the test set videos

        print('index to remove', index_to_remove.tolist())
        
        classifier = RandomForestClassifier(random_state=random_state, class_weight='balanced', n_jobs=n_jobs)

        s = StandardScaler()

        X_train_scaled = s.fit_transform(X_train)
        X_test_scaled = s.transform(X_test)

        X_test_df = pd.DataFrame(data=X_test_scaled, columns=X.columns)
        X_train_df = pd.DataFrame(data=X_train_scaled, columns=X.columns)


        # FIRST STEP OF RECURSIVE FEATURE ELIMINATION
        
        rfe_perc = int(100*rfe_step)
        rfe_method = RFE(estimator=classifier, n_features_to_select=n_features, step=rfe_step)

        rfe_method.fit(X_train_df, y_train)

        selected_features = X_train_df.columns[rfe_method.support_][:].tolist()
        print(selected_features) # Here I just take the first 50 selected features
        
        # HYPERPARAMETER TUNING USING THE FEATURES SELECTED BEFORE
        min_n_estimators = 10
        max_n_estimators = 200
        min_max_depth = 10
        max_max_depth = 110
        tune_estimators = True
        tune_depth = True

        while tune_estimators or tune_depth:
            n_estimators = [int(x) for x in np.linspace(start = min_n_estimators, stop = max_n_estimators, num = 5)]
            max_features = ['log2', 'sqrt', None]
            max_depth = [int(x) for x in np.linspace(start = min_max_depth, stop = max_max_depth, num = 5)]
            max_depth.append(None)
            min_samples_leaf = [1, 2, 4]
            bootstrap = [True, False]

            new_X_train = X_train_df[selected_features]
            new_X_test = X_test_df[selected_features]

            param_grid = {'n_estimators': n_estimators,
                        'max_features': max_features,
                        'max_depth': max_depth,
                        'min_samples_leaf': min_samples_leaf,
                        'bootstrap': bootstrap,
                        }

            grid = GridSearchCV(estimator=classifier, param_grid = param_grid, n_jobs = n_jobs)
            grid.fit(new_X_train, y_train)

            best_param = grid.best_params_
            
            # Proceding with the grid search of the max parameter is selected
            if best_param['n_estimators'] < max_n_estimators:
                tune_estimators = False
            else:
                min_n_estimators = max_n_estimators
                max_n_estimators = max_n_estimators + 200

            if best_param['max_depth']:
                if best_param['max_depth'] < max_max_depth:
                    tune_depth = False 
                else:
                    min_max_depth = max_max_depth
                    max_max_depth = max_max_depth + 100
            else:
                tune_depth = False

        print(best_param)
        
        # ADDITIONAL STEP OF RFE USING THE TUNED PARAMETERS
        classifier_2 = RandomForestClassifier(n_estimators=best_param['n_estimators'],  
            min_samples_leaf= best_param['min_samples_leaf'], 
            max_features= best_param['max_features'],
            max_depth= best_param['max_depth'],
            bootstrap= best_param['bootstrap'],
            random_state=random_state, class_weight='balanced', n_jobs=n_jobs)

        rfe_method_2 = RFE(estimator=classifier_2, n_features_to_select=n_features, step=rfe_step)

        rfe_method_2.fit(X_train_df, y_train)

        selected_features_2 = X_train_df.columns[rfe_method_2.support_][:].tolist()

        feature_ranks = rfe_method_2.ranking_
        all_features = X_train_df.columns.values

        feature_df = pd.DataFrame({'features':all_features, 'ranks':feature_ranks})
        # Feature importance for ranking the features

        ranks = np.unique(feature_ranks)
        print('ranks:', ranks)
        
        # Obtaining a ranked list of all the features based on both the rank assigned by RFE and feature importance
        ranked_features = []
        ranked_imp = []


    
        for rank in ranks:
            ranks_to_consider = np.array([int(x) for x in range(rank, rank+1)])
            temp_feat_df = feature_df[feature_df.ranks.isin(ranks_to_consider)]
            features = temp_feat_df.features

            new_classifier_2 = RandomForestClassifier(n_estimators=best_param['n_estimators'],  
                min_samples_leaf= best_param['min_samples_leaf'], 
                max_features= best_param['max_features'],
                max_depth= best_param['max_depth'],
                bootstrap= best_param['bootstrap'],
                random_state=random_state, class_weight='balanced', n_jobs=n_jobs)
            
            new_X_train_2 = X_train_df[features]
            new_X_test_2 = X_test_df[features]
            
            new_classifier_2.fit(new_X_train_2, y_train)
            feat_imp_2 = new_classifier_2.feature_importances_.tolist()
            select_df_2 = pd.DataFrame({'selected_feat':features, 'ranks':temp_feat_df.ranks, 'feat_imp':feat_imp_2})
            sorted_selected_df_2 = select_df_2.sort_values(by=['feat_imp'], ascending=False)
            ranked_features = ranked_features + sorted_selected_df_2.selected_feat[sorted_selected_df_2.ranks == rank].tolist()
            ranked_imp = ranked_imp + sorted_selected_df_2.feat_imp[sorted_selected_df_2.ranks == rank].tolist()
    
        ranked_feat_df = pd.DataFrame({'ranked_features':ranked_features, 'ranked_imp':ranked_imp})

        feat_rank = 0
        all_ranks = [0]
        for entry in range(1,ranked_feat_df.shape[0]):
            if ranked_feat_df.ranked_imp.iloc[entry] == ranked_feat_df.ranked_imp.iloc[entry - 1]:
                all_ranks.append(feat_rank)
            else:
                feat_rank = feat_rank + 1
                all_ranks.append(feat_rank)

        ranked_feat_df['feat_rank'] = all_ranks

        print('select_df_2 shape', select_df_2.shape)        
        # Saving the results into a dataframe
        if best_param['max_features'] == None:
            max_f = 'None'
        else:
            max_f = best_param['max_features']

        new_row = [
            test_indx,
            index,    
            test_i,
            val_i,
            beam,
            val_animal_group,
            grid.best_score_,
            n_features,
            ranked_features[:n_features],
            ranked_feat_df.ranked_imp.tolist(),
            int(best_param['n_estimators']),
            int(best_param['min_samples_leaf']),
            max_f,
            best_param['max_depth'],
            random_state,
            best_param['bootstrap'],
            ranked_features,
            ranked_feat_df.ranked_features.tolist(),
            feature_ranks.tolist(),
            all_ranks
        ]
        results_df.loc[len(results_df)] = new_row


if shuffle == 'True':
    
    file = 'results_' + beam + '_testIndex_' + str(test_indx) + '_' + date + '_' + str(rfe_perc) + '_multiclass_shuffle.csv'
else:
    
    file = 'results_' + beam + '_testIndex_' + str(test_indx) + '_' + date + '_' + str(rfe_perc) + '_multiclass.csv'

results_df.to_csv(os.path.join(data_path, file))
