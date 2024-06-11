"""
Feature selection

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
from sklearn.model_selection import GridSearchCV


# ******************************************************************************
# Main
# ******************************************************************************

# Given arguments
parser = argparse.ArgumentParser(description="Feature selection")
parser.add_argument("--df_data", required=True) # path to the dataframe containing the features
parser.add_argument("--folder_path", required=True) # folder where the results are stored
parser.add_argument("--test_indx", required=True) # test index to consider
parser.add_argument("--beam", required=True) # beam 
parser.add_argument("--group1", required=True)
parser.add_argument("--group2", required=True)
parser.add_argument("--n_features", required=True)
parser.add_argument("--random_state", required=True)
parser.add_argument("--shuffle", required=True)
parser.add_argument("--date", required=True)
parser.add_argument("--rfe_step", required=True)
parser.add_argument("--n_jobs", required=True)


args = parser.parse_args()

# Get the command-line arguments
df_data = args.df_data
test_indx = args.test_indx
folder_path = args.folder_path
beam = args.beam
group1 = args.group1
group2 = args.group2
n_features = int(args.n_features)
random_state = int(args.random_state)
shuffle = args.shuffle
date = args.date
rfe_step = args.rfe_step
n_jobs = int(args.n_jobs)

rfe_step = float(rfe_step)
rfe_perc = int(100*rfe_step)

test_indx_1 = int(test_indx)

############## PREPARING THE DATASET FOR THE ANALYSIS ####################

# Reading the dataset containing the features about that experiment
df = pd.read_csv(df_data, header=[0])

# Limiting the analysis to the beam and the treatments of interest
beam_treat_df = df[(df.beam == beam) & (df.group_name.isin([group1,group2]))].copy()

# Transforming the sex category into [0,1]
beam_treat_sex_df = pd.get_dummies(beam_treat_df, columns=['sex'], drop_first=True)

# Adding a column containing the groups defined based on the animal and the treatment
beam_treat_sex_df['Id'] = beam_treat_sex_df.groupby(by=['animalID', 'group_name'], sort=False).ngroup().add(1)
complete_df = beam_treat_sex_df.reset_index(drop=True)
groups = np.unique(np.array(complete_df.Id))

############## LEAVE-ONE-OUT CV WITH DIFFERENT VALIDATION SETS ###############

results_df = pd.DataFrame(columns=[
    'test_indx', 
    'validation_indx', 
    'test_videos', 
    'validation_videos', 
    'beam',
    'animal_treatment', 
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
for index in groups:
    temp_df = complete_df.copy()
    test_animal = temp_df.animalID[temp_df.Id == test_indx_1].values[0]
    
    if index != test_indx_1:
        
        animal_group = temp_df.group_name[temp_df.Id == index]

        test_i_1 = temp_df.index[temp_df.Id == test_indx_1].tolist() # index of the test set
        test_i_2 = temp_df.index[temp_df.Id == index].tolist() # index of the validation set
        
        # Removing all the videos belonging to the animal in the validation set
        print('val_animal and indices:', [animal_group, temp_df.animalID[temp_df.Id == index]])
        val_animal = temp_df.animalID[temp_df.Id == index].values[0]
        
        index_to_remove = temp_df.index[temp_df.animalID.isin([val_animal, test_animal])]
        
        # Creating the target dataframe
        prov = pd.get_dummies(temp_df, columns=['group_name'], drop_first=False)
        y = prov['group_name_' + group2]

        # Removing variables that will not be used for the classification
        X = temp_df.drop(['group_name', 'animalID', 'age', 'beam','Id', 'ID', 'ref_distance_name', 'ref_distance_value', 'session'], axis=1)

        y_test = y.iloc[test_i_2].values # selecting the validation set 
        
        if shuffle == 'True':
            y_train = y.drop(index_to_remove.tolist() + test_i_1).reset_index(drop=True)
            np.random.shuffle(y_train)
        else:
            y_train = y.drop(index_to_remove.tolist() + test_i_1).reset_index(drop=True)

        X_test = X.iloc[test_i_2,:]
        X_train = X.drop(index_to_remove.tolist()).reset_index(drop=True) # Removing both the videos from the same animal on the validation set and the test set videos

        print('index to remove', index_to_remove.tolist())
        
        classifier = RandomForestClassifier(random_state=random_state, class_weight='balanced', n_jobs=-1)

        s = StandardScaler()


        X_train_scaled = s.fit_transform(X_train)
        X_test_scaled = s.transform(X_test)

        X_test_df = pd.DataFrame(data=X_test_scaled, columns=X.columns)
        X_train_df = pd.DataFrame(data=X_train_scaled, columns=X.columns)



        # FIRST STEP OF RECURSIVE FEATURE ELIMINATION
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

            grid = GridSearchCV(estimator=classifier, param_grid = param_grid, n_jobs = -1)
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
            random_state=42, class_weight='balanced', n_jobs=-1)

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
                random_state=42, class_weight='balanced', n_jobs=15)
            
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

              
        # Saving the results into a dataframe
        if best_param['max_features'] == None:
            max_f = 'None'
        else:
            max_f = best_param['max_features']

        new_row = [
            test_indx_1,
            index,    
            test_i_1,
            test_i_2,
            beam,
            animal_group.values[0],
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
    file = 'results_' + beam + '_testIndex_' + str(test_indx) + '_' + group1 + '_' + group2 + '_' + date + '_shuffle.csv'
else:
    file = 'results_' + beam + '_testIndex_' + str(test_indx) + '_' + group1 + '_' + group2 +'_' + date +  '_' + str(rfe_perc) + '.csv'
results_df.to_csv(os.path.join(folder_path, file))
    