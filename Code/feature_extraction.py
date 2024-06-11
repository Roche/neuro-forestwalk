"""
Feature extraction

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

import os
from pathlib import Path
import pandas as pd
import numpy as np
import argparse
import warnings

# Import custom modules
from utils import beam_walk as bw

# Set up argument parsing
parser = argparse.ArgumentParser(description='Process some parameters.')
parser.add_argument('--project_path', type=str, help='project path')
parser.add_argument('--beam_names', nargs='+', help='Names of the beams')

# Parse arguments
args = parser.parse_args()

# Now you can use the parameters in your script
print(f"The value of param1 is {args.project_path}")
print(f"The value of param2 is {args.beam_names}")

exp_path = args.project_path
beam_names = args.beam_names

print('beams:', beam_names)


extension = '.h5'
metadata_path = os.path.join(exp_path, 'metadata.csv')

p_exp = Path(exp_path)
p_data = os.path.join(exp_path, 'DLC_tracking/')

# Gather the list of data files with the specified extension and filtering criteria
data_files = [i.stem for i in p_exp.glob('DLC_tracking/*.h5')]

# Read the metadata from the CSV file
metadata = pd.read_csv(metadata_path, header=[0])

trial_list = []

for data_file in data_files:
    print('Analyzing file: ' + data_file)
    
    # Construct the full path to the data file
    data_path = os.path.join(p_data, data_file + extension)
    
    # Create a Trial object using the data path and metadata
    trial = bw.Trial(data_path, metadata, beam_names)
    
    # Append the Trial object to the trial list
    trial_list.append(trial)

    
    
    

experiment = bw.Experiment(trial_list)

feature_df = experiment.resultsDf

# Create the 'Results' subfolder if it doesn't exist
results_folder_path = os.path.join(exp_path, 'Results')
os.makedirs(results_folder_path, exist_ok=True)

# Iterate over the list of subfolder names
for subfolder_name in beam_names:
    # Create each subfolder inside the 'Results' folder
    subfolder_path = os.path.join(results_folder_path, subfolder_name)
    os.makedirs(subfolder_path, exist_ok=True)

    # Define the path for the CSV file
    csv_file_path = os.path.join(subfolder_path, 'feature_df.csv')

    # Save the DataFrame to a CSV file without the index
    feature_df.to_csv(csv_file_path, index=False)