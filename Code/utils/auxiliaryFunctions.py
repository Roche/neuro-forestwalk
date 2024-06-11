from ast import literal_eval
import pandas as pd
import numpy as np
import ranky as rk
import os
from numpy import vstack, ones
from numpy.linalg import lstsq
import math

def ripristinate_lists(df, column_list):
    """
    Restore string representations of lists in specified columns of a dataframe back to list objects.

    Parameters:
    df (pandas.DataFrame): The dataframe containing string representations of lists.
    column_list (list of str): The column names in the dataframe that contain the string representations to be converted.

    Returns:
    pandas.DataFrame: The dataframe with the specified columns restored to list objects.

    Example:
    >>> df = pd.DataFrame({'col1': ["[1, 2, 3]", "[4, 5, 6]"]})
    >>> ripristinate_lists(df, ['col1'])
       col1
    0  [1, 2, 3]
    1  [4, 5, 6]
    """
    for column in column_list:
        df[column] = df[column].apply(literal_eval)
    return df

    
def get_ranks(temp_df):
    """
    Generate a consensus ranking list from a dataframe containing multiple ranked lists of features.

    Parameters:
    temp_df (pandas.DataFrame): A dataframe with a column 'all_ranked_features' that contains ranked lists of features.

    Returns:
    tuple: A tuple containing the sorted dataframe with average ranks and the list of features in consensus order.

    Example:
    >>> temp_df = pd.DataFrame({'all_ranked_features': [['b', 'a', 'c'], ['a', 'c', 'b']]})
    >>> sorted_df, consensus_list = get_ranks(temp_df)
    >>> consensus_list
    ['a', 'b', 'c']
    """

    list_of_lists = []
    f_df = pd.DataFrame(index = temp_df.all_ranked_features.iloc[0])
    for i in range(temp_df.shape[0]):
        list_of_lists.append(temp_df.all_ranked_features.iloc[i]) # creating a list containing all the lists of sorted features
    
    for feature in f_df.index:
        for l in range(len(list_of_lists)):
            f_df.loc[feature, 'list_' + str(l)] = list_of_lists[l].index(feature) # creating a dataframe in which each feature is an index and each entry is the position of that feature in the list

    f_df['ranks'] = rk.borda(f_df)
    sorted_df = f_df.sort_values('ranks', ascending=False)
    
    return sorted_df, sorted_df.index.values.tolist()

def get_final_ranks(temp_df):
    """
    Calculate a consensus ranking of features from multiple ranked lists.

    This function aggregates multiple ranked lists of features into a single consensus list. 
    Each feature's final rank is determined by the average of its positions across all provided 
    ranked lists. The consensus list is sorted based on these average ranks, from the highest 
    (most important) to the lowest (least important).

    Parameters:
    temp_df (pd.DataFrame): A DataFrame where each row contains a ranked list of features 
                            in the 'all_ranked_features' column.

    Returns:
    tuple: A tuple containing two elements:
           - sorted_df (pd.DataFrame): A DataFrame with features as the index, their ranks 
                                       in each list as columns, and their average rank in 
                                       the 'ranks' column, sorted by the average rank.
           - list (list): A list of features ordered by their average rank, from highest 
                          to lowest.

    Note:
    The function assumes that the input DataFrame 'temp_df' has a specific structure, with 
    an 'all_ranked_features' column containing the ranked lists. The function uses the Borda 
    count method for rank aggregation.

    Example:
    >>> temp_df = pd.DataFrame({'all_ranked_features': [['feat1', 'feat2', 'feat3'], 
                                                         ['feat2', 'feat3', 'feat1']]})
    >>> sorted_df, consensus_list = get_final_ranks(temp_df)
    >>> sorted_df
           list_0  list_1  ranks
    feat2       1       0    1.5
    feat3       2       1    2.5
    feat1       0       2    2.5
    >>> consensus_list
    ['feat2', 'feat3', 'feat1']
    """
    list_of_lists = []
    f_df = pd.DataFrame(index = temp_df.all_ranked_features.iloc[0])
    for i in range(temp_df.shape[0]):
        list_of_lists.append(temp_df.all_ranked_features.iloc[i]) # creating a list containing all the lists of sorted features
    
    for feature in f_df.index:
        for l in range(len(list_of_lists)):
            f_df.loc[feature, 'list_' + str(l)] = list_of_lists[l].index(feature) # creating a dataframe in which each feature is an index and each entry is the position of that feature in the list

    f_df['ranks'] = rk.borda(f_df)
    sorted_df = f_df.sort_values('ranks', ascending=False)
    
    return sorted_df, sorted_df.index.values.tolist()

def mode_with_none(x):
    """
    Calculate the mode of a series, returning None if the series is empty.

    Parameters:
    x (pandas.Series): The series for which to calculate the mode.

    Returns:
    The mode of the series or None if the series is empty.

    Example:
    >>> mode_with_none(pd.Series([1, 1, 2, 2, 3]))
    1
    >>> mode_with_none(pd.Series([]))
    None
    """

    mode_series = pd.Series.mode(x)
    if mode_series.empty:
        return None
    else:
        return mode_series[0]

def DLCandTarget(folder_path:str, file_name:str):
    """
    Reads the DeepLabCut output and corresponding target data from CSV files.

    Parameters:
    folder_path (str): The path to the folder containing the CSV files.
    file_name (str): The base name of the CSV files to read (without '_filtered' or '_labels').

    Returns:
    dlc_df (pd.DataFrame): DataFrame containing the DeepLabCut coordinates.
    target_df (pd.DataFrame): DataFrame containing the target footslip labels.
    """
    extension = '.csv'
    dlc_path = os.path.join(folder_path, file_name + extension)
    dlc_df = pd.read_csv(
    dlc_path, 
    header=[1,2], 
    dtype =np.float64, 
    index_col=0)
    
    new_name = file_name.replace('_filtered', '')
    print(new_name)
    target_path = os.path.join(folder_path + new_name + '_labels' + extension)
    prov_df = pd.read_csv(
    target_path, 
    header=[0], 
    dtype =np.float64, 
    index_col=0)
    target_df = prov_df[['footslips']]
    
    return dlc_df, target_df


def calculateBeamCoord(dlc_df:pd.DataFrame):
    """
    Calculates the slope and intercept of the line passing through the left and right upper coordinates of the beam.

    Parameters:
    dlc_df (pd.DataFrame): DataFrame containing the DeepLabCut coordinates.

    Returns:
    m (float): The slope of the line.
    c (float): The y-intercept of the line.
    """
    p1 = (dlc_df[('beamLeftUp', 'x')].iloc[0], dlc_df[('beamLeftUp', 'y')].iloc[0])
    p2 = (dlc_df[('beamRightUp', 'x')].iloc[0], dlc_df[('beamRightUp', 'y')].iloc[0])

    points = [p1,p2]
    x_coords, y_coords = zip(*points)
    A = vstack([x_coords,ones(len(x_coords))]).T
    m, c = lstsq(A, y_coords, rcond=None)[0]
    
    return m, c

def removeBeamCoord(dlc_df:pd.DataFrame, m:float, c:float):
    """
    Adjusts the y-coordinates of body parts by subtracting the beam height at the corresponding x-coordinate.

    Parameters:
    dlc_df (pd.DataFrame): DataFrame containing the DeepLabCut coordinates.
    m (float): The slope of the beam line.
    c (float): The y-intercept of the beam line.

    Returns:
    prov_dlc_df (pd.DataFrame): DataFrame with adjusted y-coordinates.
    """
    bodyparts = dlc_df.columns.get_level_values('bodyparts').unique()
    
    prov_dlc_df = dlc_df
    for i in range(dlc_df.shape[0]):
        for bp in bodyparts:
            if bp != 'target':
                beam_y = m*dlc_df[(bp, 'x')].iloc[i] + c
                prov_dlc_df.loc[i, (bp, 'y')] = dlc_df.loc[i, (bp, 'y')] - beam_y
    
    return prov_dlc_df


def removeLowLikelihood(dlc_df:pd.DataFrame):
    """
    Removes frames where the likelihood of the hind paw detection is below a threshold.

    Parameters:
    dlc_df (pd.DataFrame): DataFrame containing the DeepLabCut coordinates with likelihood values.

    Returns:
    dlc_df (pd.DataFrame): DataFrame with low likelihood frames removed.
    """
    index_names = dlc_df[dlc_df[('hindPaw', 'likelihood')] < 0.5 ].index
    dlc_df.drop(index_names, inplace = True)
    dlc_df.reset_index(inplace=True, drop=True)
    
    return dlc_df


def analyzeFootSlips(df:pd.DataFrame):
    """
    Analyzes foot slips by identifying transitions from no slip to slip.

    Parameters:
    df (pd.DataFrame): DataFrame containing binary footslip labels.

    Returns:
    single_fs_n (int): The number of individual foot slips detected.
    single_fs_idx (list): The indices of frames where individual foot slips occur.
    """
    single_fs_n = 0
    single_fs_idx = []
    for j in range(1,len(df)):
        if df[j] == 1:
            if df[j-1] == 0:
                # saving the index of the foot slip
                single_fs_idx.append(j)
                single_fs_n = single_fs_n + 1

    return single_fs_n, single_fs_idx

class Trial():

    def __init__(self,
            videoPath:str,
            dlc_df:pd.DataFrame,
            fps:int,
            ageInWeeks:int=6     
            ):

            self.videoPath = videoPath
            self.fps = fps
            self.path_fileName = os.path.split(self.videoPath)
            self.items = self.path_fileName[1].split('_')
            self.date= self.items[0]
            self.mouseID = self.items[1]
            
            self.sex = 0
        
            self.treatment = 0
            self.session = 0
            
            self.cmToPix = 0
            self.weight = 0
            self.nose_start_frame=0
            self.nose_end_frame=0
            self.footslipsIdx=[0]
            self.f_start_frame=0
            self.f_end_frame=0
            self.x_start=0
            self.x_end=0
            self.dlc_df = dlc_df
            self.beam = self._extractBeam()
            self.norm_dlc_df = self._normalizeDlc()
            # self.peaksDf = self._extractPeaks()
            
            
            self.timeToCross = self._getTimeToCross()
            self.footslips, self.pred = self._getFootslips()
            #self.feature_df = self.extract_features()
    
    def _normalizeDlc(self):

        
        bodyparts = self.dlc_df.columns.get_level_values('bodyparts').unique()
        norm_dlc_df = self.dlc_df.copy()
        m,c = self._calculateBeamCoord()
        
        for bp in bodyparts:
            beam_y = m*self.dlc_df[(bp, 'x')] + c
            norm_dlc_df[(bp, 'y')] = self.dlc_df[(bp, 'y')] - beam_y
        
        return norm_dlc_df
        
    def _calculateBeamCoord(self):

        #print('calculating beam coord')
        p1 = (self.dlc_df[('beamLeftUp', 'x')].iloc[0], self.dlc_df[('beamLeftUp', 'y')].iloc[0])
        p2 = (self.dlc_df[('beamRightUp', 'x')].iloc[0], self.dlc_df[('beamRightUp', 'y')].iloc[0])

        points = [p1,p2]
        x_coords, y_coords = zip(*points)
        A = np.vstack([x_coords,np.ones(len(x_coords))]).T
        m, c = lstsq(A, y_coords)[0]
        
        self.cmToPix = 80/math.dist(p1,p2)
        return m, c

    def _getTimeToCross(self, frameNumber:int=700):
        """
        """

        #print('getting time to cross')
        x_start= self.norm_dlc_df[('beamLeftUp', 'x')].iloc[0] #start of the 80cm region
        x_end= self.norm_dlc_df[('beamRightUp', 'x')].iloc[0]#end of the 80cm region
        fps = 120 

        nose_start_frame=0
        nose_end_frame=0

        j = 0

        while nose_start_frame == 0 or nose_end_frame==0 and j < self.norm_dlc_df.shape[0]:

            if (nose_start_frame==0 and self.norm_dlc_df[('nose', 'x')].iloc[j] > x_start and self.norm_dlc_df[('nose', 'likelihood')].iloc[j]>0.95):
                nose_start_frame=j
            if (nose_end_frame==0 and self.norm_dlc_df[('nose', 'x')].iloc[j] > x_end and self.norm_dlc_df[('nose', 'likelihood')].iloc[j]>0.95):
                nose_end_frame=j
            j = j + 1

        if nose_end_frame==0:
            nose_end_frame=self.norm_dlc_df.shape[0]-1

        time_to_cross = (nose_end_frame - nose_start_frame)/fps

        self.nose_start_frame= nose_start_frame
        self.nose_end_frame = nose_end_frame
        self.x_start=x_start
        self.x_end=x_end

        return time_to_cross

    def _getFootslips(self, v_threshold:float = 18, h_threshold:float = 32, x_start:int=0, x_end:int=0):
        
        #print('getting foot slips')
        x_start= self.norm_dlc_df[('beamLeftUp', 'x')].iloc[0] #start of the 80cm region
        x_end= self.norm_dlc_df[('beamRightUp', 'x')].iloc[0]#end of the 80cm region
        
        fps = 120

        f_start_frame=0
        f_end_frame=0
        j = 0

        while f_start_frame == 0 or f_end_frame==0 and j < self.norm_dlc_df.shape[0]:
            if (f_start_frame==0 and self.norm_dlc_df[('hindPaw', 'x')].iloc[j] > x_start and self.norm_dlc_df[('hindPaw', 'likelihood')].iloc[j] > 0.95):
                f_start_frame=j
            if (f_end_frame==0 and self.norm_dlc_df[('hindPaw', 'x')].iloc[j] > x_end and self.norm_dlc_df[('hindPaw', 'likelihood')].iloc[j] > 0.95):
                f_end_frame=j
            
            j = j + 1
        
        if f_end_frame==0:
            f_end_frame=self.dlc_df.shape[0]-1
        if f_start_frame==0:
            f_start_frame= self.dlc_df.shape[0]-1

        y_dlc_df = self.norm_dlc_df[[('hindPaw', 'y')]]
        prediction = y_dlc_df['hindPaw']>v_threshold
        prediction = prediction*1
        for i in range(prediction.shape[0]-h_threshold):
            if (prediction['y'].iloc[i] == 1 and prediction['y'].iloc[i+1] == 0):
                if sum(prediction['y'].iloc[i+1:i+h_threshold] > 0):
                        for j in range(h_threshold):
                            prediction['y'].iloc[i+j] = 1
    
        
        nFootslips = 0
        for n in range(f_start_frame, self.nose_end_frame):
            if n > f_start_frame:
                if prediction['y'].iloc[n]:
                    if prediction['y'].iloc[n-1] == False:
                        nFootslips = nFootslips + 1
        
        self.f_start_frame=f_start_frame
        self.f_end_frame=f_end_frame
        self.prediction = prediction['y']

        return nFootslips, prediction['y']
    
    def _extractBeam(self):
    
        #print('extracting beam coord')
        
        if self.dlc_df[('dot', 'likelihood')].iloc[10] > 0.95:
            
            if (abs(self.dlc_df[('dot', 'x')].iloc[10] - self.dlc_df[('beamLeftUp', 'x')].iloc[10]) < abs(self.dlc_df[('dot', 'x')].iloc[10] - self.dlc_df[('beamRightUp', 'x')].iloc[10])):
                beam = 'narrowSquare'
            else:
                beam = 'wideSquare'
        else:
            beam = 'round'
        
        return beam