"""
ForestWalk functions

"""

# Author: Francesca Tozzi <francesca.tozzi@roche.com>


import os
import pandas as pd
import numpy as np
import math
import itertools
from sklearn.metrics import f1_score
import dlc2kinematics
import warnings
from scipy.linalg import lstsq
import os


def filter_low_prob(cols, prob):
    mask = cols.iloc[:, 2] <= prob
    cols.iloc[mask, :2] = np.nan
    return cols

class Trial():

    def __init__(self,
            data_path:str,
            metadata:pd.DataFrame,
            beam_names: pd.Series,
            fps:int = 120, # in frame
            v_threshold:int = 19, # in pixel
            h_threshold:int = 32, # in frame
            beam_length:int = 80, # in cm
            pcutoff:float = 0.95, # in probability
            min_trackable_nframe:int = 10, # in frame
            ref_distance:str='elbow_shoulder_mean_dst'
            ):
        
            self.data_path = data_path
            self.path_fileName = os.path.split(self.data_path)
            self.metadata = metadata
            self.beam_names = beam_names
            self.fps = fps
            self.v_threshold = v_threshold
            self.h_threshold = h_threshold
            self.beam_length = beam_length
            self.pcutoff = pcutoff
            self.min_trackable_nframe = min_trackable_nframe
            self.ref_distance_name = ref_distance
            
            self.nameItems = self.path_fileName[1].split('_')
            self.date= self.nameItems[0]
            self.animalID = self.nameItems[1]
            self.session = self.nameItems[2]
            self.sex = self.metadata.Sex[self.metadata.Animal_ID == int(self.animalID)].values[0]

            self.file_pattern = self.nameItems[0] + '_' + self.nameItems[1]
    
            self.group = self.metadata.Group_name[self.metadata.File_pattern == self.file_pattern].values[0]
            self.age = self.metadata.Age[self.metadata.File_pattern == self.file_pattern].values[0]
            self.weight = self.metadata.Weight[self.metadata.File_pattern == self.file_pattern].values[0]
            
            
            
            # Initializing variables
            self.nose_start_frame=0
            self.nose_end_frame=0
            self.footslipsIdx=[]
            self.foot_start_frame=0
            self.foot_end_frame=0
            self.x_start=0
            self.x_end=0
            self.ref_distance = 0
            
            # Extracting data
            self.dlc_df, self.bodyparts, self.scorer = dlc2kinematics.load_data(self.data_path)
            self.beam = self._extractBeam(self.dlc_df)
            self.norm_dlc_df = self._normalizeDlc()
            self.timeToCross = self._getTimeToCross()
            self.footslips, self.pred = self._getFootslips()
            
            
    
    def _normalizeDlc(self):
        """
        Normalize the y-coordinates of body parts in the DLC tracking data relative to the beam.

        This method adjusts the y-coordinates of each body part so that they are expressed
        relative to the beam's position, which is calculated using the `_calculateBeamCoord` method.
        The beam is assumed to be the reference object in the tracking data.

        Returns:
            pandas.DataFrame: A DataFrame with the same structure as `dlc_df`, but with the y-coordinates
                              of the body parts normalized relative to the beam's position.

        Note:
            This method is intended to be used internally within the class, hence the leading underscore.
            It assumes that `self.dlc_df` is a pandas DataFrame containing the tracking data with a
            MultiIndex (scorer, bodypart, coords) and that `self.bodyparts` is a list of body parts to normalize.
            The method `_calculateBeamCoord` is expected to return the slope (m) and y-intercept (c) of the beam.
        """

        norm_dlc_df = self.dlc_df.copy()
        m,c, cmToPix = self._calculateBeamCoord()
        
        for bp in self.bodyparts:
            beam_y = m*self.dlc_df[(self.scorer, bp, 'x')] + c
            norm_dlc_df[(self.scorer, bp, 'y')] = self.dlc_df[(self.scorer, bp, 'y')] - beam_y
        
        return norm_dlc_df
        
        
    def _calculateBeamCoord(self):
        """
        Calculate the slope and y-intercept of the beam based on its coordinates in the DLC tracking data.

        This method identifies two points on the beam ('beamLeftUp' and 'beamRightUp') and uses their
        coordinates to compute the parameters of the linear equation (y = mx + c) representing the beam.
        It also calculates the conversion factor from centimeters to pixels based on the distance between
        these two points, assuming the beam is 80 centimeters long.

        Returns:
            tuple: A tuple containing the slope (m) and y-intercept (c) of the beam's linear equation.

        Note:
            This method is intended to be used internally within the class, hence the leading underscore.
            It assumes that `self.dlc_df` is a pandas DataFrame containing the tracking data with a
            MultiIndex (scorer, bodypart, coords). The method updates the `self.cmToPix` attribute with
            the calculated centimeters to pixels conversion factor.
        """
        
        # Extract the coordinates of the left and right upper points of the beam
        p1 = (self.dlc_df[(self.scorer, 'beamLeftUp', 'x')].iloc[0], self.dlc_df[(self.scorer,'beamLeftUp', 'y')].iloc[0])
        p2 = (self.dlc_df[(self.scorer, 'beamRightUp', 'x')].iloc[0], self.dlc_df[(self.scorer,'beamRightUp', 'y')].iloc[0])
        
        # Combine the points into a list and separate the coordinates
        points = [p1,p2]
        x_coords, y_coords = zip(*points)
        
        # Prepare the matrix A for linear equation solving
        A = np.vstack([x_coords,np.ones(len(x_coords))]).T
        
         # Solve the linear equation to find the slope (m) and y-intercept (c)
        m, c = lstsq(A, y_coords)[0]
        
        # Calculate the conversion factor from centimeters to pixels
        cmToPix = self.beam_length/math.dist(p1,p2) # 80 cm region
        
        return m, c, cmToPix

    def _extractBeam(self, dlc_df:pd.DataFrame):
        """
        Determine the type of beam used in the experiment by analyzing the position of a 'dot' marker.

        This method checks the likelihood of a 'dot' marker being correctly identified in the DLC tracking
        data. If the likelihood is high, it then compares the distances of the 'dot' from predefined points
        on the beam ('beamLeftUp' and 'beamRightUp') to determine the type of beam. The beam type is
        selected from a list of predefined beam names based on the position of the 'dot'.

        Args:
            dlc_df (pd.DataFrame): The DataFrame containing the DLC tracking data.

        Returns:
            str: The name of the beam type identified from the predefined list of beam names.

        Raises:
            ValueError: If the number of predefined beam names is less than 3.

        Note:
            The method assumes that `self.beam_names` is a list containing at least three elements,
            which correspond to the different types of beams that can be identified. The 'dot' marker
            is used to differentiate between the beam types, and a likelihood threshold of 0.95 is used
            to ensure accurate identification. If the likelihood is below the threshold, the method
            defaults to the second beam name in the list.
        """
        
        beams = self.beam_names
        
        # Check if the number of predefined beam names is sufficient
        if len(beams) < 3:
            print('Insufficient number of beam names provided')
        
        
        # If there are enough beam names, proceed with the analysis
        else:
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                dlc_df = self.dlc_df.groupby("bodyparts", axis=1, group_keys=False).apply(filter_low_prob, prob=self.pcutoff)
                beam_df = dlc_df[(self.scorer)][['beamLeftUp', 'beamRightUp', 'dot']].dropna()
                
                # Check if there are enough frames with trackable 'dot' marker
                if beam_df.shape[0] > self.min_trackable_nframe:
                    
                    # Calculate the distance of the 'dot' marker from 'beamLeftUp' on the x-axis
                    dist_dot_beamLeftUp = abs(beam_df[('dot', 'x')].iloc[0] - beam_df[('beamLeftUp', 'x')].iloc[0])
                    
                    # Calculate the distance of the 'dot' marker from 'beamRightUp' on the x-axis
                    dist_dot_beamRightUp = abs(beam_df[('dot', 'x')].iloc[0] - beam_df[('beamRightUp', 'x')].iloc[0])
                    
                    # Determine the beam type based on the position of the 'dot' marker
                    if dist_dot_beamLeftUp < dist_dot_beamRightUp:
                        beam = beams[2]
                    else:
                        beam = beams[0]
                else:
                    beam = beams[1]
        
        return beam
    
    
    def _getTimeToCross(self, frameNumber:int=700):
        """
        Calculate the time taken for the mouse's nose to cross an 80 cm region of the beam.

        This method determines the start and end frames where the mouse's nose crosses the defined
        start and end points of the beam region. It uses a likelihood threshold to ensure the nose
        is accurately tracked. The time to cross is calculated by dividing the frame difference by
        the frame rate (fps). If the nose does not reach the end point, the last frame is used.
        The time to cross is capped at 60 seconds.

        Args:
            frameNumber (int, optional): The frame number to start the analysis from. Defaults to 700.

        Returns:
            float: The time in seconds it takes for the mouse's nose to cross the beam region.

        Note:
            This method assumes that `self.norm_dlc_df` is a normalized pandas DataFrame containing
            the tracking data with a MultiIndex (scorer, bodypart, coords). It also updates the
            instance attributes `self.nose_start_frame`, `self.nose_end_frame`, `self.x_start`, and
            `self.x_end` with the calculated values.
        """
        
        # Define the start and end x-coordinates of the 80 cm region
        beamLeft_x= self.norm_dlc_df[(self.scorer,'beamLeftUp', 'x')].iloc[0] 
        beamRight_x= self.norm_dlc_df[(self.scorer,'beamRightUp', 'x')].iloc[0] 
        
        # Initialize the start and end frames for the nose crossing
        nose_start_frame=0
        nose_end_frame=0
        
        # Extract the 'x' positions and likelihoods for the nose from the DataFrame
        nose_x_positions = self.norm_dlc_df[(self.scorer, 'nose', 'x')]
        nose_likelihoods = self.norm_dlc_df[(self.scorer, 'nose', 'likelihood')]

        # Create boolean Series for start and end crossing conditions
        start_crossed = (nose_x_positions > beamLeft_x) & (nose_likelihoods > self.pcutoff)
        end_crossed = (nose_x_positions > beamRight_x) & (nose_likelihoods > self.pcutoff)

        # Find the first frame where the nose crosses the start point with high likelihood
        nose_start_frame = start_crossed.idxmax() if start_crossed.any() else 0

        # Find the first frame where the nose crosses the end point with high likelihood
        nose_end_frame = end_crossed.idxmax() if end_crossed.any() else 0

        # If no crossing is found for the end point, use the last frame as the end frame
        if nose_end_frame == 0:
            nose_end_frame = self.norm_dlc_df.shape[0] - 1

        time_to_cross = (nose_end_frame - nose_start_frame)/self.fps

        # Update instance attributes with the calculated values
        self.nose_start_frame = nose_start_frame
        self.nose_end_frame = nose_end_frame
        self.x_start = beamLeft_x
        self.x_end = beamRight_x
        
        # Cap the time to cross at 60 seconds
        if time_to_cross > 60:
            time_to_cross = 60

        return time_to_cross

    def _getFootslips(self, v_threshold:float = 18, h_threshold:float = 32, x_start:int=0, x_end:int=0):
        """
        Calculate the number of footslips by the mouse on the beam.

        This method analyzes the normalized DLC tracking data to identify footslips based on a vertical
        threshold and a horizontal threshold. A footslip is counted when the hind paw drops below the
        vertical threshold and does not return above it within the horizontal distance.

        Args:
            v_threshold (float, optional): The vertical threshold for detecting footslips. Defaults to 18.
            h_threshold (float, optional): The horizontal distance to check for recovery above the threshold. Defaults to 32.
            x_start (int, optional): The x-coordinate to start analyzing for footslips. Defaults to 0.
            x_end (int, optional): The x-coordinate to stop analyzing for footslips. Defaults to 0.

        Returns:
            tuple: A tuple containing the number of footslips and a pandas Series with the prediction of footslips.

        Note:
            This method assumes that `self.norm_dlc_df` is a normalized pandas DataFrame containing
            the tracking data with a MultiIndex (scorer, bodypart, coords). It updates the instance
            attributes `self.f_start_frame`, `self.f_end_frame`, and `self.prediction` with the calculated
            values. The method uses the 'hindPaw' bodypart for detecting footslips.
        """
        # Define the start and end x-coordinates of the 80 cm region
        x_start= self.x_start
        x_end= self.x_end
        fps = self.fps
        h_threshold = self.h_threshold
        v_threshold = self.v_threshold
        
        # Extract the 'x' positions and likelihoods for the hind paw from the DataFrame
        hind_paw_x_positions = self.norm_dlc_df[(self.scorer, 'hindPaw', 'x')]
        hind_paw_likelihoods = self.norm_dlc_df[(self.scorer, 'hindPaw', 'likelihood')]

        # Create boolean Series for start and end crossing conditions
        start_crossed = (hind_paw_x_positions > x_start) & (hind_paw_likelihoods > self.pcutoff)
        end_crossed = (hind_paw_x_positions > x_end) & (hind_paw_likelihoods > self.pcutoff)

        # Find the first frame where the hind paw crosses the start point with high likelihood
        foot_start_frame = start_crossed.idxmax() if start_crossed.any() else self.norm_dlc_df.shape[0] - 1

        # Find the first frame where the hind paw crosses the end point with high likelihood
        foot_end_frame = end_crossed.idxmax() if end_crossed.any() else self.norm_dlc_df.shape[0] - 1

        # If no crossing is found for the start point, use the last frame as the start frame
        if foot_start_frame == 0 and not start_crossed.iloc[0]:
            foot_start_frame = self.norm_dlc_df.shape[0] - 1

        # If no crossing is found for the end point, use the last frame as the end frame
        if foot_end_frame == 0 and not end_crossed.iloc[0]:
            foot_end_frame = self.norm_dlc_df.shape[0] - 1

        
        # Extract the y-coordinates of the hind paw
        y_dlc_df = self.norm_dlc_df[[(self.scorer,'hindPaw', 'y')]]
        
        # Create a prediction series based on the vertical threshold
        prediction = y_dlc_df[(self.scorer,'hindPaw')]>v_threshold
        prediction = prediction.astype(int) # Convert boolean to 0/1
        
        # Check for recovery above the threshold within the horizontal distance
        for i in range(prediction.shape[0]-h_threshold):
            if (prediction['y'].iloc[i] == 1 and prediction['y'].iloc[i+1] == 0):
                if sum(prediction['y'].iloc[i+1:i+h_threshold] > 0):
                        for j in range(h_threshold):
                            prediction['y'].iloc[i+j] = 1
    
        
        # Count the number of footslips
        nFootslips = 0
        footslipsIdx = []
        for n in range(foot_start_frame, self.nose_end_frame):
            if n > foot_start_frame:
                if prediction['y'].iloc[n]:
                    if prediction['y'].iloc[n-1] == False:
                        nFootslips = nFootslips + 1
                        footslipsIdx.append(n)
        
        # Update instance attributes with the calculated values
        self.foot_start_frame=foot_start_frame
        self.foot_end_frame=foot_end_frame
        self.prediction = prediction['y']
        self.footslipsIdx = footslipsIdx

        return nFootslips, prediction['y']
    
    
    def addrow(self, df, ID, feature, value):
        """
        Add a new row to a DataFrame with specified ID, feature, and value.

        This method appends a new row to the provided DataFrame with the given ID, feature name,
        and corresponding value. The new row is added at the end of the DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame to which the new row will be added.
            ID (str): The identifier associated with the new row, typically representing an entity or subject.
            feature (str): The name of the feature or attribute for the new row.
            value (various types): The value corresponding to the feature for the new row.

        Returns:
            None: The DataFrame is modified in place, and no value is returned.

        Note:
            The method assumes that the DataFrame `df` has a structure compatible with the new row
            being added, and that the DataFrame is indexed in a way that allows for appending rows
            by using the length of the DataFrame as the next index.
        """
        new_row = [ID, feature, value]
        df.loc[len(df)] = new_row
    
    
    def _getDistances(self, df:pd.DataFrame, combinations:list, ref_distance:float):
        """
        Calculate and store normalized distances between specified body part combinations.

        This method iterates over a list of body part combinations, calculates the distances between
        each pair, normalizes these distances by a reference distance, and then computes statistical
        measures (mean, minimum, maximum, variance) for these normalized distances. The results are
        stored in a temporary DataFrame with columns for the ID, feature name, and feature value.

        Args:
            df (pd.DataFrame): The DataFrame containing the normalized DLC tracking data.
            combinations (list): A list of tuples, each containing a pair of body parts for which
                                 the distance will be calculated.
            ref_distance (float): A reference distance used to normalize the calculated distances.

        Returns:
            pd.DataFrame: A DataFrame containing the calculated mean, minimum, maximum, and variance
                          of the normalized distances for each body part combination.

        Note:
            The method assumes that `df` contains the tracking data with a MultiIndex (scorer, bodypart, coords).
            Only data points with a likelihood greater than 0.95 are considered for both body parts in each
            combination. The method uses `self._calculateDistance` to compute the distances and `self.addrow`
            to append the statistical measures to the temporary DataFrame. The ID used in `self.addrow` is
            assumed to be obtained from `self.path_fileName[1]`.
        """
        
        
        temp_df = pd.DataFrame(columns = ['ID', 'feature', 'value'])

        for (n,c) in enumerate(combinations):
            
            bp1 = combinations[n][0]
            bp2 = combinations[n][1]
            
            # Filter the DataFrame for rows where both body parts have high likelihood
            correct_df = df[(df[(self.scorer,bp1, 'likelihood')] > self.pcutoff) & df[(self.scorer,bp2, 'likelihood')] > self.pcutoff]
            correct_df = correct_df.dropna()
            
            # Calculate the distance between the body parts
            dist = self._calculateDistance(correct_df, bp1, bp2)
            
            if len(dist) > 0:
                
                # Normalize the distances
                norm_dist = dist/ref_distance
                self.norm_dist = norm_dist

                # Calculate statistical measures for the normalized distances
                mean_dst = np.mean(norm_dist)
                mean_dst_name = bp1+'_'+bp2+'_' + 'mean_dst'
                min_dst = np.min(norm_dist)
                min_dst_name = bp1+'_'+bp2+'_' + 'min_dst'
                max_dst = np.max(norm_dist)
                max_dst_name = bp1+'_'+bp2+'_' + 'max_dst'
                var_dst = np.var(norm_dist)
                var_dst_name = bp1+'_'+bp2+'_' + 'var_dst'

                # Add the statistical measures to the temporary DataFrame
                self.addrow(temp_df, self.path_fileName[1], mean_dst_name, mean_dst)
                self.addrow(temp_df, self.path_fileName[1], min_dst_name, min_dst)
                self.addrow(temp_df, self.path_fileName[1], max_dst_name, max_dst)
                self.addrow(temp_df, self.path_fileName[1], var_dst_name, var_dst)
        
        return temp_df
    
    def _calculateDistance(self, df:pd.DataFrame, bodypart1:str, bodypart2:str):
        """
        Calculate the Euclidean distance between two specified body parts across all frames.

        This method computes the distance between two body parts for each frame in the provided
        DataFrame. The Euclidean distance is calculated using the x and y coordinates of the body parts.

        Args:
            df (pd.DataFrame): The DataFrame containing the DLC tracking data with coordinates for body parts.
            bodypart1 (str): The name of the first body part.
            bodypart2 (str): The name of the second body part.

        Returns:
            np.ndarray: An array containing the Euclidean distances between the two body parts for each frame.

        Note:
            The method assumes that `df` contains the tracking data with a MultiIndex (scorer, bodypart, coords).
            The distances are calculated using the numpy library for efficient computation. The method returns
            an array of distances corresponding to the number of frames in the DataFrame.
        """
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=pd.errors.PerformanceWarning)
        
            # Extract the coordinates for both body parts
            p1 = df[(self.scorer, bodypart1)][['x', 'y']].values
            p2 = df[(self.scorer, bodypart2)][['x', 'y']].values

            # Compute the difference in coordinates between the two body parts
            p_diff = (p1 - p2)

            # Calculate the Euclidean distance for each frame
            dist = np.sqrt(p_diff[:,0]**2 + p_diff[:,1]**2)
        
        return dist
        
    def _getYmeasures(self, df:pd.DataFrame, bodyparts:str, ref_distance:float):
        """
        Calculate and store normalized y-coordinate measures of specified body parts.

        This method iterates over a list of body parts, normalizes their y-coordinates by a reference
        distance, and computes statistical measures (mean, minimum, maximum, variance) for these
        normalized y-coordinates. The results are stored in a temporary DataFrame with columns for
        the ID, feature name, and feature value.

        Args:
            df (pd.DataFrame): The DataFrame containing the normalized DLC tracking data.
            bodyparts (list): A list of body parts for which the y-coordinate measures will be calculated.
            ref_distance (float): A reference distance used to normalize the y-coordinates.

        Returns:
            pd.DataFrame: A DataFrame containing the calculated mean, minimum, maximum, and variance
                          of the normalized y-coordinates for each specified body part.

        Note:
            The method assumes that `df` contains the tracking data with a MultiIndex (scorer, bodypart, coords).
            Only data points with a likelihood greater than 0.95 are considered for the y-coordinate measures.
            The method uses `self.addrow` to append the statistical measures to the temporary DataFrame. The ID
            used in `self.addrow` is assumed to be obtained from `self.path_fileName[1]`
        """
        
        
        temp_df = pd.DataFrame(columns = ['ID', 'feature', 'value'])
            
        for bp in bodyparts:
            corr_df = df[df[(self.scorer, bp, 'likelihood')] > self.pcutoff][self.scorer, bp, 'y']
            
            # Record the number of frames with high likelihood for the body part
            self.y_length = corr_df.shape[0]
            
            if len(corr_df) > 0:
            
                # Normalize the y-coordinates
                norm_y = corr_df/ref_distance

                # Calculate statistical measures for the normalized y-coordinates
                mean_y = np.mean(norm_y)
                min_y = np.min(norm_y)
                max_y = np.max(norm_y)
                var_y = np.var(norm_y)

                # Define feature names
                mean_y_name = bp + 'mean_y'
                min_y_name = bp + 'min_y'
                max_y_name = bp + 'max_y'
                var_y_name = bp + 'var_y'

                # Add the statistical measures to the temporary DataFrame
                self.addrow(temp_df, self.path_fileName[1], mean_y_name, mean_y)
                self.addrow(temp_df, self.path_fileName[1], min_y_name, min_y)
                self.addrow(temp_df, self.path_fileName[1], max_y_name, max_y)
                self.addrow(temp_df, self.path_fileName[1], var_y_name, var_y)
        
        return temp_df

    def _getAngles(self, df:pd.DataFrame, joints_dict:dict):
        """
        Calculate joint angles and velocities, and store their statistical measures.

        This method uses the provided dictionary of joints to compute the angles and velocities
        for each joint using the dlc2kinematics library. It then calculates the mean, minimum,
        maximum, and variance for the joint angles, as well as the mean, minimum, and maximum
        for the joint velocities. These statistical measures are stored in a temporary DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame containing the normalized DLC tracking data.
            joints_dict (dict): A dictionary where keys are joint names and values are lists of
                                body parts that form the joint.

        Returns:
            pd.DataFrame: A DataFrame containing the calculated statistical measures for joint angles
                          and velocities.

        Note:
            The method assumes that `dlc2kinematics.compute_joint_angles` and
            `dlc2kinematics.compute_joint_velocity` are available functions that compute joint angles
            and velocities, respectively. The DataFrame `df` is expected to contain tracking data with
            a MultiIndex (scorer, bodypart, coords). The method uses `self.addrow` to append the
            statistical measures to the temporary DataFrame. The ID used in `self.addrow` is assumed
            to be obtained from `self.path_fileName[1]`.
        """

        # Extract the unique values for each level from the columns
        # scorer = [self.scorer]  # Replace with your actual scorer value(s)
        # bodyparts = self.bodyparts  # Replace with your actual bodyparts
        # coords = ['x', 'y', 'likelihood']  # Replace with your actual coords

        # # Create the MultiIndex from the product of the level values
        # multi_index = pd.MultiIndex.from_product([scorer, bodyparts, coords], names=['scorer', 'bodyparts', 'coords'])

        # # Assign the MultiIndex to the columns of the DataFrame
        # df.columns = multi_index

        
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=FutureWarning)
            # Compute joint angles using the dlc2kinematics library
            joint_angles = dlc2kinematics.compute_joint_angles(df, joints_dict, dropnan=True, smooth=False, save=False)
            temp_df = pd.DataFrame(columns = ['ID', 'feature', 'value'])

            # Calculate statistical measures for joint angles
            for angle in joint_angles.columns:
                mean_angle = joint_angles[angle].mean()
                min_angle = joint_angles[angle].min()
                max_angle = joint_angles[angle].max()
                var_angle = joint_angles[angle].var()

                mean_name = angle+'_mean_angle'
                min_name = angle+'_min_angle'
                max_name = angle+'_max_angle'
                var_name = angle + 'var_angle'

                self.addrow(temp_df, self.path_fileName[1], mean_name, mean_angle)
                self.addrow(temp_df, self.path_fileName[1], min_name, min_angle)
                self.addrow(temp_df, self.path_fileName[1], max_name, max_angle)
                self.addrow(temp_df, self.path_fileName[1], var_name, var_angle)

            # Compute joint velocities
            joint_vel = dlc2kinematics.compute_joint_velocity(joint_angles, dropnan=True, save=False)

            # Calculate statistical measures for joint velocities
            for ang_vel in joint_vel.columns:
                mean_angle_v = joint_vel[ang_vel].mean()
                min_angle_v = joint_vel[ang_vel].min()
                max_angle_v = joint_vel[ang_vel].max()

                mean_name_v = ang_vel +'_mean_angle_vel'
                min_name_v = ang_vel +'_min_angle_vel'
                max_name_v = ang_vel +'_max_angle_vel'

                self.addrow(temp_df, self.path_fileName[1], mean_name_v, mean_angle_v)
                self.addrow(temp_df, self.path_fileName[1], min_name_v, min_angle_v)
                self.addrow(temp_df, self.path_fileName[1], max_name_v, max_angle_v)
            
        return temp_df
    
    def _getRefDistance(self, df:pd.DataFrame):
        
        ref_dist_name = self.ref_distance_name
        ref_dist_name_components = ref_dist_name.split('_')
        ref_bodypart1 = ref_dist_name_components[0]
        ref_bodypart2 = ref_dist_name_components[1]
        
        cor_df = df[(df[(self.scorer,ref_bodypart1, 'likelihood')] > self.pcutoff) & (df[(self.scorer, ref_bodypart2, 'likelihood')] > self.pcutoff)]
        
        ref_distance = np.mean(self._calculateDistance(cor_df, ref_bodypart1, ref_bodypart2))
        
        self.ref_distance = ref_distance
        
        return ref_distance

    def extractFeatures(self, all_features=True):
        """
        Extract features from normalized DeepLabCut (DLC) tracking data for analysis.

        This method computes various features such as distances, angles, and y-coordinates of body parts.
        It filters the data based on likelihood to ensure only high-confidence points are used in calculations.
        The method also allows for the selection of specific features or the extraction of all features.

        Args:
            ref_bodypart1 (str, optional): The first reference body part for distance calculation. Defaults to 'elbow'.
            ref_bodypart2 (str, optional): The second reference body part for distance calculation. Defaults to 'shoulder'.
            all_features (bool, optional): Flag to determine if all features should be extracted. Defaults to True.

        Returns:
            pd.DataFrame: A DataFrame containing the extracted features.

        Note:
            The method assumes that `self.norm_dlc_df` is a normalized pandas DataFrame containing the tracking data.
            It uses predefined body part combinations and joint dictionaries to calculate distances and angles.
            The method also appends additional metadata such as footslips, time to cross, animal ID, and other
            identifiers to the feature DataFrame. The reference distance is calculated using the mean distance
            between the specified reference body parts.
        """
        
        feature_df = pd.DataFrame()
        df = self.norm_dlc_df.copy()

        df = df.iloc[self.foot_start_frame:self.nose_end_frame, :].reset_index(drop=True)
        
        ref_distance = self._getRefDistance(df)

        joints_dict= {}
        joints_dict['elbow']  = ['shoulder', 'elbow', 'frontPaw']
        joints_dict['hip'] = ['iliacCrest', 'hip', 'knee']
        joints_dict['knee'] = ['hip', 'knee', 'ankle']
        joints_dict['ankle'] = ['knee', 'ankle', 'hindPaw']


        bodyparts = self.bodyparts

        bp_to_remove = {'beamLeftUp', 'beamLeftDown', 'beamRightUp', 'beamRightDown', 'dot'}
        temp_bodyparts = list(filter(lambda x: x not in bp_to_remove, bodyparts))
        bp_combinations = list(itertools.combinations(temp_bodyparts, 2))


        dist_df = self._getDistances(df, bp_combinations, ref_distance)
        angle_df = self._getAngles(df, joints_dict)
        y_df = self._getYmeasures(df, bodyparts[5:], ref_distance)

        feature_df = pd.concat([dist_df, y_df, angle_df], axis=0)
            
        self.addrow(feature_df, self.path_fileName[1], 'footslips', self.footslips)
        self.addrow(feature_df, self.path_fileName[1], 'timeToCross', self.timeToCross)
        self.addrow(feature_df, self.path_fileName[1], 'animalID', self.animalID)
        self.addrow(feature_df, self.path_fileName[1], 'age', self.age)
        self.addrow(feature_df, self.path_fileName[1], 'sex', self.sex)
        self.addrow(feature_df, self.path_fileName[1], 'group_name', self.group)
        self.addrow(feature_df, self.path_fileName[1], 'beam', self.beam)
        self.addrow(feature_df, self.path_fileName[1], 'weight', self.weight)
        self.addrow(feature_df, self.path_fileName[1], 'session', self.session)
        self.addrow(feature_df, self.path_fileName[1], 'ref_distance_name', self.ref_distance_name)
        self.addrow(feature_df, self.path_fileName[1], 'ref_distance_value', self.ref_distance)

        return feature_df



class Experiment():
    
    def __init__(self,
        trialList:list
        ):
        
        self.trialList = trialList
        self.resultsDf, self.long_feature_df=self._createDataframe()
    
    def _createDataframe(self):
        """
        Create a dataframe containing features extracted from trials.

        This function iterates over a list of trial objects, extracts features from each,
        and compiles them into a single dataframe. It then pivots the dataframe to have
        features as columns and IDs as rows, dropping any NaN values and keeping the first
        occurrence of each value. It also removes a reference distance column from the
        pivoted dataframe.

        Returns:
            tuple: A tuple containing:
                - feat_df (DataFrame): A pivoted dataframe with IDs as rows and features as columns.
                - tot_df (DataFrame): A concatenated dataframe with all the extracted features before pivoting.
        """
        
         # Initialize an empty dataframe with columns 'ID', 'feature', and 'value'
        tot_df = pd.DataFrame(columns=['ID', 'feature', 'value'])
        
        # Loop through each trial in the trial list
        for trial in self.trialList:
            temp_df = trial.extractFeatures()
            tot_df = pd.concat([tot_df, temp_df], axis=0)
            
        # Pivot the total dataframe to have features as columns and IDs as rows
        # Use 'first' aggregation function to keep the first occurrence of each value
        feat_df = tot_df.pivot_table(values='value', index='ID', columns='feature', dropna=True, aggfunc='first').reset_index()
        
        # Drop the reference distance column from the pivoted dataframe
        # Note: trial.ref_distance should be accessible and should refer to a valid column name
        feat_df.drop(trial.ref_distance_name, axis=1, inplace=True)
        
        return feat_df, tot_df
    