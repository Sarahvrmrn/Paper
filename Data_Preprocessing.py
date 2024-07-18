import pandas as pd
import os
from helpers import Helpers as hp
from os.path import join
from datetime import datetime
import numpy as np



# Choose the path for your train and test data
path_train = 'C:\\Users\\sverme-adm\\Desktop\\Knolle'
save_path_train = 'C:\\Users\\sverme-adm\\Desktop\\res_Knolle'

path_test = 'C:\\Users\\sverme-adm\\Desktop\\Knolle_Test'
save_path_test = 'C:\\Users\\sverme-adm\\Desktop\\res_Knolle'


eval_ts = datetime.now().strftime("_%m-%d-%Y_%H-%M-%S")
os.environ["ROOT_PATH"] = hp.mkdir_ifnotexits(
    join(save_path_train, 'result' + eval_ts))

class Preproccesing:
# Get all files for your data set, merge them in on DataFrame and save the DataFrame to CSV
    def read_files(path: str, tag: str):
        # Load all CSV-files from a folder
        files = hp.get_all_files_in_dir_and_sub(path)
        files = [f for f in files if f.find('.csv') >= 0]
        # Create a Dataframe and an Info List
        merged_df = pd.DataFrame()
        info = []
        #For each file do folllowing steps
        for file in files:
            # read the important columns of the file
            df = hp.read_file(file, dec='.', sepi=',')[['RT(milliseconds)', 'TIC']]
            # Define x and y
            x = df['RT(milliseconds)']
            y = df['TIC']
            # smooth y
            y = hp.smooth_spectrum(y)
            # Basline correction on y
            baseline = hp.baseline_correction(y)
            y_corrected = y- baseline
            # normalise y on the total area
            y = hp.area_normalization(x,y_corrected)
            # get the baseline of the normalised area
            baseline_y_area = hp.baseline_correction(y)*0.05
        
            # if neccessary use the rolling mean in order to eliminate RT-shifts
            
            # x_values = df['RT(milliseconds)']
            # y_values = df['TIC'].rolling(window=7).mean()
            # df = pd.DataFrame({'RT(milliseconds)': x_values, 'TIC': y_values})
            
            # set index to RT and rearrange the RT if neccessary
            df.set_index('RT(milliseconds)', inplace=True)
            new_index = np.arange(120000, 823100, 100)
            df = df.reindex(new_index)
            # merge the precossed files to one sinlge file
            merged_df = pd.merge(merged_df, df, how='outer', left_index=True, right_index=True)
            merged_df = merged_df.fillna(0)
            merged_df = merged_df.rename(columns={'TIC': file.split('\\')[5]})
            info.append(
                {'Class': file.split('\\')[5], 'filename': os.path.basename(file)})
        
        # if neccessary drop some parts of your chromatogramm           
        merged_df.drop(merged_df.index[6601:], inplace=True)
        merged_df.drop(merged_df.index[:600], inplace=True)

        # create an info Dataframe
        df_info = pd.DataFrame(info)
        
        # if neccessary set a threshold in order to minimize background noises
        threshold_percent = 4.5 # threshold in %
        max_value = merged_df.max().max()
        threshold = max_value * (threshold_percent / 100)
        merged_df[merged_df <= threshold] = 0

        # safe the Dataframe with extracted features and the Dataframe with the correspondig infos
        hp.save_df(merged_df, join(
            os.environ["ROOT_PATH"], 'data'), f'extracted_features_{tag}')
        hp.save_df(df_info, join(
            os.environ["ROOT_PATH"], 'data'), f'extracted_features_info_{tag}')

    
    def read_files_peakpicking(path: str, tag: str):
        # Load all CSV-files from a folder
        files = hp.get_all_files_in_dir_and_sub(path)
        files = [f for f in files if f.find('.csv') >= 0]
        # Create a Dataframe and an Info List
        merged_peak_df = pd.DataFrame()
        info = []
        #For each file do folllowing steps
        for file in files:
            # read the important columns of the file
            df = hp.read_file(file, dec='.', sepi=',')[['RT(milliseconds)', 'TIC']]
            # Define x and y
            x = df['RT(milliseconds)']
            y = df['TIC']
            # smooth y
            y = hp.smooth_spectrum(y)
            # perform baseline correction on y
            baseline = hp.baseline_correction(y)
            y_corrected = y- baseline
            # normalise y on the total area
            y = hp.area_normalization(x,y_corrected)
            # get the baseline of the normalised area
            baseline_y_area = hp.baseline_correction(y)*0.05
            # perform peak picking
            peaks = hp.pick_peaks(y)
            # integrate the areas of each peak
            peak_areas = hp.integrate_spectrum(x, y, peaks) 
            # calculate baseline of the peak areas
            y_corr_bl = baseline_y_area[peaks] 
            # get the corrsponding RTs of each peak 
            x_peaks = x[peaks] 
       
            # create a dataframe with x and y of each peak
            peak_df = pd.DataFrame({'Peak Position Index': peaks, 'RT(milliseconds)': x_peaks, 'Peak Area': peak_areas, 'Corrected baseline': y_corr_bl})
            peak_df = peak_df.drop(['Peak Position Index', 'Corrected baseline'], axis=1)
            # set RT as index
            peak_df.set_index('RT(milliseconds)', inplace=True)
            # merge all files into one
            merged_peak_df = pd.merge(merged_peak_df, peak_df, how='outer', left_index=True, right_index=True)
            merged_peak_df = merged_peak_df.fillna(0)
            merged_peak_df = merged_peak_df.rename(columns={'Peak Area': file.split('\\')[5]})
        # create a dataframe with the infos of the data files
        df_info = pd.DataFrame(info)
        # save the dataframe with extracted peaks and a dataframe withthe coressponding infos 
        hp.save_df(merged_peak_df, join(
            os.environ["ROOT_PATH"], 'data'), f'extracted_peaks_{tag}')
        hp.save_df(df_info, join(
            os.environ["ROOT_PATH"], 'data'), f'extracted_peaks_info_{tag}')
        

    
 