import json as js
from os import listdir,  scandir
from os.path import isfile, join
from pathlib import Path
from shutil import rmtree
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.markers import MarkerStyle
from scipy.signal import savgol_filter, find_peaks
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy import integrate
import pandas as pd
import os
import numpy as np


class Helpers:

    def concat_df(list_dfs:list[pd.DataFrame]):
        time_axis = list_dfs['time']
        print(time_axis)
        df = pd.concat(list_dfs, axis=1)
        # df.set_index(df.columns[0],inplace=True)
        df.drop(columns='time', inplace=True)
        df.set_index(time_axis, inplace=True)
        return df
    
    def round_data(data:pd.DataFrame, name:str):
        df_round  = data.round(2)
        df_round.set_index('Ret.Time', inplace=True)
        data =  []
        idx_new =[]
        for idx in df_round.index.unique():
           mean = df_round[df_round.index == idx].mean()[0]
           data.append(mean)
           idx_new.append(idx)
        df = pd.DataFrame({'time':idx_new, name:data})
        return df
        
        
        
    
    def df_to_dict(df: pd.DataFrame) -> dict:
        data_expanded = {}
        for col in df.columns:
            for i in df.index:
                data_expanded[f'p{i}_{col}'] = df.loc[i][col]
        return data_expanded

    def read_file(path: str, skip_header=0, dec='.', sepi=','):
        df = pd.read_csv(path,  sep=sepi, decimal=dec, skiprows=skip_header)
        return df

    def get_all_files_in_dir_and_sub(path: str):
        files = []
        for r, d, f in os.walk(path):
            for file in f:
                files.append(os.path.join(r, file))
        return files

    def get_all_folders(path):
        return [os.path.join(path, i) for i in os.listdir(path)]

    def save_html(html_object, path: str, name: str):
        Path(path).mkdir(parents=True, exist_ok=True)
        path = path + '\\' + name + '.html'
        print(path)
        html_object.write_html(path)

    def get_all_files_in_sub(path: str):
        return [join(path, f)
                for f in listdir(path) if isfile(join(path, f))]

    def get_key_by_value(data: dict, value: int):
        for key, val in data.items():
            if val == value:
                return key

    def sample_to_numbers(samlpes: pd.Series):
        samles_unique = samlpes.unique()
        sample_dict = {}
        for sample, i in zip(samles_unique, range(len(samles_unique))):
            sample_dict[sample] = i
        numbers = [sample_dict[s] for s in samlpes]
        return sample_dict, numbers

    def read_json(folder, filename):
        # read json to dict
        with open(join(folder, filename)) as json_file:
            return js.load(json_file)

    def get_name_from_info(info: dict):
        # extract name from path
        name = f"{info['sample']}_{info['height']}_{info['number']}"
        return name

    def get_subfolders(path):
        # list all subfolders
        return [f.path for f in scandir(path) if f.is_dir()]

    def clean_info_meaurement(info: dict):
        # deletes obsolete infos about measurement
        cleaned_info = {}
        for key in ['datetime', 'height', 'number', 'rate', 'sample']:
            cleaned_info[key] = info[key]
        return cleaned_info

    def get_path_data(path):
        # returns path to txt with data
        files = [f for f in listdir(path) if isfile(join(path, f))]
        for file in files:
            if file.find('txt') > 0:
                return join(path, file)

    def del_results(path):
        folders = [f.path for f in scandir(path) if f.is_dir()]
        for folder in folders:
            if folder.find('results') > 0:
                rmtree(join(path, folder))

    def one_layer_back(path: str) -> str:
        # get path one layer up
        new_path = path[:path.rfind('\\')]
        return new_path

    def get_path_info(path):
        # returns path to json with info
        files = [f for f in listdir(path) if isfile(join(path, f))]
        for file in files:
            if file.find('json') > 0:
                return join(path, file)

    def mkdir_ifnotexits(path):
        # mkdir if not exists
        Path(path).mkdir(parents=True, exist_ok=True)
        return path

    def save_df(df, path, name, index=True):
        Path(path).mkdir(parents=True, exist_ok=True)
        path = join(path, f'{name}.csv')
        print(path)
        df.to_csv(path, sep=';', decimal=',', index=index)

        # Smoothing the spectrum using Savitzky-Golay filter
    def smooth_spectrum(y, window_length=11, polyorder=3):
        return savgol_filter(y, window_length, polyorder)

    # Baseline correction using asymmetric least squares smoothing
    def baseline_correction(y, lam=1e6, p=0.001, niter=10):
        L = len(y)
        D = sparse.diags([1,-2,1],[0,-1,-2], shape=(L,L-2))
        D = lam * D.dot(D.transpose()) # Precompute this term since it does not depend on `w`
        w = np.ones(L)
        W = sparse.spdiags(w, 0, L, L)
        for i in range(niter):
            W.setdiag(w) # Do not create a new matrix, just update diagonal values
            Z = W + D
            z = spsolve(Z, w*y)
            w = p * (y > z) + (1-p) * (y < z)
        return z

    # Normalization of the spectrum to total area
    def area_normalization(x, y):
        cumulative_area = []
        normalized_area = []
        for i in range(len(x)):
                if i == 0:
                    area = 0
                    cumulative_area.append(area) # Set area to 0 for the first data point
                else:
                    area = ((x[i] - x[i-1]) * (y[i] + y[i-1]) / 2)
                    cumulative_area.append(area)
        
        total_area = sum(cumulative_area)
        normalized_area = (cumulative_area/total_area)*100
        return normalized_area


    # Integration of the spectrum
    def integrate_spectrum(x, y, peaks):
        peak_areas = []
        for peak in peaks:
            left = max(0, peak - 15)
            right = min(len(y), peak + 15)
            area = integrate.trapezoid(y[left:right], x[left:right])
            peak_areas.append(area)
        return peak_areas

    # Peak picking
    def pick_peaks(y, height=None, prominence=0.02, distance=25):
        peaks, _ = find_peaks(y, prominence=prominence)
        return peaks
    
    def check_unique(df, column_name, bin_interval):
    # Ensure each bin contains only one value
        unique_bins = df['bin'].value_counts()
        while any(unique_bins > 1):
            for bin_value in unique_bins[unique_bins > 1].index:
                indices = df[df['bin'] == bin_value].index
                for idx in range(len(indices) - 1, 0, -1):
                    current_value = df.at[indices[idx], column_name]
                    previous_value = df.at[indices[idx - 1], column_name]
                    if current_value - bin_value < bin_interval / 2 and previous_value - bin_value < bin_interval / 2:
                        # Both values are closer to the lower edge, move the first value
                        new_bin_value = bin_value - bin_interval # if bin_value - bin_interval in bins else bin_value + bin_interval
                        df.at[indices[idx - 1], 'bin'] = new_bin_value
                    elif current_value - bin_value >= bin_interval / 2 and previous_value - bin_value >= bin_interval / 2:
                        # Both values are closer to the upper edge, move the second value
                        new_bin_value = bin_value + bin_interval # if bin_value + bin_interval in bins else bin_value - bin_interval
                        df.at[indices[idx], 'bin'] = new_bin_value
                    else:
                        # Move the closer value accordingly
                        if abs(current_value - (bin_value + bin_interval / 2)) < abs(previous_value - (bin_value - bin_interval / 2)):
                            new_bin_value = bin_value + bin_interval # if bin_value + bin_interval in bins else bin_value - bin_interval
                            df.at[indices[idx], 'bin'] = new_bin_value
                        else:
                            new_bin_value = bin_value - bin_interval # if bin_value - bin_interval in bins else bin_value + bin_interval
                            df.at[indices[idx - 1], 'bin'] = new_bin_value
        unique_bins = df['bin'].value_counts()
        return unique_bins


    def bin_data(df, column_index, bin_interval):
        # Get the column name based on the index
        column_name = df.columns[column_index]
        
        # Determine the min and max values for binning
        min_value = 2.20
        max_value = df[column_name].max()
        
        # Create bin edges
        # bins = np.arange(min_value, max_value + bin_interval, bin_interval)
        bins = np.arange(start=min_value, stop=np.ceil(max_value*10)/10 + bin_interval, step=bin_interval)
        print(bins)

        # Assign each value to a bin
        # df['bin'] = pd.cut(df[column_name], bins=bins, labels=bins[:-1], include_lowest=True)
        df['bin'] = pd.cut(df[column_name], bins=bins, labels=bins[:-1], right=False, include_lowest=True).astype(float).round(2)
        print(df['bin'])
        
        check_unique(df, column_name, bin_interval)
        
        # Round the bin values to 2 decimal places
        df['bin'] = df['bin'].astype(float).round(2)
        
        return df