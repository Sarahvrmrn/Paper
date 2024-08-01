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

# Lade die CSV-Datei
df = pd.read_csv('extracted_features_train.csv', sep=';', decimal='.')


# Bestimmte x-Werte, die du extrahieren möchtest
x_values = [121300, 121800, 121900, 122000, 122300, 122400, 122500, 122600, 123600, 
            123700, 123800, 132500, 132600, 132700, 132800, 132900, 133000, 133100, 
            133200, 133300, 133400, 137400, 137500, 137600, 137700, 137800, 159200, 
            159300, 159400, 159500, 159600, 161900, 162000, 178300, 178400, 179000, 
            179100, 179200, 179500, 179600, 179700, 179800, 179900, 180000, 180100, 
            181200, 181300, 181400, 236500, 236600, 236700, 236800, 237900, 238000, 
            238100, 238200, 238300, 238400, 289600, 289700, 289800, 448100, 448200, 
            448300, 448500, 448600, 448700, 449200, 449300, 449400]  # Ersetze mit deinen gewünschten Werten

# Filtere das DataFrame nach den gewünschten x-Werten
filtered_df = df[df['RT(milliseconds)'].isin(x_values)]

# Speichere das gefilterte DataFrame in eine neue CSV-Datei
filtered_df.to_csv('RT_PC.csv', index=False)

print("Neue CSV-Datei wurde erstellt.")