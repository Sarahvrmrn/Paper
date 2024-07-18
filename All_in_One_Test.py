import pandas as pd
import os
import seaborn as sns
from helpers import Helpers as hp
from os.path import join
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA, QuadraticDiscriminantAnalysis as QDA
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score


# Choose the path for your data

path_train = 'C:\\Users\\sverme-adm\\Desktop\\Knolle'
save_path_train = 'C:\\Users\\sverme-adm\\Desktop\\res_Knolle'

path_test = 'C:\\Users\\sverme-adm\\Desktop\\Knolle_Test'
save_path_test = 'C:\\Users\\sverme-adm\\Desktop\\res_Knolle'

eval_ts = datetime.now().strftime("_%m-%d-%Y_%H-%M-%S")
os.environ["ROOT_PATH"] = hp.mkdir_ifnotexits(
    join(save_path_train, 'result' + eval_ts))

# Choose your components for LDA and PCA

components_LDA = 1


# Get all files for your data set, merge them in on DataFrame and save the DataFrame to CSV

def read_files(path: str, tag: str):
    files = hp.get_all_files_in_dir_and_sub(path)
    files = [f for f in files if f.find('.csv') >= 0]
    merged_df = pd.DataFrame()
    merged_peak_df = pd.DataFrame()
    info = []

    for file in files:
        df = hp.read_file(file, dec='.', sepi=',')[['RT(milliseconds)', 'TIC']]
        x = df['RT(milliseconds)']
        y = df['TIC']
        y = hp.smooth_spectrum(y)
        baseline = hp.baseline_correction(y)
        y_corrected = y- baseline
        y = hp.area_normalization(x,y_corrected)
        baseline_y_area = hp.baseline_correction(y)*0.05
    
        # x_values = df['RT(milliseconds)']
        # y_values = df['TIC'].rolling(window=7).mean()
        # df = pd.DataFrame({'RT(milliseconds)': x_values, 'TIC': y_values})
           
        df.set_index('RT(milliseconds)', inplace=True)
        new_index = np.arange(120000, 823100, 100)
        df = df.reindex(new_index)
        
        merged_df = pd.merge(merged_df, df, how='outer', left_index=True, right_index=True)
        merged_df = merged_df.fillna(0)
        merged_df = merged_df.rename(columns={'TIC': file.split('\\')[5]})

        info.append(
            {'Class': file.split('\\')[5], 'filename': os.path.basename(file)})
        
        peaks = hp.pick_peaks(y)
        
        peak_areas = hp.integrate_spectrum(x, y, peaks) 
        y_corr_bl = baseline_y_area[peaks]  
        x_peaks = x[peaks] 
    # integrate picked peaks
     # define list of x-values for each peak
        peak_df = pd.DataFrame({'Peak Position Index': peaks, 'RT(milliseconds)': x_peaks, 'Peak Area': peak_areas, 'Corrected baseline': y_corr_bl})
        peak_df = peak_df.drop(['Peak Position Index', 'Corrected baseline'], axis=1)

        peak_df.set_index('RT(milliseconds)', inplace=True)
        merged_peak_df = pd.merge(merged_peak_df, peak_df, how='outer', left_index=True, right_index=True)
        merged_peak_df = merged_peak_df.fillna(5)
        merged_peak_df = merged_peak_df.rename(columns={'Peak Area': file.split('\\')[5]})
        
        
    merged_df.drop(merged_df.index[6601:], inplace=True)
    # merged_df.drop(merged_df.index[:600], inplace=True)

    df_info = pd.DataFrame(info)
    
    threshold_percent = 4.5 # threshold in %
    
    max_value = merged_df.max().max()
    threshold = max_value * (threshold_percent / 100)
    
    merged_df[merged_df <= threshold] = 0

    
    hp.save_df(merged_df, join(
        os.environ["ROOT_PATH"], 'data'), f'extracted_features_{tag}')
    hp.save_df(df_info, join(
        os.environ["ROOT_PATH"], 'data'), f'extracted_features_info_{tag}')
    
    hp.save_df(merged_peak_df, join(
        os.environ["ROOT_PATH"], 'data'), f'extracted_peaks_{tag}')
    hp.save_df(df_info, join(
        os.environ["ROOT_PATH"], 'data'), f'extracted_peaks_info_{tag}')
   
   
    

# Perform classification with LDA on your reduced Training DataFrame (PCA DataFrame)

def create_lda(path_merged_data_train: str, path_merged_data_train_info: str):
    df = pd.read_csv(path_merged_data_train, decimal=',', sep=';')
    df_info = pd.read_csv(path_merged_data_train_info, decimal=',', sep=';')
    df.set_index('RT(milliseconds)', inplace=True)
    df = df.T
    X = df.values
    df.set_index(df_info['Class'], inplace=True)
    y = df.index
   
    name = df_info['filename']
    
    
    lda = LDA(n_components=components_LDA).fit(X, y)
    X_lda = lda.fit_transform(X, y)
    dfLDA_train = pd.DataFrame(data=X_lda, columns=[
        f'LD{i+1}' for i in range(components_LDA)])
    dfLDA_train['file'] = name
    dfLDA_train.index = y

    lda_likelihood = lda.score(X, y)

    # Perform Cross Validation on your LDA (Get confusion matrix)
       
    y_pred = lda.predict(X)
    cm = confusion_matrix(y, y_pred)
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    print(cm)
    top_margin =0.06
    bottom_margin = 0.06
    
    # Plot the confusion matrix as a heatmap
    
    fig, ax = plt.subplots(
        figsize=(10,8), 
        gridspec_kw=dict(top=1-top_margin, bottom=bottom_margin))
    sns.heatmap(cm / cm_sum.astype(float), annot=True , cbar=False, cmap='gist_earth', fmt='.2%')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    cbar = ax.figure.colorbar(ax.collections[0])
    cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    cbar.set_ticklabels(["0%",'20%', '40%', '60%', '80%', "100%"])
    ax.xaxis.set_ticklabels(df.index.unique().tolist(), fontsize=10)
    ax.yaxis.set_ticklabels(df.index.unique().tolist(), fontsize=10)
    plt.show()
    
    cv_scores = cross_val_score(lda, X, y, cv=10)
    print("Average cross-validation score:", cv_scores.mean())
    
    lda_loadings = lda.scalings_
    
    
    loadings_df = pd.DataFrame(data=lda_loadings, columns=[f'LD{i+1}' for i in range(lda_loadings.shape[1])])
    loadings_df.set_index(np.arange(120000, 780100, 100), inplace=True)
    print(loadings_df)
    
    loadings_df.to_csv('Scalings.csv')

    for column in loadings_df.columns:
        
        fig = px.line(loadings_df, x=loadings_df.index/60000, y=column, title=f"{column} Plot")
        fig.update_xaxes(title_text='RT / min',tickmode='linear', dtick=0.5)
        fig.update_traces(marker=dict(size=4))
        # Save the plot as an HTML file
        fig.write_html(f"{column}_Scaling_Sorten.html")

    print("Interactive plots saved.")
    
    return lda, dfLDA_train, df_info


# Perform classification for your reduced test DataFrame with the LDA of the Training DataFrame

def push_to_lda(lda: LDA , path_merged_data_test: str, path_merged_data_test_info: str,):
    df = pd.read_csv(path_merged_data_test, decimal=',', sep=';')
    df_info = pd.read_csv(path_merged_data_test_info, decimal=',', sep=';')
    df.set_index('RT(milliseconds)', inplace=True)
    df = df.T
    df.set_index(df_info['Class'], inplace=True)
    predictions = lda.predict(df.values)
    transformed_data_lda = lda.transform(df.values)
    name = df_info['filename']

    df_lda_test_transformed = pd.DataFrame(data=transformed_data_lda, columns=[
        f'LD{i+1}' for i in range(components_LDA)])
    df_lda_test_transformed['file'] = name
    df_lda_test_transformed.index = df.index
    
    return df_lda_test_transformed, predictions


def combine_data(df_test: pd.DataFrame, df_train: pd.DataFrame):
    df_test['Dataset'] = ['test' for _ in df_test.index]
    df_train['Dataset'] = ['train' for _ in df_train.index]
    df_merged = pd.concat([df_train, df_test], ignore_index=False, sort=False)
    return df_merged

# Plot the LDA and save the Plot

def plot(df: pd.DataFrame):
    fig = px.scatter(df, x='LD1', color=df.index, hover_name='file', symbol='Dataset', symbol_sequence= ['circle', 'diamond'])
    fig.update_yaxes(showticklabels=False)
    fig.update_traces(marker=dict(size=12,
                              line=dict(width=1,
                                        color='DarkSlateGrey')),
                  selector=dict(mode='markers'))
    hp.save_html(fig, join(os.environ["ROOT_PATH"], 'plots'), 'LDA')
    fig.show()
    
# Plot the PCA and save the Plot
    
def plot_PCA(df: pd.DataFrame):
    fig = px.scatter_3d(df, x='PC1', y='PC2', z='PC3', color=df.index, symbol='Dataset', symbol_sequence= ['circle', 'diamond'])
    fig.update_traces(marker=dict(size=12,
                              line=dict(width=2,
                                        color='DarkSlateGrey')),
                  selector=dict(mode='markers'))
    hp.save_html(fig, join(os.environ["ROOT_PATH"], 'plots'), 'PCA')
    fig.show()



# Main Program to use all definitions

if __name__ == '__main__':
    
    # produce training DataFrame
    
    df_raw_train = read_files(path_train, 'train')
    
    # produce test DataFrame
    
    df_raw_test = read_files(path_test, 'test')

    # create PCA with Training DataFrame
    
    lda, df_lda_train, df_info = create_lda(join(
         os.environ["ROOT_PATH"], 'data', f'extracted_features_train.csv'),
         join(
         os.environ["ROOT_PATH"], 'data', f'extracted_features_info_train.csv'))
    
    # # perform PCA with Test DataFrame
    
    df_lda_test, predictions = push_to_lda(lda, join(
         os.environ["ROOT_PATH"], 'data', f'extracted_features_test.csv'),
         join(
         os.environ["ROOT_PATH"], 'data', f'extracted_features_info_test.csv'))
    
    # Combine both LDA DataFrames
    # lda, df_lda_train_peaks, df_info = create_lda(join(
    #      os.environ["ROOT_PATH"], 'data', f'extracted_peaks_train.csv'),
    #      join(
    #      os.environ["ROOT_PATH"], 'data', f'extracted_peaks_info_train.csv'))
    
    # df_lda_test_peaks, predictions = push_to_lda(lda, join(
    #      os.environ["ROOT_PATH"], 'data', f'extracted_peaks_test.csv'),
    #      join(
    #      os.environ["ROOT_PATH"], 'data', f'extracted_peaks_info_test.csv'))
    
    #merged_df_peaks = combine_data(df_lda_train_peaks,df_lda_test_peaks)
    
    merged_df = combine_data(df_lda_test, df_lda_train)
    
    # Plot LDA
    
    plot(merged_df)
    #plot(merged_df_peaks)
    # combine both PCA DataFrames
    
    # merged_pc = combine_data(transformded_data_test, df_pca)
    
    # plot PCA
    
    # plot_PCA(merged_pc)
    
    # Calculate accuracy for each class in LDA
    accuracy_per_class = {}
    # Get the unique classes in the test dataset
    unique_classes = set(df_lda_test.index)  
    # Filter test labels for the current class and Calculate accuracy for the current class
    for cls in unique_classes:
        indices = df_lda_test.index == cls  
        accuracy = accuracy_score(df_lda_test.index[indices], predictions[indices])  
        accuracy_per_class[cls] = accuracy
    # Print accuracy for each class
    for cls, accuracy in accuracy_per_class.items():
        print(f"Accuracy for class {cls}: {accuracy:.2f}")
