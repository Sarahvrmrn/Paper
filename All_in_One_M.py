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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from scipy.spatial.distance import cdist
from scipy.spatial.distance import euclidean
from scipy.interpolate import NearestNDInterpolator


# Get all files for the training data set and merge them in on DataFrame

path_train = 'C:\\Users\\sverme-adm\\Desktop\\Mona_TIC'
save_path_train = 'C:\\Users\\sverme-adm\\Desktop\\results_DM'

path_test = 'C:\\Users\\sverme-adm\\Desktop\\Mona_TIC_Test'
save_path_test = 'C:\\Users\\sverme-adm\\Desktop\\results_DMT'

eval_ts = datetime.now().strftime("_%m-%d-%Y_%H-%M-%S")
os.environ["ROOT_PATH"] = hp.mkdir_ifnotexits(
    join(save_path_train, 'result' + eval_ts))

components_LDA = 1
components_PCA = 3



def read_files(path: str, tag: str):
    files = hp.get_all_files_in_dir_and_sub(path)
    files = [f for f in files if f.find('.csv') >= 0]
    merged_df = pd.DataFrame()
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
        df.set_index('RT(milliseconds)', inplace=True)
        new_index = np.arange(35700,1321860, 1)
        df = df.reindex(new_index)
        df = df.interpolate(method='linear',limit_direction='forward', axis=0)
        
        merged_df = pd.merge(
            merged_df, df, how='outer', left_index=True, right_index=True)

        merged_df = merged_df.fillna(0)
        merged_df = merged_df.rename(
            columns={'TIC': file.split('\\')[5]})

        info.append(
            {'Class': file.split('\\')[5], 'filename': os.path.basename(file)})

    df_info = pd.DataFrame(info)

    merged_df.drop(merged_df.index[1080000:], inplace=True)
    merged_df.drop(merged_df.index[:144300], inplace=True)

    def replace_below_threshold(column):
        max_val = column.max()
        threshold = 0.03 * max_val
        return column.where(column >= threshold, 0)
    
    merged_df = merged_df.apply(replace_below_threshold)
    
    hp.save_df(merged_df, join(
        os.environ["ROOT_PATH"], 'data'), f'extracted_features_{tag}')
    hp.save_df(df_info, join(
        os.environ["ROOT_PATH"], 'data'), f'extracted_features_info_{tag}')
    

# Perform Data Reduction with PCA on your Training DataFrame

def create_pca(path_merged_data_train: str, path_merged_data_train_info: str):
    df = pd.read_csv(path_merged_data_train, decimal=',', sep=';')
    df_info = pd.read_csv(path_merged_data_train_info, decimal=',', sep=';')
    df.set_index('RT(milliseconds)', inplace=True)
    pca = PCA(n_components=components_PCA).fit(df.T)

    principalComponents = pca.transform(df.T)

    df_PCA = pd.DataFrame(data=principalComponents, columns=[
        f'PC{i+1}' for i in range(components_PCA)])
    df_PCA.set_index(df_info['Class'], inplace=True)
    df_PCA.to_csv('PC.csv', index=False)

    # Normalize explained variance to get variance ratios
    
    variance_ratio = pca.explained_variance_ratio_

    # Print the percentage of variance explained by each PC
    
    for i, ratio in enumerate(variance_ratio):
        print(f"PC{i + 1}: {ratio * 100:.2f}%")
        
    # Print the sum of all the percentages
        
    total_variance = np.sum(variance_ratio)   
    print(f"Total variance explained: {total_variance * 100:.2f}%")
    
    # plot the Loadings for each PC 
    
    pca_loadings = pca.components_
    
    feature_names = df.index
    top_n = 10

    # Ergebnisse für jede Hauptkomponente ausgeben
    for i in range(components_PCA):
        component = pca_loadings[i]
        # Beträge der Koeffizienten und deren Indizes
        abs_component = np.abs(component)
        top_indices = np.argsort(abs_component)[::-1][:top_n]
        
        print(f"Top {top_n} Features für PC{i+1}:")
        for index in top_indices:
            print(f"  {feature_names[index]}: {component[index]:.4f}")
        print()
        
    loadings_df = pd.DataFrame(data=pca_loadings.T, columns=[f'PC{i+1}' for i in range(len(pca_loadings))])
    for column in loadings_df.columns:
        
        fig = px.line(loadings_df, x=df.index/60000, y=column, title=f"{column} Plot")
        fig.update_xaxes(title_text='RT / min',tickmode='linear', dtick=0.5)
        fig.update_traces(marker=dict(size=4))
        # Save the plot as an HTML file
        fig.write_html(f"{column}_plot_WGes.html")

    # print("Interactive plots saved.")
    
    return pca, df_PCA, df_info

# Perform classification with LDA on your reduced Training DataFrame (PCA DataFrame)

def create_lda(df_pca: pd.DataFrame, df_info: pd.DataFrame):
    X = df_pca.values
    y = df_pca.index
    name = df_info['filename']
    lda = LDA(n_components=components_LDA).fit(X, y)
    X_lda = lda.fit_transform(X, y)
    dfLDA_train = pd.DataFrame(data=X_lda, columns=[
        f'LD{i+1}' for i in range(components_LDA)])
    dfLDA_train['file'] = name
    dfLDA_train.index = y

    # Get your most influential PCs
    
    lda_loadings = lda.coef_
    most_influential_pcs = sorted(range(components_PCA), key=lambda x: abs(lda_loadings[0][x]), reverse=True)
    print(most_influential_pcs)
    
    # Perform Cross Validaion on your LDA (Get confusion matrix)
       
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
    ax.xaxis.set_ticklabels(df_pca.index.unique().tolist(), fontsize=10)
    ax.yaxis.set_ticklabels(df_pca.index.unique().tolist(), fontsize=10)
    plt.show()
    
    cv_scores = cross_val_score(lda, X, y, cv=10)
    print("Average cross-validation score:", cv_scores.mean())
    
    return lda, dfLDA_train

# Perform Data Reduction for your Test DataFrame with the PCA of the Training DataFrame

def push_to_pca(pca: PCA, path_merged_data_test: str, path_merged_data_test_info: str):
    df = pd.read_csv(path_merged_data_test, decimal=',', sep=';')
    df_info = pd.read_csv(path_merged_data_test_info, decimal=',', sep=';')
    df.set_index('RT(milliseconds)', inplace=True)
    transformed_data = pca.transform(df.T)
    dfPCA_test = pd.DataFrame(data=transformed_data, columns=[
        f'PC{i+1}' for i in range(components_PCA)], index=df_info['Class'])
 
    return dfPCA_test, df_info

# Perform classification for your reduced test DataFrame with the LDA of the Training DataFrame

def push_to_lda(lda: LDA, transformed_data: pd.DataFrame, transformed_data_info: pd.DataFrame):
    predictions = lda.predict(transformed_data.values)
    transformed_data_lda = lda.transform(transformed_data.values)
    name = transformed_data_info['filename']

    df_lda_test_transformed = pd.DataFrame(data=transformed_data_lda, columns=[
        f'LD{i+1}' for i in range(components_LDA)])
    df_lda_test_transformed['file'] = name
    df_lda_test_transformed.index = transformed_data.index
    
    return df_lda_test_transformed, predictions

# Combine the Training and Test DataFrames

def combine_data(df_test: pd.DataFrame, df_train: pd.DataFrame):
    df_test['Dataset'] = ['test' for _ in df_test.index]
    df_train['Dataset'] = ['train' for _ in df_train.index]
    df_merged = pd.concat([df_train, df_test], ignore_index=False, sort=False)
    return df_merged

# Plot the LDA and save the Plot

def plot(df: pd.DataFrame):
    fig = px.scatter(df, x='LD1', color=df.index, hover_name='file', symbol='Dataset', symbol_sequence= ['circle', 'diamond'])
    fig.update_traces(marker=dict(size=12,
                              line=dict(width=1,
                                        color='DarkSlateGrey')),
                  selector=dict(mode='markers'))
    hp.save_html(fig, join(os.environ["ROOT_PATH"], 'plots'), 'LDA')
    fig.show()
    
# Plot the PCA and save the Plot
    
def plot_PCA(df: pd.DataFrame):
    fig = px.scatter_3d(df, x='PC1', y='PC2', z='PC3', color=df.index, symbol='Dataset', symbol_sequence= ['circle', 'x'])
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
    
    pca, df_pca, df_info = create_pca(join(
        os.environ["ROOT_PATH"], 'data', f'extracted_features_train.csv'),
        join(
        os.environ["ROOT_PATH"], 'data', f'extracted_features_info_train.csv'))
    
    # create LDA with Training DataFrame
    
    lda, df_lda_train = create_lda(df_pca, df_info)
    
    # perform PCA with Test DataFrame
    
    transformded_data_test, df_info_test = push_to_pca(pca, join(
        os.environ["ROOT_PATH"], 'data', f'extracted_features_test.csv'),
        join(
        os.environ["ROOT_PATH"], 'data', f'extracted_features_info_test.csv'))
    
    # perform LDA with test DataFrame
    
    df_lda_test, predictions = push_to_lda(lda, transformded_data_test, df_info_test)
    
    # Combine both LDA DataFrames
    
    merged_df = combine_data(df_lda_test, df_lda_train)
    merged_df.to_csv('LDA_data.csv', index=False)
    
    # Plot LDA
    
    plot(merged_df)
    
    # combine both PCA DataFrames
    
    merged_pc = combine_data(transformded_data_test, df_pca)
    
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
