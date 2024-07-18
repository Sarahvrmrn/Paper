import pandas as pd
import os
import seaborn as sns
from helpers import Helpers as hp
from Data_Preprocessing import Preprocessing as pp
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score


# Choose your components for LDA and PCA
components_LDA = 1

class useLDA:
    
    #create LDA with your tain data
    def create_lda(path_merged_data_train: str, path_merged_data_train_info: str):
        # read your preprocessed and merged train data and set x and y
        df = pd.read_csv(path_merged_data_train, decimal=',', sep=';')
        df_info = pd.read_csv(path_merged_data_train_info, decimal=',', sep=';')
        df.set_index('RT(milliseconds)', inplace=True)
        df = df.T
        X = df.values
        df.set_index(df_info['Class'], inplace=True)
        y = df.index
        # use you info file in order to give each data column the right name
        name = df_info['filename']
        
        # perform LDA
        lda = LDA(n_components=components_LDA).fit(X, y)
        X_lda = lda.fit_transform(X, y)
        dfLDA_train = pd.DataFrame(data=X_lda, columns=[
            f'LD{i+1}' for i in range(components_LDA)])
        dfLDA_train['file'] = name
        dfLDA_train.index = y

        # Perform confusion matrix on your LDA to check your train data on correctness
        y_pred = lda.predict(X)
        cm = confusion_matrix(y, y_pred)
        cm_sum = np.sum(cm, axis=1, keepdims=True)
        # Extract values
        TN, FP, FN, TP = cm.ravel()
        # Calculate precision and recall
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        # Calculate F1-score
        f1_score = 2 * (precision * recall) / (precision + recall)
        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"F1 Score: {f1_score:.2f}")

        # Plot the confusion matrix as a heatmap
        top_margin =0.06
        bottom_margin = 0.06
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
        # if neccessary perform cross validation to test robustness of your LDA
        cv_scores = cross_val_score(lda, X, y, cv=10)
        print("Average cross-validation score:", cv_scores.mean())
        
        # get features of your LDA
        lda_loadings = lda.scalings_
        # create a dataframe with all features of each LD and calculate them into the corresponding RT
        loadings_df = pd.DataFrame(data=lda_loadings, columns=[f'LD{i+1}' for i in range(lda_loadings.shape[1])])
        loadings_df.set_index(np.arange(120000, 780100, 100), inplace=True)
        print(loadings_df)
        # save the features to CSV files
        loadings_df.to_csv('Scalings.csv')
        # plot the features in order to visualise the most important RT times for classification in each LD
        for column in loadings_df.columns: 
            fig = px.line(loadings_df, x=loadings_df.index/60000, y=column, title=f"{column} Plot")
            fig.update_xaxes(title_text='RT / min',tickmode='linear', dtick=0.5)
            fig.update_traces(marker=dict(size=4))
            # Save the plot as an HTML file
            fig.write_html(f"{column}_Scaling_Sorten.html")
        print("Interactive plots saved.")
        
        return lda, dfLDA_train, df_info

    # push your test data in the created LDA to get predicted classification
    def push_to_lda(lda: LDA , path_merged_data_test: str, path_merged_data_test_info: str,):
        # read your preprocessed and merged test data and set x
        df = pd.read_csv(path_merged_data_test, decimal=',', sep=';')
        df_info = pd.read_csv(path_merged_data_test_info, decimal=',', sep=';')
        df.set_index('RT(milliseconds)', inplace=True)
        df = df.T
        df.set_index(df_info['Class'], inplace=True)
        # predict y with the created LDA and set name for classes
        predictions = lda.predict(df.values)
        transformed_data_lda = lda.transform(df.values)
        name = df_info['filename']
    #create a dataframe with all LDA values from your test data
        df_lda_test_transformed = pd.DataFrame(data=transformed_data_lda, columns=[
            f'LD{i+1}' for i in range(components_LDA)])
        df_lda_test_transformed['file'] = name
        df_lda_test_transformed.index = df.index
        
        return df_lda_test_transformed, predictions

    # combine train and test data into one dataframe in order to plot them
    def combine_data(df_test: pd.DataFrame, df_train: pd.DataFrame):
        df_test['Dataset'] = ['test' for _ in df_test.index]
        df_train['Dataset'] = ['train' for _ in df_train.index]
        df_merged = pd.concat([df_train, df_test], ignore_index=False, sort=False)
        return df_merged
    
    # Plot the LDA and save the Plot
    def plot(df: pd.DataFrame):
        # use scatter plot for 1D/2D and scatter 3d for 3D plot, adjust x,y,z 
        fig = px.scatter(df, x='LD1', color=df.index, hover_name='file', symbol='Dataset', symbol_sequence= ['circle', 'diamond'])
        fig.update_yaxes(showticklabels=False)
        fig.update_traces(marker=dict(size=12,
                                line=dict(width=1,
                                            color='DarkSlateGrey')),
                    selector=dict(mode='markers'))
        hp.save_html(fig, join(os.environ["ROOT_PATH"], 'plots'), 'LDA')
        fig.show()
 
    # Calculate accuracy of test data for each class in LDA
    def accuracy(df_train:pd.DataFrame, df_test: pd.DataFrame):
        accuracy_per_class = {}
        # Get the unique classes in the test dataset
        unique_classes = set(df_train.index)  
        # Filter test labels for the current class and Calculate accuracy for the current class
        for cls in unique_classes:
            indices = df_train.index == cls  
            accuracy = accuracy_score(df_train.index[indices], df_test[indices])  
            accuracy_per_class[cls] = accuracy
        # Print accuracy for each class
        for cls, accuracy in accuracy_per_class.items():
            print(f"Accuracy for class {cls}: {accuracy:.2f}")
