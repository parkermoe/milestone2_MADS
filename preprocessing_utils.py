import pandas as pd
import plotly.graph_objs as go

def MVAnalyzer(df, target, height=1000, margin=dict(l=20, r=20, t=50, b=20)):
    """
    Generate a heatmap to visualize missing values in a pandas DataFrame.

    Parameters:
    df (pandas.DataFrame): The DataFrame to analyze.
    target (str): The name of the target column.
    height (int): The height of the plot in pixels. Default is 1000.
    margin (dict): The margin of the plot. Default is {'l': 20, 'r': 20, 't': 50, 'b': 20}.

    Returns:
    None
    """
    # Generate a matrix to represent missing values
    missing_matrix = df.isnull().astype(int)
    
    # Calculate total rows
    total_rows = len(df)
    
    # Calculate total missing values for each feature
    total_missing_per_feature = df.isnull().sum()

    # Create a DataFrame to hold the counts of missing values per class, per feature
    classes = df[target].dropna().unique()
    missing_counts_per_class = {cls: (df[df[target] == cls].isnull().sum()) for cls in classes}

    # Create the hover text
    hover_text = []
    for i in range(missing_matrix.shape[0]):
        hover_text.append([])
        for j in range(missing_matrix.shape[1]):
            feature = missing_matrix.columns[j]
            total_missing = total_missing_per_feature[j]
            
            # Calculate the percentage of total missing values for the feature
            total_missing_percent = (total_missing / total_rows) * 100
            
            text = f"Row: {i+1}, Feature: {feature}<br>Total Missing: {total_missing} ({total_missing_percent:.2f}%)<br>Missing counts per {target}:<br>"
            for cls in classes:
                missing_for_class = missing_counts_per_class[cls][j]
                if total_missing != 0:
                    percent_missing = (missing_for_class / total_missing) * 100
                else:
                    percent_missing = 0
                text += f"{target} {cls}: {missing_for_class} ({percent_missing:.2f}%)<br>"
            hover_text[-1].append(text)

    # Create the Plotly figure
    fig = go.Figure()

    fig.add_trace(go.Heatmap(z=missing_matrix.values,
                             x=missing_matrix.columns,
                             y=missing_matrix.index,
                             hoverinfo="text",
                             hovertext=hover_text,
                             colorscale="Viridis",
                             showscale=False))

    # Add title and labels
    fig.update_layout(title="Missing Value Heatmap",
                      xaxis_title="Columns",
                      yaxis_title="Rows",
                      height=height,
                      margin=margin)

    # Show the figure
    fig.show()

import numpy as np
import pandas as pd
import plotly.express as px
def standardize_missing_values(df):
    """
    Standardizes and handles missing values in a DataFrame.
    
    Parameters:
    - df: Input dataframe.
    
    Returns:
    - df_cleaned: DataFrame after standardizing and handling missing values.
    """
    
    # Replace blank strings and strings with only spaces with np.nan
    df.replace(["", " ", "nan", "NaN", "N/A", "NA"], np.nan, inplace=True)
    
    # Replace any string consisting only of whitespace with np.nan
    df.replace(r'^\s*$', np.nan, regex=True, inplace=True)
    
    return df

def plot_missing_values(df):
    # Calculate the percentage of missing values for each column
    missing_percentage = (df.isnull().sum() / len(df)) * 100
    missing_data = pd.DataFrame({'Column': missing_percentage.index, 'Missing Percentage': missing_percentage.values})

    # Filter out columns with 0% missing values to declutter the visualization
    missing_data = missing_data[missing_data['Missing Percentage'] > 0]

    # Create an interactive bar chart
    fig = px.bar(missing_data, x='Column', y='Missing Percentage', title='Percentage of Missing Values by Column',
                 labels={'Column': 'Columns', 'Missing Percentage': 'Percentage (%)'}, height=600)

    return fig.show()

def remove_columns_over_threshold(df, threshold=20):
    """
    Removes columns from the dataframe where missing values exceed the given threshold.
    
    Parameters:
    - df: Input dataframe.
    - threshold: Threshold percentage of missing values to decide column removal.
    
    Returns:
    - df_cleaned: Dataframe after columns exceeding the threshold are removed.
    """
    
    # Calculate the percentage of missing values for each column
    missing_percentage = (df.isnull().sum() / len(df)) * 100
    
    # Identify columns where the missing percentage is over the threshold
    columns_to_drop = missing_percentage[missing_percentage > threshold].index.tolist()
    
    # Drop those columns from the dataframe
    df_cleaned = df.drop(columns=columns_to_drop)
    
    return df_cleaned


def convert_columns_to_list(column_string):
    """
    Convert a string of column names separated by spaces or newlines into a Python list.
    
    Parameters:
        column_string (str): A string containing column names separated by spaces or newlines.
        
    Returns:
        list: A list of column names.
    """
    # Replace newlines with spaces and split the string into a list
    return column_string.replace("\n", " ").split()


import pandas as pd

def impute_missing_values(df, columns, method='N'):
    """
    Impute missing values in the specified columns of the DataFrame using the given method.
    
    Parameters:
        df (pd.DataFrame): The DataFrame to impute missing values in.
        columns (list): A list of column names to impute missing values for.
        method (str): The imputation method ('N', 'zero', 'mean', 'median').
        
    Returns:
        pd.DataFrame: A new DataFrame with missing values imputed.
    """
    # Make a copy of the DataFrame to avoid modifying the original
    df_copy = df.copy()
    
    if method == 'N':
        df_copy[columns] = df_copy[columns].fillna('N')
    elif method == 'zero':
        df_copy[columns] = df_copy[columns].fillna(0)
    elif method == 'mean':
        for col in columns:
            df_copy[col] = df_copy[col].fillna(df[col].mean())
    elif method == 'median':
        for col in columns:
            df_copy[col] = df_copy[col].fillna(df[col].median())
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return df_copy
