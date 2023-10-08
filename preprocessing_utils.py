import pandas as pd
import plotly.graph_objs as go
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.compose import make_column_selector as selector
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns


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

import pandas as pd

def remove_columns_over_threshold(df, threshold=20, exclude_cols=[]):
    """
    Removes columns from the dataframe where missing values exceed the given threshold,
    except for columns in the exclude_cols list.
    
    Parameters:
        df (pd.DataFrame): Input dataframe.
        threshold (float): Threshold percentage of missing values to decide column removal.
        exclude_cols (list): List of columns to exclude from removal.
    
    Returns:
        pd.DataFrame: Dataframe after columns exceeding the threshold are removed.
    """
    
    # Calculate the percentage of missing values for each column
    missing_percentage = (df.isnull().sum() / len(df)) * 100
    
    # Identify columns where the missing percentage is over the threshold
    # and are not in the exclude_cols list
    columns_to_drop = missing_percentage[
        (missing_percentage > threshold) & (~missing_percentage.index.isin(exclude_cols))
    ].index.tolist()
    
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


def map_categorical_values(df, column_mappings):
    """
    Map categorical values in specified columns to more descriptive labels.
    
    Parameters:
        df (pd.DataFrame): The DataFrame to perform the mapping on.
        column_mappings (dict): A dictionary where keys are column names and values
                                are another dictionary for mapping.
        
    Returns:
        pd.DataFrame: A new DataFrame with mapped values.
    """
    # Make a copy of the DataFrame to avoid modifying the original
    df_copy = df.copy()
    
    for col, mapping in column_mappings.items():
        df_copy[col] = df_copy[col].map(mapping).fillna("Unknown")
        
    return df_copy



from sklearn.preprocessing import LabelEncoder
import pandas as pd

def preprocess_dataframe(df, use_frequency_encoding=False, drop_converted_cols=True):
    """
    Preprocesses the dataframe based on the specified steps.
    
    Parameters:
        df (DataFrame): Input dataframe.
        use_frequency_encoding (bool): Whether to use frequency encoding for ZIP, State, and County.
        drop_converted_cols (bool): Whether to drop the original columns after encoding.
        
    Returns:
        DataFrame: Preprocessed dataframe.
    """
    print("Starting data preparation...")
    sleep(1)

    df = df.applymap(lambda x: x.upper() if type(x) == str else x)

    df['PRFL_MINWAGE'] = df['PRFL_MINWAGE'].replace('N', 'UNKNOWN')

    # one-Hot Encoding
    one_hot_cols = [
        'CENSUS_ST', 'AI_COUNTY_NAME', 'ADD_TYPE', 'CENSUS_TRK', 'CONG_DIST',
        'COUNTY_TYPE', 'DON_CHARIT', 'DON_POLIT', 'ETHNIC_INFER',
        'GENDER_MIX', 'GENERATION', 'HOMEOWNER', 'HOMEOWNRNT', 'LANGUAGE',
        'LIFESTAGE_CLUSTER', 'PARTY_CODE', 'PARTY_MIX', 'PRESENCHLD', 'RELIGION',
        'SEX', 'ST_LO_HOUS', 'ST_UP_HOUS', 'STATUS'
    ]
    
    #df = pd.get_dummies(df, columns=one_hot_cols, drop_first=True)

    # Handle PRFL_ columns
    prfl_cols = [col for col in df.columns if col.startswith('PRFL_')]
    df = pd.get_dummies(df, columns=prfl_cols, drop_first=True)

    vtr_cols = [col for col in df.columns if col.startswith('VTR_')]
    df = pd.get_dummies(df, columns=vtr_cols, drop_first=True)

    # splitting Columns
    split_cols = ['TOD_PRES_DIFF_2016', 'TOD_PRES_DIFF_2016_PREC', 'TOD_PRES_DIFF_2020_PREC']
    for col in split_cols:
        df[col + '_num'] = df[col].str.extract('(\d+)').astype('float')
        df[col + '_party'] = df[col].str.extract('([RD])')

    new_one_hot_cols = [col + '_party' for col in split_cols]
    one_hot_cols.extend(new_one_hot_cols)

    df = pd.get_dummies(df, columns=one_hot_cols, drop_first=True)

    #convert to Int
    int_cols = ['VOTER_CNT', 'TRAIL_CNT', 'CNS_MEDINC', 'HH_SIZE', 'LENGTH_RES', 'PERSONS_HH']
    df[int_cols] = df[int_cols].astype(int)

    # label Encoding
    label_cols = ['CREDRATE', 'EDUCATION', 'HH_SIZE', 'HOMEMKTVAL', 'INCOMESTHH', 'NETWORTH']
    for col in label_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    # drop Columns
    drop_cols = ['ETHNICCODE']
    df.drop(columns=drop_cols, inplace=True)
    
    # frequency Encoding for ZIP, State, and County
    if use_frequency_encoding:
        freq_cols = ['ZIP', 'STATE', 'COUNTY_ST']
        for col in freq_cols:
            freq_map = df[col].value_counts(normalize=True)
            df[col + '_freq'] = df[col].map(freq_map)

    # Drop the original columns if specified
    if drop_converted_cols:
        all_converted_cols = one_hot_cols + int_cols + label_cols + freq_cols + prfl_cols + vtr_cols + split_cols
        all_converted_cols = [col for col in all_converted_cols if col in df.columns]
        df.drop(columns=all_converted_cols, inplace=True)
            
    return df


def outlier_detection_pipeline(survey_df, remove_by='none', threshold=1):
    original_df = survey_df.copy()
    total_original_rows = len(original_df)
    
    # Automatically select numerical and categorical columns
    numerical_cols = selector(dtype_include=np.number)(original_df)
    categorical_cols = selector(dtype_exclude=np.number)(original_df)
    
    # Convert all categorical columns to string type
    original_df[categorical_cols] = original_df[categorical_cols].astype(str)
    
    # Data Preprocessing
    numerical_transformer = RobustScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )
    
    # Apply preprocessing
    X = preprocessor.fit_transform(original_df)
    
    # Initialize models
    models = {
        'IsolationForest': IsolationForest(contamination=0.1),
        'LocalOutlierFactor': LocalOutlierFactor(contamination=0.1),
        'OneClassSVM': OneClassSVM(nu=0.1)
    }
    
    # Results storage
    results = {}
    columns_with_outliers = set()
    
    # Outlier detection
    for name, model in models.items():
        y_pred = model.fit_predict(X)
        y_pred = [1 if x == -1 else 0 for x in y_pred]
        results[name] = y_pred
        
        if sum(y_pred) > 0:
            columns_with_outliers.add(name)
    
    # Calculate consensus score
    original_df['consensus'] = np.sum(list(results.values()), axis=0)
    
    # Remove outliers based on the given condition
    if remove_by == 'consensus':
        filtered_df = original_df[original_df['consensus'] < threshold]
    elif remove_by in models.keys():
        filtered_df = original_df[np.array(results[remove_by]) == 0]
    else:
        filtered_df = original_df.copy()
    
    # Calculate and print percentage of removed values
    initial_count = len(original_df)
    final_count = len(filtered_df)
    removed_count = initial_count - final_count
    
    if removed_count > 0:
        removed_percentage = 100 * removed_count / initial_count
        print(f"Removed {removed_percentage:.2f}% of values from all columns.")
    
    # Calculate and print the total percentage of removed rows
    total_removed_percentage = 100 * removed_count / total_original_rows
    print(f"Total: Removed {total_removed_percentage:.2f}% of the original data points.")
    
    print(f"Columns with outliers detected: {', '.join(columns_with_outliers)}")
    
    return filtered_df.drop(columns=['consensus'], errors='ignore'), results


def plot_outliers(df, column_names, target_col):
    for column_name in column_names:
        plt.figure(figsize=(10, 6))

        # Check if the column_name is a valid column in the DataFrame
        if column_name not in df.columns:
            print(f'{column_name} is not a valid column in the DataFrame')
            continue

        # Automatically detect the type of the column
        if np.issubdtype(df[column_name].dtype, np.number):
            sns.boxplot(x=target_col, y=df[column_name], data=df)
            plt.title(f'Boxplot of {column_name}')
            plt.xlabel(target_col)
            plt.ylabel(column_name)

        else:
            sns.countplot(data=df, x=column_name, hue=target_col, order=df[column_name].value_counts().index)
            plt.title(f'Bar Plot of {column_name}')
            plt.xlabel(column_name)
            plt.ylabel('Frequency')

        plt.show()


def plot_feature_distribution(df, feature_columns, target_column, use_kde=False):
    """
    Plot the distribution of one or multiple features conditioned on the target column.

    Parameters:
        df (DataFrame): The DataFrame containing the data.
        feature_columns (list or str): The name(s) of the feature column(s).
        target_column (str): The name of the target column.
        use_kde (bool): Whether to use KDE for numerical features.

    Returns:
        None
    """
    if not isinstance(feature_columns, list):
        feature_columns = [feature_columns]
        
    for feature_column in feature_columns:
        if feature_column not in df.columns:
            print(f"Feature column {feature_column} not found in DataFrame.")
            continue

        if target_column not in df.columns:
            print(f"Target column {target_column} not found in DataFrame.")
            return

        is_numeric = np.issubdtype(df[feature_column].dtype, np.number)

        plt.figure(figsize=(12, 6))

        if is_numeric:
            if use_kde:
                for label in df[target_column].unique():
                    sns.kdeplot(df[df[target_column] == label][feature_column], label=label)
                plt.ylabel('Density')
            else:
                for label in df[target_column].unique():
                    sns.histplot(df[df[target_column] == label][feature_column], label=label, element="step", stat="density", common_norm=False)
                plt.ylabel('Density')
        else:
            sns.countplot(data=df, x=feature_column, hue=target_column)
            plt.ylabel('Count')

        plt.title(f'{feature_column} by {target_column}')
        plt.xlabel(f'{feature_column}')
        plt.legend(title=target_column, loc='upper right')
        plt.show()

def plot_multiple_feature_distributions(df, feature_columns, target_column, use_kde=False):
    for feature_column in feature_columns:
        plot_feature_distribution(df, feature_column, target_column, use_kde)


from scipy.stats import skew
import numpy as np

def apply_skew_transformations(df, columns):
    transformed_df = df.copy()
    for col in columns:
        skewness = skew(transformed_df[col].dropna())
        # Check for skewness
        if skewness > 1:  # Right-skewed
            transformed_df[col] = np.log1p(transformed_df[col])
        elif skewness < -1:  # Left-skewed
            # Example: You can use square transformation for left-skewed data
            transformed_df[col] = np.square(transformed_df[col])
        else:  
            pass
    return transformed_df