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


def set_inferred_party(row):
   not_in_list = ['D', 'R', 'M', 'P', 'X', 'Z']
   rpx = ['R', 'P', 'X']
   dmz = ['D', 'M', 'Z']
   if row['PARTY_CODE'] in ['N', 'U']:
       if (
           (row['FUND_POLIT'] == 'R' or row['DON_POLCONS'] or row['PRFL_HEALTHCARE_REFORM'] == '2' or
            row['PRFL_2NDAMEND'] == 'Y' or row['PRFL_CHOICELIFE'] == '1') and row['FUND_POLIT'] != 'D' and
            not row['DON_POLLIB'] and all(row[col] not in not_in_list for col in
            ['VTR_PRI' + "{:02}".format(i) for i in range(22, 2, -1)] + ['VTR_PPP' + "{:02}".format(i) for i in [20, 16, 12, 8, 4, 0]])
       ) or (
           sum(1 for col in ['VTR_PRI' + "{:02}".format(i) for i in range(22, 2, -1)] + ['VTR_PPP' + "{:02}".format(i) for i in [20, 16, 12, 8, 4, 0]]
            if row[col] in rpx) > sum(1 for col in ['VTR_PRI' + "{:02}".format(i) for i in range(22, 2, -1)] +
            ['VTR_PPP' + "{:02}".format(i) for i in [20, 16, 12, 8, 4, 0]] if row[col] in dmz)
       ):
           return 'S'
       elif (
           (row['FUND_POLIT'] == 'D' or row['DON_POLLIB'] or row['PRFL_HEALTHCARE_REFORM'] == '1' or
            row['PRFL_CHOICELIFE'] == '2') and row['FUND_POLIT'] != 'r' and not row['DON_POLCONS'] and
            all(row[col] not in not_in_list for col in ['VTR_PRI' + "{:02}".format(i) for i in range(22, 2, -1)] +
            ['VTR_PPP' + "{:02}".format(i) for i in [20, 16, 12, 8, 4, 0]])
       ) or (
           sum(1 for col in ['VTR_PRI' + "{:02}".format(i) for i in range(22, 2, -1)] + ['VTR_PPP' + "{:02}".format(i) for i in [20, 16, 12, 8, 4, 0]]
            if row[col] in rpx) < sum(1 for col in ['VTR_PRI' + "{:02}".format(i) for i in range(22, 2, -1)] +
            ['VTR_PPP' + "{:02}".format(i) for i in [20, 16, 12, 8, 4, 0]] if row[col] in dmz)
       ):
           return 'E'
   return row['PARTY_CODE'] 



from itertools import groupby
from itertools import combinations

def create_interaction_terms(df, combinations, interaction_type='cat'):
    for combination in combinations:
        # Create an interaction term name dynamically based on the number of features involved
        interaction_term_name = "_".join(combination) + "_interaction"

        if interaction_type == 'cat':
            # Create the categorical interaction term
            df[interaction_term_name] = df[list(combination)].astype(str).apply(lambda x: "_".join(x), axis=1)
            
        elif interaction_type == 'num':
            # Create the numerical interaction term
            df[interaction_term_name] = df[list(combination)].apply(lambda x: x.prod(), axis=1)

    return df

def feature_engineering_voting_data(survey_df, columns_to_use, interaction_type='cat', extend_columns=True):

    # Define the vote types
    democrat_votes = ['D', 'M', 'Z']
    republican_votes = ['R', 'P', 'X']
    early_votes = ['E', 'M', 'P']
    absentee_votes = ['A', 'Z', 'X']

    # Interaction terms for 2020
   # survey_df['interaction_mult_2020'] = survey_df['CNSUS_PCTW'] * survey_df['TOD_PRES_R_2020_PREC']
    survey_df['interaction_div_2020'] = survey_df['CNSUS_PCTW'] / survey_df['TOD_PRES_R_2020_PREC']
   # survey_df['interaction_add_2020'] = survey_df['CNSUS_PCTW'] + survey_df['TOD_PRES_R_2020_PREC']

    # Interaction terms for 2016
  #  survey_df['interaction_mult_2016'] = survey_df['CNSUS_PCTW'] * survey_df['TOD_PRES_R_2016_PREC']
    survey_df['interaction_div_2016'] = survey_df['CNSUS_PCTW'] / survey_df['TOD_PRES_R_2016_PREC']
  #  survey_df['interaction_add_2016'] = survey_df['CNSUS_PCTW'] + survey_df['TOD_PRES_R_2016_PREC']

    # Combined interaction terms
   # survey_df['interaction_mult_combined'] = (survey_df['TOD_PRES_R_2016_PREC'] + survey_df['TOD_PRES_R_2020_PREC']) * survey_df['CNSUS_PCTW']

    # Delta interaction term
    survey_df['interaction_mult_delta'] = (survey_df['TOD_PRES_R_2020_PREC'] - survey_df['TOD_PRES_R_2016_PREC']) * survey_df['CNSUS_PCTW']

    # Extend the list of columns to use in the model
    if extend_columns:
        columns_to_use.extend([
            'interaction_div_2020',
            'interaction_div_2016',  
            'interaction_mult_delta',
        #  'interaction_mult_2020',
        #  'interaction_mult_2016'
        ])

    # Interaction terms for 2020 with respect to Democratic turnout
   # survey_df['interaction_mult_D_2020'] = survey_df['CNSUS_PCTB'] * survey_df['TOD_PRES_D_2020_PREC']
    survey_df['interaction_div_D_2020'] = survey_df['CNSUS_PCTB'] / survey_df['TOD_PRES_D_2020_PREC']
   # survey_df['interaction_add_D_2020'] = survey_df['CNSUS_PCTB'] + survey_df['TOD_PRES_D_2020_PREC']

    # Interaction terms for 2016 with respect to Democratic turnout
   # survey_df['interaction_mult_D_2016'] = survey_df['CNSUS_PCTB'] * survey_df['TOD_PRES_D_2016_PREC']
    survey_df['interaction_div_D_2016'] = survey_df['CNSUS_PCTB'] / survey_df['TOD_PRES_D_2016_PREC']
   # survey_df['interaction_add_D_2016'] = survey_df['CNSUS_PCTB'] + survey_df['TOD_PRES_D_2016_PREC']

    # Combined interaction terms for Democratic turnout
    #survey_df['interaction_mult_D_combined'] = (survey_df['TOD_PRES_D_2016_PREC'] + survey_df['TOD_PRES_D_2020_PREC']) * survey_df['CNSUS_PCTB']

    # Delta interaction term for Democratic turnout
    survey_df['interaction_mult_D_delta'] = (survey_df['TOD_PRES_D_2020_PREC'] - survey_df['TOD_PRES_D_2016_PREC']) * survey_df['CNSUS_PCTB']

    # Extend the list of columns to use in the model
    if extend_columns:
        columns_to_use.extend([
            'interaction_div_D_2020',
            'interaction_div_D_2016', 
            'interaction_mult_D_delta'
        ])

    survey_df['Years_Voted_Democrat'] = survey_df[['VTR_PPP04', 'VTR_PPP08', 'VTR_PPP12', 'VTR_PPP16', 'VTR_PPP20']].apply(lambda x: sum(x.isin(['D', 'M', 'Z'])), axis=1)
    survey_df['Years_Voted_Republican'] = survey_df[['VTR_PPP04', 'VTR_PPP08', 'VTR_PPP12', 'VTR_PPP16', 'VTR_PPP20']].apply(lambda x: sum(x.isin(['R', 'P', 'X'])), axis=1)

    if extend_columns:
        columns_to_use.extend(['Years_Voted_Democrat', 'Years_Voted_Republican'])


    survey_df['Years_Absentee_Democrat'] = survey_df[['VTR_PPP04', 'VTR_PPP08', 'VTR_PPP12', 'VTR_PPP16', 'VTR_PPP20']].apply(lambda x: sum(x.isin(['Z'])), axis=1)
    survey_df['Years_Early_Democrat'] = survey_df[['VTR_PPP04', 'VTR_PPP08', 'VTR_PPP12', 'VTR_PPP16', 'VTR_PPP20']].apply(lambda x: sum(x.isin(['M'])), axis=1)
    survey_df['Years_Absentee_Republican'] = survey_df[['VTR_PPP04', 'VTR_PPP08', 'VTR_PPP12', 'VTR_PPP16', 'VTR_PPP20']].apply(lambda x: sum(x.isin(['X'])), axis=1)
    survey_df['Years_Early_Republican'] = survey_df[['VTR_PPP04', 'VTR_PPP08', 'VTR_PPP12', 'VTR_PPP16', 'VTR_PPP20']].apply(lambda x: sum(x.isin(['P'])), axis=1)

    # 'Years_Absentee_Democrat', 'Years_Absentee_Republican',
    if extend_columns:
        columns_to_use.extend(['Years_Early_Democrat', 'Years_Early_Republican', 'Years_Absentee_Democrat', 'Years_Absentee_Republican'])

    primary_columns = ['VTR_PRI06', 'VTR_PRI10', 'VTR_PRI14', 'VTR_PRI16', 'VTR_PRI18', 'VTR_PRI20', 'VTR_PRI21', 'VTR_PRI22']
    survey_df['Years_Voted_Democrat_Primaries'] = survey_df[primary_columns].apply(lambda x: sum(x.isin(['D', 'M', 'Z'])), axis=1)
    survey_df['Years_Voted_Republican_Primaries'] = survey_df[primary_columns].apply(lambda x: sum(x.isin(['R', 'P', 'X'])), axis=1)
    survey_df['Years_Early_Democrat_Primaries'] = survey_df[primary_columns].apply(lambda x: sum(x.isin(['M'])), axis=1)
    survey_df['Years_Early_Republican_Primaries'] = survey_df[primary_columns].apply(lambda x: sum(x.isin(['P'])), axis=1)

    if extend_columns:
        columns_to_use.extend(['Years_Voted_Democrat_Primaries', 'Years_Voted_Republican_Primaries', 'Years_Early_Democrat_Primaries', 'Years_Early_Republican_Primaries'])

  #  for col in primary_columns:
  #      columns_to_use.remove(col)
    
    # function to count specific vote types
    def count_votes(vote_counts, vote_types):
        return sum(vote_counts.get(vote_type, 0) for vote_type in vote_types)

    # function to count longest streak for a party
    def longest_streak(votes, party_votes):
        streaks = [sum(1 for _ in g) for k, g in groupby(votes) if k in party_votes]
        return max(streaks) if streaks else 0

    # count early and absentee votes
    survey_df['count_Early'] = survey_df.filter(like='VTR_GEN').apply(lambda row: count_votes(row.value_counts(), early_votes), axis=1)
    survey_df['count_Absentee'] = survey_df.filter(like='VTR_GEN').apply(lambda row: count_votes(row.value_counts(), absentee_votes), axis=1)
    
    if extend_columns:
        columns_to_use.extend(['count_Early', 'count_Absentee'])



    for prefix in ['VTR_GEN', 'VTR_PPP', 'VTR_PRI']:
        survey_df[f'count_D_{prefix}'] = survey_df.filter(like=prefix).apply(lambda row: count_votes(row.value_counts(), democrat_votes), axis=1)
        survey_df[f'count_R_{prefix}'] = survey_df.filter(like=prefix).apply(lambda row: count_votes(row.value_counts(), republican_votes), axis=1)
        
        if extend_columns:
            columns_to_use.extend([f'count_R_{prefix}'])
            columns_to_use.extend([f'count_D_{prefix}'])
            #[f'count_D_{prefix}', f'count_R_{prefix}'])

        # Count longest streak of consistent voting for each party
        survey_df[f'longest_streak_D_{prefix}'] = survey_df.filter(like=prefix).apply(lambda row: longest_streak(row.tolist(), democrat_votes), axis=1)
        survey_df[f'longest_streak_R_{prefix}'] = survey_df.filter(like=prefix).apply(lambda row: longest_streak(row.tolist(), republican_votes), axis=1)
        
        if extend_columns:
            columns_to_use.extend([f'longest_streak_D_{prefix}', f'longest_streak_R_{prefix}'])

    #survey_df['recent_party_2022'] = survey_df['VTR_GEN22'].apply(lambda x: 'D' if x in democrat_votes else ('R' if x in republican_votes else 'Other'))
    #survey_df['recent_party_2020'] = survey_df['VTR_GEN18'].apply(lambda x: 'D' if x in democrat_votes else ('R' if x in republican_votes else 'Other'))
   # columns_to_use.extend(['recent_party_2022'])

    survey_df['total_votes'] = survey_df.filter(like='VTR_').apply(lambda row: sum(row != 'N'), axis=1)
    
    if extend_columns:
        columns_to_use.append('total_votes')
    #columns_to_use.remove("count_R_VTR_GEN")

    #feature_combinations = [
    #['PRFL_LIBERAL_NEWS', 'PRFL_IMMIGRATION_REFORM'] ]

    #survey_df = create_interaction_terms(survey_df, feature_combinations, interaction_type)
    
    # Add the new interaction term columns to your columns_to_use list
    #new_columns = ["_".join(combination) + "_interaction" for combination in feature_combinations]
   # columns_to_use.extend(new_columns)

    # drop count_D_VTR_GEN
   # columns_to_use.remove('count_D_VTR_GEN')

    
    return survey_df, columns_to_use
