import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from time import sleep
import json

def split_and_transform(series):
    # Use str.extract to get the numerical part and the letter part
    extracted = series.str.extract(r'(\d+)([RD])')
    
    # Convert the numerical part to integers, handle NaNs gracefully
    decimals = pd.to_numeric(extracted[0], errors='coerce') / 100.0
    
    # Convert the letter part to categorical, handle NaNs gracefully
    categories = extracted[1].astype('category')
    
    return decimals, categories

class DataPreprocessor:
    def __init__(self, df, is_training=True,config_path='/Volumes/DeepLearner/MADS/Milestone2_Party_prediction/milestone2_MADS/preprocessing_config.json'):
        self.df = df.copy()
        self.is_training = is_training
        self.config = self.load_config(config_path)
        print("DataPreprocessor initialized.")
        self.has_preprocessed = False
    
    @staticmethod
    def load_config(config_path):
        """Load the JSON configuration file and convert keys to appropriate types."""
        print(f"Loading config from {config_path}")
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Convert keys to integers where applicable
        for col, mapping in config.get('column_mappings', {}).items():
            new_mapping = {int(k) if k.isdigit() else k: v for k, v in mapping.items()}
            config['column_mappings'][col] = new_mapping
        
        return config
        
    def apply_config(self, skip_preprocess_dataframe=False):
        """Apply all the settings from the loaded JSON configuration."""
        print("Bleep bloop...")
        print("Applying config...")

        if self.is_training:
            self.replace_candidates()
            self.drop_small_categories()

        self.standardize_missing_values()
        
        # Apply imputation strategies
        for config_key, impute_settings in self.config.items():
            if config_key in ['columns_NO', 'columns_median', 'columns_zero']:
                self.impute_missing_values(impute_settings['cols'], impute_settings['impute_strategy'])
        
        # Apply categorical mappings
        self.map_categorical_values()
        self.map_party_code()
        self.engineer_features()
        self.preprocess_turnouts()
        
        # Remove columns over the threshold
        #self.remove_columns_over_threshold(threshold=20)
        
        # Any other preprocessing steps
        if not skip_preprocess_dataframe:
            self.preprocess_dataframe()
            self.has_preprocessed = True
        
        return self.df
    
    def preprocess_turnouts(self):
        print("Preprocessing turnout columns...")
        
        # Handle 2016 columns
        for col in ['TOD_PRES_DIFF_2016_PREC', 'TOD_PRES_DIFF_2016']:
            if col in self.df.columns:
                decimals, categories = split_and_transform(self.df[col])
                self.df[f"{col}_decimal"] = decimals
                self.df[f"{col}_party"] = categories
                # Drop the original column
                self.df.drop(columns=[col], inplace=True)
        
        # Handle 2020 column
        col_2020 = 'TOD_PRES_DIFF_2020_PREC'
        if col_2020 in self.df.columns:
            decimals, categories = split_and_transform(self.df[col_2020])
            self.df[f"{col_2020}_decimal"] = decimals
            self.df[f"{col_2020}_party"] = categories
            # Drop the original column
            self.df.drop(columns=[col_2020], inplace=True)

        print("Voila! cleaning chores have finished, time to explore!")



    def standardize_missing_values(self):
        print("Standardizing missing values...")
        """Standardizes and handles missing values in the DataFrame."""

        # Replace blank strings and strings with only spaces with np.nan
        self.df.replace(["", " ", "nan", "NaN", "N/A", "NA"], np.nan, inplace=True)
        # Replace any string consisting only of whitespace with np.nan
        self.df.replace(r'^\s*$', np.nan, regex=True, inplace=True)
        self.df = self.df.applymap(lambda x: x.upper() if type(x) == str else x)
        self.df['PRFL_MINWAGE'] = self.df['PRFL_MINWAGE'].replace('N', 'UNKNOWN')
        
        if self.is_training:
            self.df.drop(columns=["SURVEY_TYPE"], inplace=True)
        
        self.df.drop(columns=["RECORD_ID"], inplace=True)
        # making sure HH_SIZE is an integer
        

    def engineer_features(self):
        """Engineer new features in the DataFrame."""
        print("Engineering new features...")
        
        # R_donor
        self.df['R_DONOR'] = ((self.df['FUND_POLIT'] == 'R') | (self.df['DON_POLCONS'] > '')).astype(int)
        
        # D_donor
        self.df['D_DONOR'] = ((self.df['FUND_POLIT'] == 'D') | (self.df['DON_POLLIB'] > '')).astype(int)
        
        # Voted_R_Election
        vtr_columns_r = [col for col in self.df.columns if col.startswith('VTR_')]
        condition_r = self.df[vtr_columns_r].isin(['R', 'P', 'X']).any(axis=1)
        self.df['VOTED_R_ELECTION'] = condition_r.astype(int)
        
        # Voted_D_Election
        vtr_columns_d = [col for col in self.df.columns if col.startswith('VTR_')]
        condition_d = self.df[vtr_columns_d].isin(['D', 'M', 'Z']).any(axis=1)
        self.df['VOTED_D_ELECTION'] = condition_d.astype(int)
        
        # Additional engineered features for unique county, congressional, state upper house & state lower house columns
        self.df['STATE_COUNTY_FIPS'] = self.df['CENSUS_ST'].astype(str) + self.df['COUNTY_ST'].astype(str)
        self.df['STATE_CD'] = self.df['CENSUS_ST'].astype(str) + self.df['CONG_DIST'].astype(str)
        self.df['STATE_LOWER_HOUSE'] = self.df['CENSUS_ST'].astype(str) + self.df['ST_LO_HOUS'].astype(str)
        self.df['STATE_UPPER_HOUSE'] = self.df['CENSUS_ST'].astype(str) + self.df['ST_UP_HOUS'].astype(str)
        self.df['CENSUS_TRACT'] = self.df['CENSUS_ST'].astype(str) + self.df['COUNTY_ST'].astype(str)+ self.df['CENSUS_TRK'].astype(str)

    def impute_missing_values(self, columns, method='N'):
        print("Imputing missing values...")
        """Impute missing values in the specified columns of the DataFrame using the given method."""
        if method == 'N':
            self.df[columns] = self.df[columns].fillna('N')
        elif method == 'zero':
            self.df[columns] = self.df[columns].fillna(0)
        elif method == 'mean':
            for col in columns:
                self.df[col] = self.df[col].fillna(self.df[col].mean())
        elif method == 'median':
            for col in columns:
                self.df[col] = self.df[col].fillna(self.df[col].median())
        else:
            raise ValueError(f"Unknown method: {method}")
        print("Hang in there, still sweeping the floor...")

    def map_categorical_values(self):
        print("Mapping categorical values...")
        """Map categorical values in specified columns to more descriptive labels."""
        column_mappings = self.config.get('column_mappings', {})
        for col, mapping in column_mappings.items():
            self.df[col] = self.df[col].map(mapping).fillna("Unknown")
            #self.df[col] = self.df[col].astype(str).map(mapping).fillna("Unknown")

        print("Oh smokes, forgot to make the bed...")

    def map_party_code(self):
        """Map and consolidate the party codes to simpler categories based on the config."""
        if 'PARTY_CODE' in self.config:
            party_mapping = self.config['PARTY_CODE']
            self.df['PARTY_CODE'] = self.df['PARTY_CODE'].map(party_mapping).fillna('Other')

    def replace_candidates(self):
        # Replacing specified candidates with broader categories
        candidates_to_replace1 = self.config['candidates_to_replace1']
        self.df['Q1_Candidate'] = self.df['Q1_Candidate'].replace(candidates_to_replace1, "Other GOP")
        
        candidates_to_replace2 = self.config['candidates_to_replace2']
        self.df['Q1_Candidate'] = self.df['Q1_Candidate'].replace(candidates_to_replace2, "Other DEM")
        
        candidates_to_replace3 = self.config['candidates_to_replace3']
        self.df['Q1_Candidate'] = self.df['Q1_Candidate'].replace(candidates_to_replace3, "Other/Undecided")

    def drop_small_categories(self):
        # Drop the 2 smallest categories to focus on the top 3
        self.df = self.df[~self.df['Q1_Candidate'].isin(['Other/Undecided', 'Other DEM'])]


    def remove_columns_over_threshold(self, threshold=30):
        print("Removing columns over the threshold...")
        """Removes columns from the DataFrame where missing values exceed the given threshold."""
        exclude_cols = self.config.get('cols_to_keep_below_20', [])
        missing_percentage = (self.df.isnull().sum() / len(self.df)) * 100
        columns_to_drop = missing_percentage[
            (missing_percentage > threshold) & (~missing_percentage.index.isin(exclude_cols))
        ].index.tolist()
        self.df.drop(columns=columns_to_drop, inplace=True)
        
    def run_preprocessing_pipeline(self, skip_preprocess_dataframe=False, drop_converted_cols=True, use_frequency_encoding=False):
            """Run the entire preprocessing pipeline in a specific order."""
            self.apply_config(skip_preprocess_dataframe=skip_preprocess_dataframe)
            self.df.drop(columns=['Unnamed: 0', 'RECORD_ID'], errors='ignore')
            self.df['HH_SIZE'] = self.df['HH_SIZE'].astype(int)
            self.df["VOTER_CNT"] = self.df["VOTER_CNT"].astype(int)
            
            if not skip_preprocess_dataframe and not self.has_preprocessed:
                self.preprocess_dataframe(drop_converted_cols=drop_converted_cols, use_frequency_encoding=use_frequency_encoding)
                self.has_preprocessed = True  # Set flag to True
                
            return self.df