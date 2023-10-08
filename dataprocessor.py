import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from time import sleep
import json

class DataPreprocessor:
    def __init__(self, df, config_path='/Volumes/DeepLearner/MADS/Milestone2_Party_prediction/milestone2_MADS/preprocessing_config.json'):
        self.df = df.copy()
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
        #self.replace_candidates()
        #self.drop_small_categories()
        self.engineer_features()
        
        # Remove columns over the threshold
        self.remove_columns_over_threshold(threshold=20)
        
        # Any other preprocessing steps
        if not skip_preprocess_dataframe:
            self.preprocess_dataframe()
            self.has_preprocessed = True
        
        return self.df

    def standardize_missing_values(self):
        print("Standardizing missing values...")
        """Standardizes and handles missing values in the DataFrame."""

        # Replace blank strings and strings with only spaces with np.nan
        self.df.replace(["", " ", "nan", "NaN", "N/A", "NA"], np.nan, inplace=True)
        # Replace any string consisting only of whitespace with np.nan
        self.df.replace(r'^\s*$', np.nan, regex=True, inplace=True)
        self.df = self.df.applymap(lambda x: x.upper() if type(x) == str else x)
        self.df['PRFL_MINWAGE'] = self.df['PRFL_MINWAGE'].replace('N', 'UNKNOWN')
        self.df.drop(columns=["SURVEY_TYPE"], inplace=True)
        self.df.drop(columns=["RECORD_ID"], inplace=True)
        # making sure HH_SIZE is an integer
        

    def engineer_features(self):
        """Engineer new features in the DataFrame."""
        print("Engineering new features...")
        
        # R_donor
       # self.df['R_DONOR'] = ((self.df['FUND_POLIT'] == 'R') | (self.df['DON_POLCONS'] > '')).astype(int)
        
        # D_donor
       # self.df['D_DONOR'] = ((self.df['FUND_POLIT'] == 'D') | (self.df['DON_POLLIB'] > '')).astype(int)
        
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

    def map_categorical_values(self):
        print("Mapping categorical values...")
        """Map categorical values in specified columns to more descriptive labels."""
        column_mappings = self.config.get('column_mappings', {})
        for col, mapping in column_mappings.items():
            self.df[col] = self.df[col].map(mapping).fillna("Unknown")
            #self.df[col] = self.df[col].astype(str).map(mapping).fillna("Unknown")

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


    def remove_columns_over_threshold(self, threshold=20):
        print("Removing columns over the threshold...")
        """Removes columns from the DataFrame where missing values exceed the given threshold."""
        exclude_cols = self.config.get('cols_to_keep_below_20', [])
        missing_percentage = (self.df.isnull().sum() / len(self.df)) * 100
        columns_to_drop = missing_percentage[
            (missing_percentage > threshold) & (~missing_percentage.index.isin(exclude_cols))
        ].index.tolist()
        self.df.drop(columns=columns_to_drop, inplace=True)
        
    
    def preprocess_dataframe(self, use_frequency_encoding=True, drop_converted_cols=True):
        """Preprocesses the DataFrame based on the specified steps."""
        print("Starting data preparation...")
        sleep(1)
        

        #self.df['PRFL_MINWAGE'] = self.df['PRFL_MINWAGE'].replace('N', 'UNKNOWN')

        # one-Hot Encoding
      #  one_hot_cols = [
      #      'CENSUS_ST', 'AI_COUNTY_NAME', 'ADD_TYPE', 'CENSUS_TRK', 'CONG_DIST',
      #      'COUNTY_TYPE', 'DON_CHARIT', 'DON_POLIT', 'ETHNIC_INFER',
      #      'GENDER_MIX', 'GENERATION', 'HOMEOWNER', 'HOMEOWNRNT', 'LANGUAGE',
      #      'LIFESTAGE_CLUSTER', 'PARTY_CODE', 'PARTY_MIX', 'PRESENCHLD', 'RELIGION',
      #      'SEX', 'ST_LO_HOUS', 'ST_UP_HOUS', 'STATUS'
      #  ]
        one_hot_cols = []
        #self.df = pd.get_dummies(self.df, columns=one_hot_cols, drop_first=True)

        # Handle PRFL_ columns
        prfl_cols = [col for col in self.df.columns if col.startswith('PRFL_')]

        # drop prfl_ columns
        self.df.drop(columns=prfl_cols, inplace=True)

        #self.df = pd.get_dummies(self.df, columns=prfl_cols, drop_first=True)

        vtr_cols = [col for col in self.df.columns if col.startswith('VTR_')]
        self.df = pd.get_dummies(self.df, columns=vtr_cols, drop_first=True)

        # splitting Columns
        split_cols = ['TOD_PRES_DIFF_2016', 'TOD_PRES_DIFF_2016_PREC', 'TOD_PRES_DIFF_2020_PREC']
        for col in split_cols:
            self.df[col + '_num'] = self.df[col].str.extract('(\d+)').astype('float')
            self.df[col + '_party'] = self.df[col].str.extract('([RD])')

        new_one_hot_cols = [col + '_party' for col in split_cols]
        one_hot_cols.extend(new_one_hot_cols)

        self.df = pd.get_dummies(self.df, columns=one_hot_cols, drop_first=True)

        #convert to Int
        int_cols = ['VOTER_CNT', 'TRAIL_CNT', 'CNS_MEDINC', 'HH_SIZE', 'LENGTH_RES', 'PERSONS_HH']
        self.df[int_cols] = self.df[int_cols].astype(int)

        # label Encoding
        label_cols = ['CREDRATE', 'EDUCATION', 'HH_SIZE', 'HOMEMKTVAL', 'INCOMESTHH', 'NETWORTH', 'STATE_CD',
                      'STATE_LOWER_HOUSE', 'STATE_UPPER_HOUSE', 'CENSUS_TRACT','CENSUS_ST', 'AI_COUNTY_NAME', 'ADD_TYPE', 'CENSUS_TRK', 'CONG_DIST',
            'COUNTY_TYPE', 'DON_CHARIT', 'DON_POLIT', 'ETHNIC_INFER',
            'GENDER_MIX', 'GENERATION', 'HOMEOWNER', 'HOMEOWNRNT', 'LANGUAGE',
            'LIFESTAGE_CLUSTER', 'PARTY_CODE', 'PARTY_MIX', 'PRESENCHLD', 'RELIGION',
            'SEX', 'ST_LO_HOUS', 'ST_UP_HOUS', 'STATUS']
        
        le = LabelEncoder()
        for col in label_cols:
            #if self.df[col].dtype != 'object':
                self.df[col] = self.df[col].astype(str)
                self.df[col] = le.fit_transform(self.df[col])

        # drop Columns
        drop_cols = ['ETHNICCODE']
        self.df.drop(columns=drop_cols, inplace=True)
        freq_cols = []  # Initialize freq_cols to an empty list
        if use_frequency_encoding:
            freq_cols = ['ZIP', 'STATE', 'COUNTY_ST']
            for col in freq_cols:
                freq_map = self.df[col].value_counts(normalize=True)
                self.df[col + '_freq'] = self.df[col].map(freq_map)
        else:
        
            self.df = pd.get_dummies(self.df, columns=['ZIP', 'STATE', 'COUNTY_ST'], drop_first=True)

        # Drop the original columns if specified
        if drop_converted_cols:
            all_converted_cols = one_hot_cols + int_cols + label_cols + freq_cols + prfl_cols + vtr_cols + split_cols
            all_converted_cols = [col for col in all_converted_cols if col in self.df.columns]
            self.df.drop(columns=all_converted_cols, inplace=True)


        print("Viola, you have a cleaned, preprocessed DataFrame!")
        return self.df
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