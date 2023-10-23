from preprocessing_utils import *
from dataprocessor import DataPreprocessor

import warnings
import pandas as pd
# suppress warning messages
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_selector as selector
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,confusion_matrix, classification_report

from imblearn.over_sampling import SMOTE


from xgboost import XGBClassifier
#from lightgbm import LGBMClassifier

from tqdm import tqdm
from time import sleep
from category_encoders import TargetEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
import json
from sklearn.model_selection import GridSearchCV, cross_val_score
import json
from tqdm import tqdm
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold
from itertools import product
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.utils import shuffle
from sklearn.preprocessing import OrdinalEncoder
import os





class ModelTrainingPipeline:
    def __init__(self, df, target_col, model_type='All', feature_importance=False, 
                 scale_data=False, evaluate=False, use_grid_search=False, 
                 feature_engineering=None, add_stats=False,save_intermediate_data=True):
        
        self.df = df
        self.target_col = target_col
        self.save_intermediate_data = save_intermediate_data
        self.intermediate_data = {}
        self.model_type = model_type
        self.feature_importance = feature_importance
        self.scale_data = scale_data
        self.should_evaluate_models = evaluate
        self.use_grid_search = use_grid_search
        self.feature_engineering = feature_engineering
        self.add_stats = add_stats
        self.models = {}
        self.evaluation_results = {}
        self.classification_reports_dict = {}
        self.best_params_dict = {}
        self.trained_models = []
        self.loaded_params = {}
        self.hyperparameters = None
        self.log_file = "experiment_logs.csv"
        with open('hyperparameters.json', 'r') as f:
            self.default_param_grids = json.load(f)


    def prepare_data(self, encoding_type='target'):
        if self.df is None:
            raise ValueError("self.df is None. Please initialize it with a valid DataFrame before calling this method.")
        print("Starting data preparation...")
        
        # Store the encoding type
        self.encoding_type = encoding_type
        
        # Drop unnecessary columns
        self.df = self.df.drop(columns=['Unnamed: 0', "RECORD_ID"], errors='ignore')

        self.df.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Optionally, add a NaN check or imputation here
        nan_check = self.df.isna().sum()
        if nan_check.any():
            print("Warning: NaN values detected. Consider imputation.")
            
        # Identify the prefix of the chosen target column
        target_prefix = "_".join(self.target_col.split("_")[:-1])
        
        # Drop all other questionnaire columns
        other_q_cols = [col for col in self.df.columns if col.startswith("Q") and not col.startswith(target_prefix)]
        self.df = self.df.drop(columns=other_q_cols)
        
        # Save intermediate data if flag is set
        if self.save_intermediate_data:
            self.intermediate_data['Initial_Dataframe'] = self.df.copy()
            
        # Label-encode the target column if it's not numerical
        if not np.issubdtype(self.df[self.target_col].dtype, np.number):
            self.target_label_encoder = LabelEncoder()
            self.df[self.target_col] = self.target_label_encoder.fit_transform(self.df[self.target_col])
            print(self.target_label_encoder.classes_)
        
        # Feature selection
        num_features = self.df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        cat_features = self.df.select_dtypes(include=['object']).columns.tolist()
        if self.target_col in num_features: 
            num_features.remove(self.target_col)
        if self.target_col in cat_features: 
            cat_features.remove(self.target_col)
        
        # Store feature types
        self.num_features = num_features
        self.cat_features = cat_features

        for col in self.num_features:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')

        for col in self.cat_features:
            self.df[col] = self.df[col].astype(str)
        
        # Train-test split
        X = self.df.drop(columns=[self.target_col])
        y = self.df[self.target_col]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Save intermediate data if flag is set
        if self.save_intermediate_data:
            self.intermediate_data['Train_Test_Split'] = {
                'X_train': self.X_train.copy(),
                'X_test': self.X_test.copy(),
                'y_train': self.y_train.copy(),
                'y_test': self.y_test.copy()
            }

        # Update self.df to be the concatenation of the transformed train and test sets
        self.df = pd.concat([self.X_train, self.X_test])
        
        # Save intermediate data if flag is set
        if self.save_intermediate_data:
            self.intermediate_data['Encoded_Data'] = self.df.copy()


    def get_training_data(self):
        if hasattr(self, 'X_train') and hasattr(self, 'y_train'):
            return self.X_train.copy(), self.y_train.copy()
        else:
            raise AttributeError("Training data has not been initialized. Call `prepare_data` first.")

    def get_test_data(self):
        if hasattr(self, 'X_test') and hasattr(self, 'y_test'):
            return self.X_test.copy(), self.y_test.copy()
        else:
            raise AttributeError("Test data has not been initialized. Call `prepare_data` first.")

    def create_preprocessing_pipeline(self, scale_data=True, feature_engineering=None,kmeans_n_clusters=8, pca_n_components=7):
        # Basic imputers
        num_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='mean'))])
        
        # Choose the appropriate categorical transformer based on encoding_type
        if self.encoding_type == 'one_hot':
            one_hot_encoder = OneHotEncoder(drop='first', sparse=False, handle_unknown='ignore')
            cat_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),
                                            ('encoder', one_hot_encoder)])
            self.one_hot_encoder = one_hot_encoder  # Save as instance variable
        elif self.encoding_type == 'label':
            label_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            cat_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),
                                            ('encoder', label_encoder)])
            self.label_encoder = label_encoder  # Save as instance variable
        elif self.encoding_type == 'target':
            self.target_encoder = TargetEncoder()  # Initialize and set as an instance variable
            self.target_encoder.fit(self.X_train[self.cat_features], self.y_train)  # Explicitly fit it
            cat_transformer = Pipeline(steps=[('encoder', self.target_encoder)])

        # Additional feature engineering steps
        feature_engineering_steps = []
        if feature_engineering == 'KMeans':
            feature_engineering_steps.append(('kmeans', KMeans(n_clusters=kmeans_n_clusters, n_init=10 )))
        elif feature_engineering == 'PCA':
            feature_engineering_steps.append(('pca', PCA(n_components=pca_n_components, random_state=42)))
        elif feature_engineering == 'BOTH':
            feature_engineering_steps.append(('kmeans', KMeans(n_clusters=kmeans_n_clusters, n_init=10)))
            feature_engineering_steps.append(('pca', PCA(n_components=pca_n_components, random_state=42)))

        # Assemble the numerical transformer
        if scale_data:
            num_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())
            ] + feature_engineering_steps)
        else:
            num_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean'))
            ] + feature_engineering_steps)

        # Create the final preprocessor object
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', num_transformer, self.num_features),
                ('cat', cat_transformer, self.cat_features)
            ])
        
        if self.save_intermediate_data:
            self.intermediate_data['Preprocessor'] = self.preprocessor

        return self.preprocessor
    

    def _get_model_types_to_fit(self, model_type):
                model_types_to_fit = []
                if 'RF' in model_type or 'All' in model_type: 
                    model_types_to_fit.append('RF')
                if 'XGBoost' in model_type or 'All' in model_type: 
                    model_types_to_fit.append('XGBoost')
                if 'LogisticRegression' in model_type or 'All' in model_type: 
                    model_types_to_fit.append('LogisticRegression')
                if 'SVM' in model_type or 'All' in model_type: 
                    model_types_to_fit.append('SVM')
                if 'GBC' in model_type or 'All' in model_type:  
                    model_types_to_fit.append('GBC')
                if 'Ensemble' in model_type or 'All' in model_type: 
                    model_types_to_fit.append('Ensemble')
                if 'Stacked' in model_type or 'All' in model_type: 
                    model_types_to_fit.append('Stacked')
                
                return model_types_to_fit
    
    def _get_pipeline_for_model_type(self, model_type):
        if model_type == 'RF':
            clf = Pipeline(steps=[('preprocessor', self.preprocessor),
                                ('classifier', RandomForestClassifier(random_state=42))])
        elif model_type == 'XGBoost':
            clf = Pipeline(steps=[('preprocessor', self.preprocessor),
                                ('classifier', XGBClassifier(random_state=42))])
        elif model_type == 'GBC':
            clf = Pipeline(steps=[('preprocessor', self.preprocessor),
                                ('classifier', GradientBoostingClassifier(random_state=42))])
        elif model_type == 'LogisticRegression':
            clf = Pipeline(steps=[('preprocessor', self.preprocessor),
                                ('classifier', LogisticRegression(penalty='l1', solver='liblinear', C=1.0, random_state=42))])
        elif model_type == 'SVM':
            clf = Pipeline(steps=[('preprocessor', self.preprocessor),
                                ('classifier', SVC(C=1.0, kernel='rbf', random_state=42))])
        elif model_type == 'Ensemble':
            clf = Pipeline(steps=[('preprocessor', self.preprocessor),
                                ('classifier', VotingClassifier(estimators=[
                                    ('RF', RandomForestClassifier(random_state=42)),
                                    ('XGBoost', XGBClassifier(random_state=42)),
                                    ('LogisticRegression', LogisticRegression(penalty='l1', solver='liblinear', C=1.0, random_state=42)),
                                    ('SVM', SVC(C=1.0, kernel='rbf', random_state=42, probability=True)),
                                    ('GBC', GradientBoostingClassifier(random_state=42))],
                                                                voting='soft'))])
        elif model_type == 'Stacked':
            clf = Pipeline(steps=[('preprocessor', self.preprocessor),
                                ('classifier', StackingClassifier(estimators=[
                                    ('RF', RandomForestClassifier(random_state=42)),
                                    ('XGBoost', XGBClassifier(random_state=42)),
                                    ('LogisticRegression', LogisticRegression(penalty='l1', solver='liblinear', C=1.0, random_state=42)),
                                    ('SVM', SVC(C=1.0, kernel='linear', random_state=42, probability=True)),
                                    ('GBC', GradientBoostingClassifier(random_state=42))],
                                                                    final_estimator=LogisticRegression(penalty='l1', solver='liblinear', C=1.0, random_state=42)))])
        
        if self.loaded_params and model_type in self.loaded_params:
            clf.set_params(**self.loaded_params[model_type])

        return clf

    
    def train_models(self, model_type='All', feature_importance=False, 
                    evaluate=True, use_grid_search=False, 
                    custom_param_grids=None, saved_params_path=None,
                    use_cv=False, cv_folds=5, plot_learning_curve=False, compare_models=False,
                    experiment_logging=False, notes={}):

        # Initialize dictionaries to store results
        self.models = {}
        self.evaluation_results = {}
        self.classification_reports = {}
        self.best_params = {}
        self.use_grid_search = use_grid_search
        self.should_evaluate_models = evaluate
        self.feature_importance = feature_importance
        self.model_type = model_type
        #self.trained_models = model_type
        self.use_cv = use_cv
        self.cv_folds = cv_folds
        model_types_to_fit = self._get_model_types_to_fit(model_type)
        self.model_types_to_fit = model_types_to_fit
        self.feature_names = self.X_train.columns.tolist()

        # Parameter grids for grid search
        param_grids = custom_param_grids if custom_param_grids else self.default_param_grids



        print("Starting model training...")

        if saved_params_path:
            with open(saved_params_path, 'r') as f:
                loaded_params = json.load(f)
            self.loaded_params = loaded_params  # Assign it to class attribute

            # Validate loaded parameters for each model
            for model_type in model_types_to_fit:
                if not self.validate_params(loaded_params.get(model_type, {}), model_type):
                    print(f"Invalid parameters detected for {model_type}. Proceeding with default parameters.")
                else:
                    print(f"Loaded parameters for {model_type} validated successfully.")

        for model_type in tqdm(model_types_to_fit):
            clf = self._get_pipeline_for_model_type(model_type)  # Assuming this function returns the correct pipeline
            
            
            def extract_fitted_transformers(fitted_clf):
                # Extract and save the fitted transformers based on encoding type
                if 'one_hot' in self.encoding_type:
                    self.one_hot_encoder = fitted_clf.named_steps['preprocessor'].named_transformers_['cat'].named_steps['encoder']
                elif 'label' in self.encoding_type:
                    self.label_encoder = fitted_clf.named_steps['preprocessor'].named_transformers_['cat'].named_steps['encoder']
                elif 'target' in self.encoding_type:
                    self.target_encoder = fitted_clf.named_steps['preprocessor'].named_transformers_['cat'].named_steps['encoder']

            if use_grid_search:
                grid_search = GridSearchCV(clf, param_grids[model_type], cv=cv_folds, verbose=1, n_jobs=-1)
                grid_search.fit(self.X_train, self.y_train)
                self.best_params[model_type] = grid_search.best_params_
                print(f"Best parameters for {model_type}: {grid_search.best_params_}")
                clf = grid_search.best_estimator_  # Update clf to be the best estimator
                self.models[model_type] = clf  # Store the best model
                extract_fitted_transformers(clf)  # Extract and save the fitted transformers
                with open('best_params.json', 'w') as f:
                    json.dump(self.best_params, f, indent=4)

            elif use_cv:
                stratified_kfold = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
                cv_scores = cross_val_score(clf, self.X_train, self.y_train, cv=stratified_kfold, scoring='accuracy')
                print(f"Cross-validation scores for {model_type}: {cv_scores}")
                print(f"Mean CV score for {model_type}: {np.mean(cv_scores)}")
                clf.fit(self.X_train, self.y_train)  # Fit the model on the entire training set
                self.models[model_type] = clf  # Store the fitted model
                extract_fitted_transformers(clf)
            else:
                clf.fit(self.X_train, self.y_train)  # Regular fit
                self.models[model_type] = clf  # Store the fitted model
                extract_fitted_transformers(clf)

        

        if feature_importance:
            self.plot_feature_importances()
                
        if self.should_evaluate_models:
                self.evaluate_models()

        if experiment_logging:
            evaluation_results = self.gather_evaluation_results()  
            self.log_experiment_results_csv(self.log_file, exp_id=None, 
                                       evaluation_results=evaluation_results, notes=notes)

        if plot_learning_curve:
            self.plot_learning_curve(cv_folds=self.cv_folds)

        if compare_models:
            self.plot_model_comparison_heatmap()

       # return self.models, self.best_params

    def gather_evaluation_results(self):
        """
        Consolidate various evaluation metrics and configurations into a dictionary.
        """
        results = {}
        
        # Storing the best parameters for each model:
        results['best_params'] = self.best_params

        # Storing evaluation metrics like accuracy, precision, etc.:
        results['evaluation_metrics'] = self.evaluation_results

        # Storing classification reports:
        results['classification_reports'] = self.classification_reports

        # Storing pipeline information:
        pipeline_info = {}
        for model_type, model in self.models.items():
            pipeline_info[model_type] = {}
            for step_name, step_component in model.named_steps.items():
                if hasattr(step_component, 'get_feature_names_out'):
                    pipeline_info[model_type][step_name] = step_component.get_feature_names_out().tolist()
                else:
                    pipeline_info[model_type][step_name] = str(step_component)
        results['pipeline_info'] = pipeline_info

        return results
    

    def get_next_experiment_id(self,log_file):
        if os.path.exists(log_file) and os.path.getsize(log_file) > 0:
            try:
                experiment_log = pd.read_csv(log_file)
                last_exp_id = experiment_log['exp_id'].iloc[-1]
                next_exp_id = f"experiment_{int(last_exp_id.split('_')[1]) + 1}"
                print(f"Next exp_id generated: {next_exp_id}")  # Debug print
            except pd.errors.EmptyDataError:
                next_exp_id = "experiment_1"
        else:
            next_exp_id = "experiment_1"
        return next_exp_id

    def log_experiment_results_csv(self,log_file, exp_id=None, evaluation_results=None, notes={}):
        if exp_id is None:
            exp_id = self.get_next_experiment_id(log_file)
            
        if os.path.exists(log_file) and os.path.getsize(log_file) > 0:
            try:
                experiment_log = pd.read_csv(log_file)
            except pd.errors.EmptyDataError:
                experiment_log = pd.DataFrame(columns=['exp_id', 'evaluation_results', 'notes'])
        else:
            experiment_log = pd.DataFrame(columns=['exp_id', 'evaluation_results', 'notes'])

        new_row = {'exp_id': exp_id, 'evaluation_results': str(evaluation_results), 'notes': str(notes)}
        experiment_log = experiment_log.append(new_row, ignore_index=True)
        experiment_log.to_csv(log_file, index=False)

    def validate_params(self, loaded_params, model_type):
        valid_params = self._get_pipeline_for_model_type(model_type).get_params().keys()
        for param in loaded_params:
            if param not in valid_params:
                print(f"Warning: Invalid parameter {param} for model {model_type}")
                return False
        return True

    def evaluate_models(self):
        # Initialize dictionary to store classification reports
        self.classification_reports = {}
        self.evaluation_results = {}
        
        for model_type, model in self.models.items():
            print(f"Evaluating {model_type}...")
            
            # Predictions
            y_pred_train = model.predict(self.X_train)
            y_pred_test = model.predict(self.X_test)
            
            # Classification Report
            report_train = classification_report(self.y_train, y_pred_train, output_dict=True)
            report_test = classification_report(self.y_test, y_pred_test, output_dict=True)
            self.classification_reports[model_type] = {'train': report_train, 'test': report_test}
            
            print(f"Classification Report for {model_type} on Training Data:")
            print(report_train)
            
            print(f"Classification Report for {model_type} on Test Data:")
            print(report_test)
            
            # Additional Metrics on Test Data
            acc = accuracy_score(self.y_test, y_pred_test)
            prec = precision_score(self.y_test, y_pred_test, average='weighted')
            rec = recall_score(self.y_test, y_pred_test, average='weighted')
            f1 = f1_score(self.y_test, y_pred_test, average='weighted')
            
            print(f"Additional Metrics for {model_type} on Test Data:")
            print(f"Accuracy: {acc}")
            print(f"Precision: {prec}")
            print(f"Recall: {rec}")
            print(f"F1 Score: {f1}")

            self.evaluation_results[model_type] = {'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1': f1}
            
            # Confusion Matrix
            cm_train = confusion_matrix(self.y_train, y_pred_train)
            cm_test = confusion_matrix(self.y_test, y_pred_test)
            
            fig, ax = plt.subplots(1, 2, figsize=(14, 6))
            
            sns.heatmap(cm_train, annot=True, fmt="d", ax=ax[0])
            ax[0].set_title(f"{model_type} Confusion Matrix - Training Data")
            
            sns.heatmap(cm_test, annot=True, fmt="d", ax=ax[1])
            ax[1].set_title(f"{model_type} Confusion Matrix - Test Data")
            
            plt.show()

    def plot_model_comparison_heatmap(self, metrics=['accuracy', 'precision', 'recall', 'f1'], use_hyperparameters=False):
        # Initialize an empty DataFrame to store the metrics
        df_metrics = pd.DataFrame(index=self.trained_models, columns=metrics)
        
        for model_type in self.model_type:
            clf = self._get_pipeline_for_model_type(model_type)
            
            # If use_hyperparameters is True, set the best hyperparameters for the model
            if use_hyperparameters:
                best_params = self._get_best_hyperparameters_for_model(model_type)  # Assuming you have a method to get best hyperparameters
                clf.set_params(**best_params)
            
            # Fit the classifier and make predictions
            clf.fit(self.X_train, self.y_train)
            y_pred = clf.predict(self.X_test)
            
            # Compute the metrics and store them in the DataFrame
            if 'accuracy' in metrics:
                df_metrics.loc[model_type, 'accuracy'] = accuracy_score(self.y_test, y_pred)
            if 'precision' in metrics:
                df_metrics.loc[model_type, 'precision'] = precision_score(self.y_test, y_pred, average='weighted')
            if 'recall' in metrics:
                df_metrics.loc[model_type, 'recall'] = recall_score(self.y_test, y_pred, average='weighted')
            if 'f1' in metrics:
                df_metrics.loc[model_type, 'f1'] = f1_score(self.y_test, y_pred, average='weighted')
        
        # Generate the heatmap
        plt.figure(figsize=(10, 6))
        sns.heatmap(df_metrics.astype(float), annot=True, cmap='coolwarm', cbar=True, fmt='.2f')
        plt.title('Model Comparison')
        plt.xlabel('Metric')
        plt.ylabel('Model')
        plt.show()

    def sample_data(self, X, y, fraction):
        combined = pd.concat([X, y], axis=1)
        sampled = combined.sample(frac=fraction, random_state=42)
        X_sampled = sampled.drop([self.target_col], axis=1)
        y_sampled = sampled[self.target_col]
        return X_sampled, y_sampled
    
    def plot_learning_curve_graph(self, mean_accuracies, confidence_intervals, model_type):
        fractions = np.linspace(0.1, 1.0, 10)  # The same fractions you use in plot_learning_curve
        plt.figure(figsize=(14, 8))
        
        line_color = '#1f77b4'  
        confidence_color = '#ff7f0e'  
        
        plt.plot(fractions, mean_accuracies, marker='o', linestyle='-', color=line_color, 
                label=f'Mean Accuracy for {model_type}', linewidth=2, markersize=8)
        

        plt.fill_between(fractions,
                        np.array(mean_accuracies) - np.array(confidence_intervals),
                        np.array(mean_accuracies) + np.array(confidence_intervals),
                        color=confidence_color, alpha=0.3, label='95% Confidence Interval')
        
        plt.title(f'Learning Curve for {model_type}', fontsize=18)
        plt.xlabel('Fraction of Training Data', fontsize=14)
        plt.ylabel('Mean Accuracy', fontsize=14)
        plt.grid(True, linestyle='--', linewidth=0.5, color='gray')
        plt.legend(fontsize=12, loc='lower right')
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.show()

    def plot_learning_curve(self, cv_folds=5):
        all_mean_accuracies = {}
        all_confidence_intervals = {}
        
        for model_type in self.model_types_to_fit:
            mean_accuracies = []
            confidence_intervals = []
            
            for data_fraction in np.linspace(0.1, 1.0, 10):
                X_sample, y_sample = self.sample_data(self.X_train, self.y_train, fraction=data_fraction)
                
                clf = self._get_pipeline_for_model_type(model_type)
                stratified_kfold = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
                cv_score = cross_val_score(clf, X_sample, y_sample, cv=stratified_kfold, scoring='accuracy')
                
                mean_cv_score = np.mean(cv_score)
                confidence_interval = np.std(cv_score) * 1.96 / np.sqrt(cv_folds)
                
                mean_accuracies.append(mean_cv_score)
                confidence_intervals.append(confidence_interval)
            
            all_mean_accuracies[model_type] = mean_accuracies
            all_confidence_intervals[model_type] = confidence_intervals
            self.plot_learning_curve_graph(mean_accuracies, confidence_intervals, model_type)

    def plot_feature_importances(self):
        for model_type, model in self.models.items():
            print(f"Plotting feature importance for {model_type}")
            
            if model_type in ['RF', 'XGBoost', 'GBC']:
                # For tree-based models
                importances = model.named_steps['classifier'].feature_importances_
            elif model_type == 'LogisticRegression':
                # For Logistic Regression
                importances = np.abs(model.named_steps['classifier'].coef_[0])
            elif model_type == 'SVM':
                # For SVM with linear kernel
                if model.named_steps['classifier'].kernel == 'linear':
                    importances = np.abs(model.named_steps['classifier'].coef_[0])
                else:
                    print("For non-linear kernels in SVM, feature importance is not straightforward to compute.")
                    continue
            else:
                print(f"Feature importance for {model_type} is not supported.")
                continue
            # Sort feature importances in descending order and take the top 10
            indices = np.argsort(importances)[::-1][:10]
            names = [self.feature_names[i] for i in indices]
            sorted_importances = importances[indices]
            df = pd.DataFrame({
                'Features': names,
                'Importance': sorted_importances
            })
            fig = px.bar(df, x='Importance', y='Features', orientation='h', 
                        title=f"Feature Importance for {model_type}")
            fig.show()

    def load_hyperparameters(self, json_path):
        with open(json_path, 'r') as f:
            self.hyperparameters = json.load(f)
        
    def perform_sensitivity_analysis(self, model_type, cv_folds=5, metric='accuracy', log_scale=False):
        if model_type not in self.hyperparameters:
            print(f"No hyperparameters found for {model_type}")
            return

        for param_name, param_values in self.hyperparameters[model_type].items():
            mean_accuracies = []
            std_errors = []
            used_param_values = []

            clf = self._get_pipeline_for_model_type(model_type)

            plt.figure(figsize=(14, 8))
            
            line_color = '#1f77b4'  
            confidence_color = '#ff7f0e'
            
            for value in param_values:
                try:
                    stratified_kfold = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
                    clf.set_params(**{param_name: value})
                    cv_scores = cross_val_score(clf, self.X_train, self.y_train, cv=stratified_kfold, scoring=metric)
                    mean_accuracy = np.mean(cv_scores)
                    std_error = np.std(cv_scores) / np.sqrt(cv_folds)
                    
                    mean_accuracies.append(mean_accuracy)
                    std_errors.append(std_error)
                    used_param_values.append(value)
                    
                except ValueError as e:
                    print(f"Skipping {param_name}={value} due to error: {e}")
                    continue

            plt.errorbar(used_param_values, mean_accuracies, yerr=std_errors, fmt='o-', 
                        color=line_color, linewidth=2, markersize=8,
                        label=f'Mean {metric} ± Std. Error for {model_type}')
            

            filtered_indices = [i for i, x in enumerate(used_param_values) if x is not None]

            filtered_param_values = [used_param_values[i] for i in filtered_indices]
            filtered_mean_accuracies = [mean_accuracies[i] for i in filtered_indices]
            filtered_std_errors = [std_errors[i] for i in filtered_indices]

            plt.fill_between(filtered_param_values,
                    np.array(filtered_mean_accuracies) - np.array(filtered_std_errors),
                    np.array(filtered_mean_accuracies) + np.array(filtered_std_errors),
                    color=confidence_color, alpha=0.3, label='Std. Error')

            plt.title(f'Sensitivity Analysis for {param_name} in {model_type}', fontsize=18)
            plt.xlabel(param_name, fontsize=14)
            plt.ylabel(f'Mean {metric}', fontsize=14)
            plt.grid(True, linestyle='--', linewidth=0.5, color='gray')
            plt.legend(fontsize=12, loc='lower right')
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)

            if log_scale:
                plt.xscale('log')
            
            plt.show()

    def plot_3D_sensitivity(self, model_type, param1, param2, cv_folds=5, metric='accuracy', log_scale=False):
        values1 = self.hyperparameters[model_type].get(param1, [])
        values2 = self.hyperparameters[model_type].get(param2, [])

        if not values1 or not values2:
            print(f"Invalid parameters specified: {param1}, {param2}")
            return

        Z = []  # Store metric values
        X, Y = np.meshgrid(values1, values2)

        clf = self._get_pipeline_for_model_type(model_type)
        stratified_kfold = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

        for val1, val2 in product(values1, values2):
            try:
                clf.set_params(**{param1: val1, param2: val2})
                cv_scores = cross_val_score(clf, self.X_train, self.y_train, cv=stratified_kfold, scoring=metric)
                mean_score = np.mean(cv_scores)
                Z.append(mean_score)
            except Exception as e:
                print(f"Skipping combination {param1}={val1}, {param2}={val2} due to error: {e}")
                Z.append(np.nan)
        Z = np.array(Z).reshape(len(values2), len(values1))

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Log-transform X and Y if log_scale is True
        if log_scale:
            X_log = np.log10(X)
            Y_log = np.log10(Y)
            ax.set_xticks(np.log10([0.001, 0.01, 0.1, 1, 10, 100]))
            ax.set_xticklabels([0.001, 0.01, 0.1, 1, 10, 100])
            ax.set_yticks(np.log10([0.001, 0.01, 0.1, 1, 10, 100]))
            ax.set_yticklabels([0.001, 0.01, 0.1, 1, 10, 100])
        else:
            X_log, Y_log = X, Y

        surf = ax.plot_surface(X_log, Y_log, Z, cmap='viridis')
        cbar = fig.colorbar(surf)
        cbar.ax.set_ylabel(metric)

        ax.set_zlim(0, 1)
        ax.set_xlabel(param1)
        ax.set_ylabel(param2)
        ax.set_zlabel(metric)
        ax.set_title(f'3D Sensitivity Analysis for {model_type}')

        plt.show()



    def prepare_new_data(self, new_data, sample_fraction=None, state_col='STATE'):
        print("Starting new data preparation...")
        
        # If a sample fraction is specified, shuffle and sample the data
        if sample_fraction:
            new_data = shuffle(new_data, random_state=42)
            sample_size = int(len(new_data) * sample_fraction)
            new_data = new_data.iloc[:sample_size]
         # Replace infinite values with NaN
        new_data.replace([np.inf, -np.inf], np.nan, inplace=True)

        self.new_data_original_state_values = new_data[state_col].copy()
        
        # Drop unnecessary columns, similar to the original data
        new_data = new_data.drop(columns=['Unnamed: 0', "RECORD_ID"], errors='ignore')
        
        # Feature selection: Assume new_data has the same features as the training data
        num_features = self.num_features
        cat_features = self.cat_features

        for col in self.num_features:
            new_data[col] = pd.to_numeric(new_data[col], errors='coerce')

        for col in self.cat_features:
            new_data[col] = new_data[col].astype(str)

         # Filter for columns that are used in training
        relevant_columns = self.num_features + self.cat_features
        self.original_columns = relevant_columns

        print("relevant_columns: ", relevant_columns)
        new_data = new_data[relevant_columns]

        # Use the same preprocessor fitted during training to transform new data
        #new_data_preprocessed = self.preprocessor.transform(new_data)

        return new_data
    
    def make_inference(self, new_data, specified_models=None, return_dataframe=False):
        """
        Make predictions on new data.
        Parameters:
        - new_data: The new data that needs to be predicted.
        - specified_models: List of model types to use for inference. If None, use all trained models.
        - return_dataframe: Whether to return predictions as a column in the original dataframe.
        Returns:
        - A dictionary containing predictions from each specified model or
        the original dataframe with added columns for predictions, depending on `return_dataframe`.
        """
        print("** rubs crystal ball **")
        if specified_models is None:
            specified_models = list(self.models.keys())
            
        predictions = {}
        for model_type in specified_models:
            if model_type in self.models:
                model = self.models[model_type]
                preds = model.predict(new_data)  # This will apply all the steps in the pipeline
                predictions[model_type] = preds
               # Model-specific comments
                if model_type == "RF":
                    print("Ah, the forest is dense but full of wisdom.")
                elif model_type == "SVM":
                    print("Support vectors to the rescue!")
                elif model_type == "LogisticRegression":
                    print("Calculating odds and making bets.")
                else:
                    print(f"Hmm, so that's interesting. Totally makes sense. {model_type} gets it.")

        else:
            print(f"No trained model found for type {model_type}. Skipping.")
        
        if return_dataframe:
            df_with_predictions = new_data.copy()  # Create a copy of the original data
            for model_type, preds in predictions.items():
                df_with_predictions[f"{model_type}_Prediction"] = preds  # Add prediction columns
            return df_with_predictions
        
        return predictions

    def plot_majority_predictions(self, new_data_with_predictions, state_col='STATE', pred_col='RF_predictions'):
        """
        Plot a choropleth map showing the ratio of party predictions by state.

        Parameters:
        - new_data_with_predictions: DataFrame containing the original data along with prediction columns
        - state_col: Column name that contains the state labels
        - pred_col: Column name that contains the prediction labels
        """

        # Decode the prediction columns back to original labels using target_label_encoder
        if new_data_with_predictions[pred_col].dtype == 'object':
            print("Labels are already decoded. Skipping label decoding.")
        else:
            pred_encoder = self.target_label_encoder
            if pred_encoder is not None:
                new_data_with_predictions[pred_col] = pred_encoder.inverse_transform(new_data_with_predictions[pred_col])
            else:
                print(f"No label encoder found for the '{self.target_col}' column.")

        # Group by STATE and prediction, and count the number of occurrences
        grouped_df = new_data_with_predictions.groupby([state_col, pred_col]).size().reset_index(name='count')
        
        # Calculate the total count for each state
        total_count_df = grouped_df.groupby([state_col]).agg({'count': 'sum'}).reset_index()
        total_count_df.rename(columns={'count': 'total_count'}, inplace=True)

        # Merge to get ratio
        merged_df = pd.merge(grouped_df, total_count_df, on=state_col)
        merged_df['ratio'] = merged_df['count'] / merged_df['total_count']

        # Plotting using Plotly
        fig = px.choropleth(merged_df, 
                            locations=state_col, 
                            color='ratio',
                            hover_name=state_col, 
                            locationmode="USA-states",
                            scope="usa",
                            title="Degree of Party Prediction by State",
                            color_continuous_scale=px.colors.sequential.Plasma,
                            animation_frame=pred_col)
        
        fig.show()

    def compare_with_actual_votes(self, csv_path, predicted_df, state_col='STATE', pred_col='Predictions'):
        # Create a mapping of class names
        class_name_mapping = {
            'PRESIDENT DONALD TRUMP': 'TRUMP',
            'PRESIDENT JOE BIDEN': 'BIDEN',
            'OTHER GOP': 'OTHER'
        }
        
        # First, calculate the predicted ratios
        if predicted_df[pred_col].dtype == 'object':
            print("Labels are already decoded. Skipping label decoding.")
        else:
            pred_encoder = self.target_label_encoder
            if pred_encoder is not None:
                predicted_df[pred_col] = pred_encoder.inverse_transform(predicted_df[pred_col])
            else:
                print(f"No label encoder found for the '{self.target_col}' column.")
        
        # Map class names in predicted_df to match those in the CSV
        predicted_df[pred_col] = predicted_df[pred_col].map(class_name_mapping)
        
        grouped_df = predicted_df.groupby([state_col, pred_col]).size().reset_index(name='count')
        total_count_df = grouped_df.groupby([state_col]).agg({'count': 'sum'}).reset_index()
        total_count_df.rename(columns={'count': 'total_count'}, inplace=True)

        merged_df = pd.merge(grouped_df, total_count_df, on=state_col)
        merged_df['Predicted_Ratio'] = merged_df['count'] / merged_df['total_count']
        predicted_ratios_df = merged_df.pivot(index=state_col, columns=pred_col, values='Predicted_Ratio').reset_index()

        # Read the CSV and calculate the actual ratios
        votes_by_state_df = pd.read_csv(csv_path)
        votes_by_state_df['Total_Votes'] = votes_by_state_df.sum(axis=1)
        for party in ['TRUMP', 'BIDEN', 'OTHER']:
            votes_by_state_df[f'RATIO_{party}'] = votes_by_state_df[f'VOTES_{party}'] / votes_by_state_df['Total_Votes']

        actual_ratios_df = votes_by_state_df[['STATE', 'RATIO_TRUMP', 'RATIO_BIDEN', 'RATIO_OTHER']]

        # Merge and compare
        comparison_df = pd.merge(actual_ratios_df, predicted_ratios_df, on='STATE', how='inner')

        # Visualization
        parties = ['TRUMP', 'BIDEN', 'OTHER']
        for party in parties:
            plt.figure(figsize=(12, 6))
            plt.bar(comparison_df['STATE'], comparison_df[f'RATIO_{party}'], alpha=0.5, label='Actual Ratios')
            plt.bar(comparison_df['STATE'], comparison_df[party], alpha=0.5, label='Predicted Ratios')
            plt.title(f'Vote Ratios for {party}')
            plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
            plt.legend()
            plt.show()
