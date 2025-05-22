import sys
from os.path import join
import os
import cv2
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from os.path import join
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc
)
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import ShuffleSplit
from imblearn.combine import SMOTEENN
from collections import Counter
import json
from sklearn.model_selection import KFold
from sklearn.base import clone
















def run_evaluation(X_train, X_test, y_train, y_test, models, use_bootstrap=False, n_additional_true=100):
    """
    Run evaluation with and without bootstrap.
    If bootstrap is True, uses SMOTEENN for combined over and undersampling.
    """
    results = {}
    predictions = []
    confusion_matrices = {}
    
    for name, model in models.items():
        if use_bootstrap:
            # Initialize SMOTEENN with correct parameters
            smote_enn = SMOTEENN(sampling_strategy='auto', random_state=42)
            
            # Apply SMOTEENN
            X_resampled, y_resampled = smote_enn.fit_resample(X_train, y_train)
            
            # Train model on resampled data
            model.fit(X_resampled, y_resampled)
            y_pred = model.predict(X_test)
            
        else:
            # Train on original data
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        
        # Calculate confusion matrix and get detailed predictions
        cm = confusion_matrix(y_test, y_pred)
        confusion_matrices[name] = cm
        
        # Get exact counts of predictions
        true_negatives = cm[0, 0]
        true_positives = cm[1, 1]
        false_positives = cm[0, 1]
        false_negatives = cm[1, 0]
        
        # Store predictions
        for true_label, pred_label in zip(y_test, y_pred):
            predictions.append({
                'Model': name,
                'True Label': true_label,
                'Predicted Label': pred_label,
                'Method': 'With SMOTEENN' if use_bootstrap else 'Without Sampling',
                'Correct': true_label == pred_label
            })
        
        results[name] = {
            'Total_Test_Cases': len(y_test),
            'True_Positives': true_positives,
            'True_Negatives': true_negatives,
            'False_Positives': false_positives,
            'False_Negatives': false_negatives
        }
    
    return results, predictions, confusion_matrices



def main_train(dataset_path):
    try:
        # Create results directory if it doesn't exist
        os.makedirs('result', exist_ok=True)

        # Load and prepare data
        df = pd.read_csv(dataset_path)
        
        # Define required and optional feature columns
        required_features = ['asymmetry_score', 'irregularity_score', 'color_variation_score']
        optional_features = ['blue_white_veil_score', 'hair_coverage_ratio']
        
        # Check which features are available
        available_features = required_features + [f for f in optional_features if f in df.columns]
        
        print("\nUsing the following features:")
        for feature in available_features:
            print(f"- {feature}")
            
        X = df[available_features]
        y = df['label']

        # Convert to numpy arrays
        X = X.to_numpy()
        y = y.to_numpy()

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Initialize Stratified K-fold cross-validation
        n_splits = 5
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        # Initialize models with fixed random states
        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced', penalty='l1', solver='liblinear', random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_leaf=5, class_weight='balanced', random_state=42)
        }

        # Store results for each fold
        cv_results = {
            'Regular': {name: [] for name in models.keys()},
            'Augmented': {name: [] for name in models.keys()}
        }
        all_predictions = []

        # Perform cross-validation
        for fold, (train_idx, test_idx) in enumerate(skf.split(X_scaled, y)):
            X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Run regular evaluation (without SMOTEENN)
            regular_results, regular_predictions, _ = run_evaluation(
                X_train, X_test, y_train, y_test, 
                models, use_bootstrap=False
            )

            # Run evaluation with SMOTEENN
            augmented_results, augmented_predictions, _ = run_evaluation(
                X_train, X_test, y_train, y_test, 
                models, use_bootstrap=True
            )

            # Store results for this fold
            for name in models.keys():
                cv_results['Regular'][name].append(regular_results[name])
                cv_results['Augmented'][name].append(augmented_results[name])
            
            all_predictions.extend(regular_predictions + augmented_predictions)

        # Calculate and plot metrics
        metrics = ['Accuracy', 'Recall', 'Precision', 'F1']
        
        for name in models.keys():
            plt.figure(figsize=(12, 6))
            x = np.arange(len(metrics))
            width = 0.35

            # Calculate metrics for regular and augmented
            regular_metrics = []
            augmented_metrics = []
            
            for method in ['Regular', 'Augmented']:
                results = cv_results[method][name]
                
                # Calculate average metrics across folds
                avg_tp = np.mean([fold['True_Positives'] for fold in results])
                avg_tn = np.mean([fold['True_Negatives'] for fold in results])
                avg_fp = np.mean([fold['False_Positives'] for fold in results])
                avg_fn = np.mean([fold['False_Negatives'] for fold in results])
                
                # Calculate performance metrics
                accuracy = (avg_tp + avg_tn) / (avg_tp + avg_tn + avg_fp + avg_fn)
                recall = avg_tp / (avg_tp + avg_fn) if (avg_tp + avg_fn) > 0 else 0
                precision = avg_tp / (avg_tp + avg_fp) if (avg_tp + avg_fp) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                if method == 'Regular':
                    regular_metrics = [accuracy, recall, precision, f1]
                else:
                    augmented_metrics = [accuracy, recall, precision, f1]

            # Create grouped bar chart
            plt.bar(x - width/2, regular_metrics, width, label='Without SMOTEENN', color='skyblue')
            plt.bar(x + width/2, augmented_metrics, width, label='With SMOTEENN', color='lightcoral')

            plt.xlabel('Metrics')
            plt.ylabel('Score')
            plt.title(f'Performance Metrics Comparison for {name}')
            plt.xticks(x, metrics)
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Add value labels on top of each bar
            for i, v in enumerate(regular_metrics):
                plt.text(i - width/2, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
            for i, v in enumerate(augmented_metrics):
                plt.text(i + width/2, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
            
            plt.ylim(0, 1.1)  # Set y-axis limit to accommodate labels
            plt.tight_layout()

            # Save plot
            plt.savefig(f'result/{name.lower().replace(" ", "_")}_metrics.png')
            plt.close()

        # Save predictions to CSV
        predictions_df = pd.DataFrame(all_predictions)
        predictions_df.to_csv('result/all_predictions.csv', index=False)

        print("\nResults saved to result folder:")
        print("- Metric comparison plots saved as PNG files")
        print("- All predictions saved to all_predictions.csv")

    except Exception as e:
        print(f"\nError occurred during training: {str(e)}")
        raise e