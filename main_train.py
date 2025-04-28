import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc
)

def main_train(dataset_path):
    try:
        # Make folders for outputs
        os.makedirs('outputs/confusion_matrices', exist_ok=True)
        os.makedirs('outputs/plots', exist_ok=True)

        df = pd.read_csv(dataset_path)
        feature_cols = ['asymmetry_score', 'irregularity_score', 'color_variation_score', 'blue_white_veil_score']
        X = df[feature_cols]
        y = df['label']

        # Train/Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced'),
            'Decision Tree': DecisionTreeClassifier(class_weight='balanced'),
            'KNN': KNeighborsClassifier(n_neighbors=5)
        }

        results = {}
        all_predictions = []
        auc_scores = {}
        fpr_tpr_data = {}

        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            cm = confusion_matrix(y_test, y_pred)

            fpr, tpr, _ = roc_curve(y_test, y_pred)
            roc_auc = auc(fpr, tpr)

            results[name] = {
                'Accuracy': acc,
                'Precision': prec,
                'Recall': rec,
                'F1 Score': f1,
                'AUC': roc_auc,
                'Confusion Matrix': cm
            }

            auc_scores[name] = roc_auc
            fpr_tpr_data[name] = (fpr, tpr)

            # Save individual model predictions
            for true_label, pred_label in zip(y_test, y_pred):
                all_predictions.append({
                    'Model': name,
                    'True Label': true_label,
                    'Predicted Label': pred_label
                })

            # Save confusion matrix plots
            confusion_matrix_path = f"outputs/confusion_matrices/confusion_{name.replace(' ', '_')}.png"
            if not os.path.exists(confusion_matrix_path):
                disp = ConfusionMatrixDisplay(confusion_matrix=cm)
                fig, ax = plt.subplots(figsize=(6,6))
                disp.plot(cmap=plt.cm.Blues, ax=ax, values_format='d')
                plt.title(f"Confusion Matrix: {name}")
                plt.savefig(confusion_matrix_path)
                plt.close()

        # Accuracy comparison bar chart between models
        model_names = list(results.keys())
        accuracies = [results[m]['Accuracy'] for m in model_names]

        accuracy_chart_path = 'outputs/plots/accuracy_comparison.png'
        if not os.path.exists(accuracy_chart_path):
            plt.figure(figsize=(10,6))
            plt.bar(model_names, accuracies, color=['skyblue', 'lightgreen', 'salmon'])
            plt.ylabel('Accuracy')
            plt.title('Comparison of Classifier Accuracies')
            plt.ylim(0, 1)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.savefig(accuracy_chart_path)
            plt.close()

        # ROC curves
        roc_curve_path = 'outputs/plots/roc_curves.png'
        if not os.path.exists(roc_curve_path):
            plt.figure(figsize=(10,8))
            for name, (fpr, tpr) in fpr_tpr_data.items():
                plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {auc_scores[name]:.2f})')
            plt.plot([0, 1], [0, 1], 'k--', lw=2)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve Comparison for All Models')
            plt.legend(loc='lower right')
            plt.grid(True)
            plt.savefig(roc_curve_path)
            plt.close()

        # Save all outputs only if not already existing
        predictions_path = 'outputs/model_predictions.csv'
        if not os.path.exists(predictions_path):
            predictions_df = pd.DataFrame(all_predictions)
            predictions_df.to_csv(predictions_path, index=False)

        auc_scores_path = 'outputs/auc_scores.csv'
        if not os.path.exists(auc_scores_path):
            auc_df = pd.DataFrame.from_dict(auc_scores, orient='index', columns=['AUC'])
            auc_df.to_csv(auc_scores_path)

        model_metrics_path = 'outputs/model_metrics.csv'
        if not os.path.exists(model_metrics_path):
            metrics_df = pd.DataFrame.from_dict(results, orient='index')
            metrics_df.to_csv(model_metrics_path)

        print("\n All evaluation outputs saved successfully.")

    except Exception as e:
        print(f"\n Error occurred during training: {str(e)}")

if __name__ == "__main__":
    dataset_path = r"PATH TO DATASET.CSV"
    main_train(dataset_path)
