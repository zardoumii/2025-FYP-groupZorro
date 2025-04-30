import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
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
from xgboost import XGBClassifier

def main_train(dataset_path):
    try:
        os.makedirs('outputs/confusion_matrices', exist_ok=True)
        os.makedirs('outputs/plots', exist_ok=True)

        # Load features → Standardize them → Cluster similar lesions → Group-aware split → Honest training/testing.
        df = pd.read_csv(dataset_path)
        feature_cols = ['asymmetry_score', 'irregularity_score', 'color_variation_score', 'blue_white_veil_score']
        X = df[feature_cols]
        y = df['label']

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)


        kmeans = KMeans(n_clusters=100, random_state=42)
        df['cluster'] = kmeans.fit_predict(X_scaled)

        gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
        for train_idx, test_idx in gss.split(X_scaled, y, groups=df['cluster']):
            X_train = X.iloc[train_idx]
            y_train = y.iloc[train_idx]
            X_test = X.iloc[test_idx]
            y_test = y.iloc[test_idx]

        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced', penalty='l1', solver='liblinear'),
            'Decision Tree': DecisionTreeClassifier(max_depth=5, min_samples_leaf=5, class_weight='balanced', random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=5, class_weight='balanced', random_state=42),
            'KNN': KNeighborsClassifier(n_neighbors=5, weights='distance', algorithm='auto', n_jobs=-1)
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

            if hasattr(model, "predict_proba"):
                y_scores = model.predict_proba(X_test)[:, 1]
            else:
                y_scores = y_pred

            fpr, tpr, _ = roc_curve(y_test, y_scores)
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

            for true_label, pred_label in zip(y_test, y_pred):
                all_predictions.append({
                    'Model': name,
                    'True Label': true_label,
                    'Predicted Label': pred_label
                })

            confusion_matrix_path = f"outputs/confusion_matrices/confusion_{name.replace(' ', '_')}.png"
            if not os.path.exists(confusion_matrix_path):
                disp = ConfusionMatrixDisplay(confusion_matrix=cm)
                fig, ax = plt.subplots(figsize=(6,6))
                disp.plot(cmap=plt.cm.Blues, ax=ax, values_format='d')
                plt.title(f"Confusion Matrix: {name}")
                plt.savefig(confusion_matrix_path)
                plt.close()

        model_names = list(results.keys())
        accuracies = [results[m]['Accuracy'] for m in model_names]

        accuracy_chart_path = 'outputs/plots/accuracy_comparison.png'
        if not os.path.exists(accuracy_chart_path):
            plt.figure(figsize=(10,6))
            plt.bar(model_names, accuracies, color=['skyblue', 'lightgreen', 'salmon', 'orange', 'purple', 'gray'])
            plt.ylabel('Accuracy')
            plt.title('Comparison of Classifier Accuracies')
            plt.ylim(0, 1)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.savefig(accuracy_chart_path)
            plt.close()

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
    dataset_path = r"C:\Users\Dara\Desktop\itu\2025-FYP-Final\result\dataset.csv"
    main_train(dataset_path)
