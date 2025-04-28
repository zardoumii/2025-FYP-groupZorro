import pandas as pd

predictions_df = pd.read_csv("outputs/model_predictions.csv")
summary = {}

for model_name, group in predictions_df.groupby('Model'):
    true_counts = group['True Label'].value_counts().to_dict()
    pred_counts = group['Predicted Label'].value_counts().to_dict()
    
    summary[model_name] = {
        'True 0s': true_counts.get(0, 0),
        'True 1s': true_counts.get(1, 0),
        'Predicted 0s': pred_counts.get(0, 0),
        'Predicted 1s': pred_counts.get(1, 0)
    }

summary_df = pd.DataFrame(summary).T
print("\nmodel health check:")
print(summary_df)

dataset_df = pd.read_csv(r"PATH TO DATASET.CSV")
label_counts = dataset_df['label'].value_counts()
print("\nfrom dataset:")
print(label_counts)

metadata_df = pd.read_csv(r"PATH TO METADATA.CSV")
metadata_counts = metadata_df['diagnostic'].value_counts()
print("\nfrom metadata:")
print(metadata_counts)
