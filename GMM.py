import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from DataProcessing import load_and_select_dataset, extract_bert_embeddings, evaluate_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load and select dataset
summaries, labels, n_clusters, dataset_name = load_and_select_dataset()

# Encode labels as integers for visualization purposes
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# Extract BERT embeddings for the clean text
X_bert = extract_bert_embeddings(summaries)

# Scale the features
scaler = StandardScaler()
X_bert_scaled = scaler.fit_transform(X_bert)

# Split the dataset into training and testing sets
X_train_bert, X_test_bert, labels_train, labels_test = train_test_split(X_bert_scaled, labels_encoded, test_size=0.2, random_state=42)

# Train Logistic Regression classifier
logreg_model = LogisticRegression(max_iter=2000, random_state=42, solver='lbfgs')
logreg_model.fit(X_train_bert, labels_train)

# Predict labels for the test set
predicted_labels = logreg_model.predict(X_test_bert)

# Create output directory
output_dir = os.path.join('results', 'LogisticRegression', dataset_name)
os.makedirs(output_dir, exist_ok=True)

# Evaluate the model
evaluate_model(labels_test, predicted_labels)

# Apply PCA to reduce embeddings to 2D and plot (using training set embeddings for visualization)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train_bert)

plt.figure(figsize=(10, 8))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels_train, cmap='viridis', alpha=0.5)
plt.title('PCA of BERT Embeddings with Logistic Regression Classification')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar(label='True Labels')

# Save PCA plot to PNG
pca_png_path = os.path.join(output_dir, 'pca_plot.png')
plt.savefig(pca_png_path)
plt.close()

# Print paths of saved files
print(f"PCA plot saved to: {pca_png_path}")
print(f"Evaluation metrics saved to: {os.path.join(output_dir, 'evaluation_metrics.csv')}")