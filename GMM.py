import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from DataProcessing import load_and_select_dataset, extract_bert_embeddings, match_clusters_to_labels, evaluate_model
from sklearn.mixture import GaussianMixture

# Load and select dataset
summaries, labels, n_clusters, dataset_name = load_and_select_dataset()

# Extract BERT embeddings for the clean text
X_bert = extract_bert_embeddings(summaries)

# Perform Gaussian Mixture Model (GMM) clustering on text embeddings
def perform_gmm_clustering(X_bert, n_clusters):
    gmm_model = GaussianMixture(n_components=n_clusters, random_state=42)
    gmm_model.fit(X_bert)
    return gmm_model.predict(X_bert)

cluster_labels = perform_gmm_clustering(X_bert, n_clusters)

# Map each cluster label to the most common true label (majority voting)
label_mapping = match_clusters_to_labels(cluster_labels, labels)
predicted_labels = [label_mapping[cluster] for cluster in cluster_labels]

# Create output directory
output_dir = os.path.join('results', 'GMM', dataset_name)
os.makedirs(output_dir, exist_ok=True)

# Evaluate the model
evaluate_model(labels, predicted_labels)
# Apply PCA to reduce embeddings to 2D and plot
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_bert)

plt.figure(figsize=(10, 8))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis', alpha=0.5)
plt.title('PCA of BERT Embeddings with GMM Clustering')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar(label='Cluster Labels')

# Save PCA plot to PNG
pca_png_path = os.path.join(output_dir, 'pca_plot.png')
plt.savefig(pca_png_path)
plt.close()

# Print paths of saved files
print(f"PCA plot saved to: {pca_png_path}")
print(f"Evaluation metrics saved to: {os.path.join(output_dir, 'evaluation_metrics.csv')}")