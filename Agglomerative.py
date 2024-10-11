import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from DataProcessing import load_and_select_dataset, extract_bert_embeddings, match_clusters_to_labels, evaluate_model
from sklearn.cluster import AgglomerativeClustering

# Step 1: Load and select dataset
summaries, labels, n_clusters, dataset_name = load_and_select_dataset()

# Step 4: Extract BERT embeddings for the clean text
X_bert = extract_bert_embeddings(summaries)

# Step 6: Perform hierarchical clustering on text embeddings
def perform_hierarchical_clustering(X_bert, n_clusters):
    model = AgglomerativeClustering(n_clusters=n_clusters)
    return model.fit_predict(X_bert)

cluster_labels = perform_hierarchical_clustering(X_bert, n_clusters)

# Step 7: Map each cluster label to the most common true label (majority voting)
label_mapping = match_clusters_to_labels(cluster_labels, labels)
predicted_labels = [label_mapping[cluster] for cluster in cluster_labels]

# Step 8: Create output directory
output_dir = os.path.join('results', 'Agglomerative', dataset_name)
os.makedirs(output_dir, exist_ok=True)

# Step 9: Evaluate the model
evaluate_model(labels, predicted_labels)

# Step 10: Apply PCA to reduce embeddings to 2D and plot
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_bert)

plt.figure(figsize=(10, 8))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis', alpha=0.5)
plt.title('PCA of BERT Embeddings with Agglomerative Clustering')
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