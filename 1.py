import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.decomposition import PCA
from tqdm import tqdm

# Load and select dataset
dataset_choice = input("Select dataset (imdb, app, cancer, bbc, other): ").strip().lower()

if dataset_choice == 'bbc':
    # Load summaries from 'dataset/bbc' directory
    summaries, labels = [], []
    for root, dirs, files in os.walk('dataset/bbc'):
        for filename in files:
            if filename.endswith('.txt'):
                file_path = os.path.join(root, filename)
                if os.path.isfile(file_path):
                    with open(file_path, 'r', encoding='latin1') as file:
                        text = file.read()
                        category = os.path.basename(root)
                        summaries.append(text)
                        labels.append(category)
    if len(summaries) > 3000:
        selected_indices = random.sample(range(len(summaries)), 3000)
        summaries = [summaries[i] for i in selected_indices]
        labels = [labels[i] for i in selected_indices]
    n_clusters = 5
elif dataset_choice == 'imdb':
    filepath = 'dataset/imdb/imdb.csv'
    df = pd.read_csv(filepath, encoding='latin1').dropna()
    df = df.sample(3000, random_state=42) if len(df) > 3000 else df
    text_column, label_column = df.columns[0], df.columns[1]
    df[text_column] = df[text_column].astype(str).str.replace(r'[^\x00-\x7F]+', '', regex=True)
    summaries, labels = df[text_column].tolist(), df[label_column].tolist()
    n_clusters = 2
else:
    raise ValueError("Invalid dataset choice. Please select from 'imdb', 'app', 'cancer', or 'bbc'.")

# Print sample reviews and sentiments
print("Sample reviews (1-5):")
for i, review in enumerate(summaries[:5]):
    print(f"{i + 1}: {review[:75]}...")
print("Sample labels (1-5):", labels[:5])

# Extract TF-IDF features with progress bar
class ProgressVectorizer(TfidfVectorizer):
    def fit_transform(self, raw_documents, y=None):
        self.num_docs = len(raw_documents)
        return super().fit_transform(tqdm(raw_documents, desc="Extracting TF-IDF features", total=self.num_docs))

vectorizer = ProgressVectorizer(max_features=1000)  # Limit to 1000 features for simplicity
X_tfidf = vectorizer.fit_transform(summaries)

# Perform KMeans clustering
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(X_tfidf)

# Map clusters to labels (majority voting)
label_mapping = {}
for cluster in tqdm(np.unique(cluster_labels), desc="Matching clusters to labels"):
    mask = (cluster_labels == cluster)
    true_label = pd.Series(np.array(labels)[mask]).mode()[0]  # Find most common label in the cluster
    label_mapping[cluster] = true_label
predicted_labels = [label_mapping[cluster] for cluster in cluster_labels]

# Evaluate the model
precision = precision_score(labels, predicted_labels, average='macro')
recall = recall_score(labels, predicted_labels, average='macro')
f1 = f1_score(labels, predicted_labels, average='macro')

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

# Create output directory
output_dir = os.path.join('results', 'KMeans', dataset_choice)
os.makedirs(output_dir, exist_ok=True)

# Apply PCA to reduce embeddings to 2D and plot
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_tfidf.toarray())

plt.figure(figsize=(10, 8))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis', alpha=0.5)
plt.title('PCA of TF-IDF Features with KMeans Clustering')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar(label='Cluster Labels')

# Save PCA plot to PNG
pca_png_path = os.path.join(output_dir, 'pca_plot.png')
plt.savefig(pca_png_path)
plt.close()

# Print paths of saved files
print(f"PCA plot saved to: {pca_png_path}")
