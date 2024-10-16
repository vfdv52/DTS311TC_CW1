import os
import torch
import random
import numpy as np
import pandas as pd
from transformers import BertTokenizer, BertModel
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm

def load_and_select_dataset():
    # Input dataset selection
    dataset_choice = input("Select dataset (imdb, app, cancer, bbc, other): ").strip().lower()

    if dataset_choice == 'bbc':
        # Load summaries from 'dataset/bbc' directory
        summaries, labels = load_summaries('dataset/bbc')
        if len(summaries) > 3000:
            selected_indices = random.sample(range(len(summaries)), 3000)
            summaries = [summaries[i] for i in selected_indices]
            labels = [labels[i] for i in selected_indices]
        n_clusters = 5
    elif dataset_choice == 'cancer':
        filepath = 'dataset/cancer/alldata_1_for_kaggle.csv'
        summaries, labels = load_summaries_from_csv(filepath)
        n_clusters = 3
    elif dataset_choice == 'imdb':
        filepath = 'dataset/imdb/imdb.csv'
        summaries, labels = load_summaries_from_csv(filepath)
        n_clusters = 2
    elif dataset_choice == 'app':
        filepath = 'dataset/app/apple-twitter-sentiment-texts.csv'
        summaries, labels = load_summaries_from_csv(filepath)
        n_clusters = 3
    elif dataset_choice == 'books':
        summaries, labels = load_summaries('dataset/books')
        n_clusters = 7
    else:
        raise ValueError("Invalid dataset choice. Please select from 'imdb', 'app', 'cancer', or 'bbc'.")

    # Print sample reviews and sentiments
    print("Sample reviews (1-5):")
    for i, review in enumerate(summaries[:5]):
        print(f"{i + 1}: {review[:75]}...")  # Print the first 75 characters followed by "..."

    print("Sample labels (1-5):", labels[:5])

    return summaries, labels, n_clusters, dataset_choice

def load_summaries(directory):
    summaries = []
    labels = []
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.endswith('.txt'):
                file_path = os.path.join(root, filename)
                if os.path.isfile(file_path):
                    with open(file_path, 'r', encoding='latin1') as file:
                        text = file.read()
                        category = os.path.basename(root)
                        summaries.append(text)
                        labels.append(category)  # Assign category as label
    return summaries, labels

def load_summaries_from_csv(filepath):
    df = pd.read_csv(filepath, encoding='latin1')
    df = df.dropna()
    if len(df) > 3000:
        df = df.sample(3000, random_state=42)  # Randomly select 3000 samples if dataset is large
        if 'cancer' in filepath:
            text_column = df.columns[2]  # First column contains text
            label_column = df.columns[1]  # Second column contains labels
        elif 'imdb' in filepath:
            text_column = df.columns[0]
            label_column = df.columns[1]
    
    # Convert entire text column to strings to avoid AttributeError
    df[text_column] = df[text_column].astype(str)
    df[text_column] = df[text_column].str.replace(r'[^\x00-\x7F]+', '', regex=True)  # Remove non-ASCII characters

    summaries = df[text_column].tolist()  # Use the detected text column as the summaries
    labels = df[label_column].tolist()  # Use the detected label column as labels
    return summaries, labels


def get_bert_embeddings(text_list, tokenizer, model, device):
    embeddings = []
    for text in tqdm(text_list, desc="Extracting BERT embeddings"):
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        # Get the [CLS] token embedding
        cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        embeddings.append(cls_embedding)
    return embeddings

def load_bert_model():
    # Load pre-trained BERT model and tokenizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer.from_pretrained('models/bert-base-uncased')
    model = BertModel.from_pretrained('models/bert-base-uncased').to(device)
    return tokenizer, model, device

def extract_bert_embeddings(summaries):
    tokenizer, model, device = load_bert_model()
    bert_embeddings = get_bert_embeddings(summaries, tokenizer, model, device)
    return np.squeeze(np.array(bert_embeddings))

def match_clusters_to_labels(cluster_labels, true_labels):
    label_mapping = {}
    for cluster in tqdm(np.unique(cluster_labels), desc="Matching clusters to labels"):
        mask = (cluster_labels == cluster)
        true_label = pd.Series(np.array(true_labels)[mask]).mode()[0]  # Use pandas mode to find the most common label
        label_mapping[cluster] = true_label
    return label_mapping

def evaluate_model(true_labels, predicted_labels):
    # Calculate and print evaluation metrics
    precision = precision_score(true_labels, predicted_labels, average='macro')
    recall = recall_score(true_labels, predicted_labels, average='macro')
    f1 = f1_score(true_labels, predicted_labels, average='macro')

    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
