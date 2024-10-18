# Clustering Algorithms Test

This repository is designed to test the performance of clustering algorithms. The library includes the following main files:

- **Agglomerative.py**: Implements Agglomerative Clustering.
- **GMM.py**: Implements Gaussian Mixture Model (GMM) clustering.
- **KMeans.py**: Implements K-Means clustering.
- **DataProcessing.py**: Contains functions for data processing and evaluation metrics.

## Installation

To install the required dependencies, use the following command:

```bash
pip install -r requirements.txt
```

In addition, this model uses BERT to extract embeddings for the algorithm. Please download BERT pre-trained models and configs from bert-base-uncased (config.json, pytorch_model.bin, tokenizer.json, tokenizer_config.json, vocab.txt)
Link:
[bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased)


## Datasets

The datasets used for testing clustering algorithms can be found at the following links:

1. [Computer Science Text Classification (booksm)](https://www.kaggle.com/datasets/deepak711/4-subject-data-text-classification?select=Computer_Science)
2. [Biomedical Text Publication Classification (cancer)](https://www.kaggle.com/datasets/falgunipatel19/biomedical-text-publication-classification)
3. [BBC News Summary (bbcm)](https://www.kaggle.com/datasets/pariza/bbc-news-summary)

## Usage

1. **Data Processing**: Use `DataProcessing.py` to load, process, and evaluate the datasets.
2. **Clustering Algorithms**: Use `Agglomerative.py`, `GMM.py`, or `KMeans.py` to apply different clustering techniques to the processed data.

## Evaluation

Evaluation metrics for clustering models, such as Precision, Recall, and F1 Score, are calculated in `DataProcessing.py`.



