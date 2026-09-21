# SMS Spam Classification using Artificial Neural Network

An end-to-end NLP and Deep Learning practice project for classifying SMS messages as **ham** or **spam** using **TF-IDF features and a Feedforward Artificial Neural Network (ANN)**.

> **Project type:** Learning project  
> **Author:** Bijayeeni Halder  
> **Primary notebook:** `SMS_Spam_Classification_ANN.ipynb`

## Overview

This project implements the complete SMS spam-classification workflow:

**Raw SMS → Text Preprocessing → TF-IDF → ANN → Evaluation → Error Analysis → Baseline Comparison**

The work covers:
- Text cleaning and normalization
- Tokenization, stopword removal and lemmatization
- TF-IDF vectorization with unigrams and bigrams
- Stratified train/test splitting
- Feedforward ANN using TensorFlow/Keras
- Accuracy, Precision, Recall and F1-score
- Training/validation loss and accuracy curves
- Confusion-matrix analysis
- False-positive and false-negative analysis
- Logistic Regression baseline comparison
- Optional model/vectorizer export

## Dataset

The project uses the **SMS Spam Collection Dataset**. The notebook is designed to obtain the dataset from the UCI Machine Learning Repository, so the raw dataset is intentionally not committed to this repository.

The dataset contains SMS messages labelled `ham` or `spam`. In working copy, the distribution was:

| Class | Count | Percentage |
|---|---:|---:|
| Ham | 4,825 | 86.59% |
| Spam | 747 | 13.41% |
| **Total** | **5,572** | **100%** |

## Preprocessing

Each SMS is processed through:
1. Lowercasing
2. URL removal
3. Digit removal
4. Punctuation/special-character removal
5. Tokenization
6. Stopword removal
7. Lemmatization

The resulting text is stored as `clean_text`.

## Feature Engineering

TF-IDF is used with:

- `ngram_range=(1, 2)` — unigrams + bigrams
- `min_df=2`

The vocabulary is learned only from the training data and then applied to the test set.

## ANN Architecture

The classifier uses a feedforward neural network:

- Input layer: TF-IDF feature vector
- Dense layer: 128 neurons, ReLU
- Dropout: 30%
- Dense layer: 64 neurons, ReLU
- Dropout: 20%
- Output layer: 1 neuron, Sigmoid

Training configuration:
- Optimizer: Adam
- Loss: Binary Crossentropy
- Maximum epochs: 15
- Batch size: 32
- Validation split: 20%
- Early stopping based on validation loss

## Results

The completed run produced the following test-set results:

| Metric | ANN | Logistic Regression |
|---|---:|---:|
| Accuracy | **97.93%** | 96.95% |
| Precision | 95.65% | **97.52%** |
| Recall | **88.59%** | 79.19% |
| F1-score | **91.98%** | 87.4% |

The ANN had higher accuracy, recall and F1-score in this run, while Logistic Regression had higher precision.

## Error Analysis

The project explicitly examines five false positives and five false negatives.

Examples observed in the completed run included very short ham messages such as `"k sent"` and `"pick ur fone u dumb"` being classified as spam, while some spam messages used conversational or unusual wording and were classified as ham.

This analysis highlights a limitation of TF-IDF-based text representations: individual word/phrase statistics may not capture the full semantic context of short or informal messages.

## Limitations

- The dataset is imbalanced toward ham messages.
- Removing digits, URLs and special characters can remove useful spam signals.
- TF-IDF does not capture deeper semantic relationships between words.
- Dense conversion of TF-IDF features is practical for this dataset but is less suitable for very large vocabularies.
- The ANN is less directly interpretable than Logistic Regression.

## Possible Improvements

- Tune the ANN architecture and dropout rates.
- Tune the classification threshold.
- Experiment with TF-IDF parameters such as `max_features`, `min_df` and `max_df`.
- Preserve selected URL/number indicators as explicit features.
- Use class weighting when higher spam recall is required.
- Compare with Multinomial Naive Bayes.
- Experiment with word embeddings or transformer-based NLP models.

## Repository Structure

```text
sms-spam-classification-ann/
│
├── SMS_Spam_Classification_ANN.ipynb
├── SMS Spam Classification Using Artificial Neural Network.pdf
├── requirements.txt
└── README.md
```

## How to Run

### Option 1 — Google Colab

1. Open `SMS_Spam_Classification_ANN.ipynb` in Google Colab.
2. Run the cells from top to bottom.
3. If prompted, install the packages listed in `requirements.txt`.
4. The notebook downloads/extracts the dataset and performs preprocessing, training and evaluation.

### Option 2 — Local Jupyter

```bash
git clone <YOUR-GITHUB-REPOSITORY-URL>
cd sms-spam-classification-ann

pip install -r requirements.txt
jupyter notebook
```

Then open:

```text
SMS_Spam_Classification_ANN.ipynb
```
