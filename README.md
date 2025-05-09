# Tweet-Emotions-RNN-vs-LSTM

Empirical comparison of RNN and LSTM architectures for tweet-based emotion classification (*sadness*, *neutral*, *happiness*). Includes experiments with various preprocessing strategies, GloVe embeddings, bidirectional models, and training visualizations.

## Project Overview

This repository contains a deep learning project focused on comparing the performance of **Recurrent Neural Networks (RNNs)** and **Long Short-Term Memory networks (LSTMs)** in classifying emotions in tweets. The study investigates the effect of different preprocessing techniques, the use of pretrained word embeddings (GloVe), and the impact of bidirectional sequence modeling.

A systematic experimentation pipeline was followed to:

- Preprocess tweet text using:
  - Basic cleaning (lowercasing, removing URLs, mentions, hashtags, punctuation)
  - Enhanced preprocessing (emoji demojization and lemmatization)
- Train RNN and LSTM models under identical conditions
- Integrate GloVe embeddings to introduce semantic information
- Explore bidirectional RNN and LSTM architectures
- Monitor training and validation accuracy to assess overfitting
- Visualize performance across all configurations

## Models and Phases

Experiments were organized into three phases:

- **Phase 1**: Baseline RNN and LSTM models with basic preprocessing
- **Phase 2**: RNN and LSTM with enhanced preprocessing and GloVe embeddings
- **Phase 3**: Bidirectional RNNs and LSTMs, including final fine-tuned variants

Each model followed a standardized architecture and was trained using categorical cross-entropy loss and the Adam optimizer, with EarlyStopping enabled for validation-based convergence.

## GloVe Embeddings

To use pretrained GloVe embeddings (100-dimensional), download them from the official Stanford NLP website:  
[https://nlp.stanford.edu/projects/glove/](https://nlp.stanford.edu/projects/glove/)

Unzip the file and place it in a directory named `glove/`, or update the path in the notebook accordingly.

## Results Summary

| Model                                              | Epochs | Train Accuracy | Val Accuracy |
|---------------------------------------------------|--------|----------------|--------------|
| Bi-LSTM + Enhanced Preprocessing and GloVe - Final| 3      | 0.6186         | 0.6223       |
| Bi-LSTM + Enhanced Preprocessing and GloVe        | 3      | 0.6198         | 0.6160       |
| Simple LSTM + GloVe                               | 5      | 0.6168         | 0.6128       |
| LSTM + Enhanced Preprocessing and GloVe           | 5      | 0.5825         | 0.6039       |
| Bi-LSTM + Enhanced Preprocessing                  | 4      | 0.7995         | 0.5957       |
| Bi-RNN + Enhanced Preprocessing and GloVe         | 3      | 0.5401         | 0.5674       |
| Simple LSTM + Simple Preprocessing and GloVe - Final | 3    | 0.5452         | 0.5526       |
| Bi-RNN + Enhanced Preprocessing                   | 5      | 0.9093         | 0.5424       |
| Simple LSTM + Enhanced Preprocessing and GloVe - Final | 3   | 0.5197         | 0.5105       |
| Simple RNN + GloVe                                | 5      | 0.4549         | 0.4520       |
| Simple LSTM + Enhanced Preprocessing              | 5      | 0.4549         | 0.4520       |
| Simple RNN + Enhanced Preprocessing               | 5      | 0.4549         | 0.4520       |
| Simple LSTM (Baseline)                            | 5      | 0.4549         | 0.4520       |
| Simple RNN (Baseline)                             | 5      | 0.4536         | 0.4520       |

The **Bidirectional LSTM with GloVe and enhanced preprocessing** achieved the highest validation accuracy of 62.2%, confirming the advantage of LSTM-based architectures when combined with quality input representations and pretrained embeddings.

