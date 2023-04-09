# Movie Review Sentiment Analysis (NLP) - Study Project

This study project aims to build a natural language processing model to
classify movie reviews as positive or negative based on their text content. The
project uses various text preprocessing techniques, feature extraction methods,
and machine learning algorithms for model training and evaluation. The advice
of OpenAI's GPT-4 was used throughout the development process, assisting in the
learning and coding stages.

## Dataset

The Large Movie Review Dataset is used for this study project. The dataset
consists of 50,000 movie reviews, equally divided into 25,000 training and
25,000 test sets. Each set contains an equal number of positive and negative
reviews. The dataset can be found at [this link](https://ai.stanford.edu/~amaas/data/sentiment/), and the paper describing
the dataset can be accessed [here](http://www.aclweb.org/anthology/P11-1015).

## Files

- `spacy_preprocess.py`: Preprocessing using SpaCy library.
- `nltk_preprocess.py`: Preprocessing using NLTK library.
- `train.py`: Training using logistic regression and Naive Bayes.
- `train_transformers.py`: Training using Transformers (DistilBert).

## Preprocessing

Two different preprocessing scripts are provided, one using the NLTK library
and the other using the SpaCy library. Both scripts perform text preprocessing
tasks such as tokenization, stop word removal, and stemming/lemmatization.

## Training

The `train.py` script trains machine learning models using logistic regression
and Naive Bayes. These models are trained on the preprocessed movie review data
and their performance is evaluated using metrics like accuracy, precision,
recall, and F1-score.

The `train_transformers.py` script trains a Transformers-based model
(DistilBert) on the preprocessed movie review data. This script demonstrates
the use of a more advanced NLP technique for sentiment analysis and achieves
similar accuracy compared to the other models.
The poor results may be explained by insufficient training data, time,
lack of hyperparameter tuning, text preprocessing or model's architecture.

## Results

The following results were achieved during the development of this study project:

- Logistic Regression: 
  - Accuracy: 0.87188
  - Precision: 0.84923
  - Recall: 0.90458
  - F1-score: 0.87590

- Naive Bayes: 
  - Accuracy: 0.8336
  - Precision: 0.86872
  - Recall: 0.78616
  - F1-score: 0.82517

- Transformers (DistilBert): 
  - Accuracy: 0.8646

## Usage

1. Train the machine learning models using the `train.py` script:

    python train.py

2. Train the Transformers-based model using the `train_transformers.py` script:

    python train_transformers.py

3. Evaluate the performance of the trained models and compare their results.

## Requirements

- Python 3.6 or later
- SpaCy
- NLTK
- NumPy
- Pandas
- Scikit-learn
- Transformers

## License

This project is licensed under the [MIT License](LICENSE).

