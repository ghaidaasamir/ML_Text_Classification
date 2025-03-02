# Traditional ML Pipeline for Text Classification

This project implements a traditional machine learning pipeline for text classification. Text preprocessing, feature extraction using TF-IDF, model training with several classifiers, evaluation of model performance, and saving the complete pipeline for future use.

## Process

- **Enhanced Text Preprocessing**  
  - Preprocess the text data.
  - Each phrase is converted to lowercase, stripped of extra spaces, tokenized, and lemmatized.
  - Stop words and punctuation are removed.
  - The cleaned text is stored in a new column called `clean_phrase`.

- **Data Splitting**  
  - The dataset is read from `overview-of-recordings.csv`.
  - The data is split into training (80%), validation (10%), and test (10%) sets using `train_test_split`.

- **Feature Engineering**  
  - A TF-IDF vectorizer converts the preprocessed text into numerical features.
  - The vectorizer is set to use a maximum of 5000 features, unigrams, and bigrams with sublinear term frequency scaling.

- **Label Encoding**  
  - The target labels (from the `prompt` column) are encoded into numeric values using `LabelEncoder`.

- **Model Building and Training**  
  - Three classifiers are trained on the training data:
    - Logistic Regression
    - Random Forest Classifier
    - XGBoost Classifier
  - Each model is evaluated on the validation set using accuracy and weighted F1-score.
  - The best model (highest F1-score) is selected.

- **Model Evaluation**  
  - The selected best model is evaluated on the test set to determine its final performance.
  - Evaluation metrics include accuracy and weighted F1-score.

- **Pipeline Serialization**  
  - The final pipeline is saved using joblib for later use.

## Requirements

- Python 3.x
- Pandas
- Scikit-learn
- spaCy
- XGBoost
- Joblib
