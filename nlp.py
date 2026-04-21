import pandas as pd
import numpy as np
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')
from transformers import pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

nlp_llm = pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")

def get_vader_sentiment(text, scoreType):
    if scoreType not in ['neg', 'neu', 'pos', 'compound']:
        raise ValueError("Invalid scoreType. Must be 'neg', 'neu', 'pos', or 'compound'.")
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(text)
    return scores[scoreType]

def get_llm_sentiment(text, labelOrScore):
    if labelOrScore not in ["label" , "score"]:
        raise ValueError("Invalid labelOrScore. Must be 'label' or 'score'.")
    result = nlp_llm(text)[0]
    if labelOrScore == 'label':
        return result['label']
    else:
        score = result['score']
        if result['label'] == "NEGATIVE":
            score = -1 * score
        return score

def analyzeSentiment(df):
    #Use get_vader_sentiment to analyze sentiment of titles and add columns
    df['vader_positive'] = df['title'].apply(lambda x: get_vader_sentiment(x, 'pos'))
    df['vader_negative'] = df['title'].apply(lambda x: get_vader_sentiment(x, 'neg'))
    df['vader_neutral'] = df['title'].apply(lambda x: get_vader_sentiment(x, 'neu'))
    df['vader_compound'] = df['title'].apply(lambda x: get_vader_sentiment(x, 'compound'))
    df['vader_positive_true'] = (df['vader_compound'] > 0.05).astype(int)
    df['vader_negative_true'] = (df['vader_compound'] < -0.05).astype(int)
    df['llm_sentiment'] = df['title'].apply(lambda x: get_llm_sentiment(x, "label"))
    df['llm_sentiment_score'] = df['title'].apply(lambda x: get_llm_sentiment(x, "score"))
    return df



def modelTopics(df, k_components ):
    vectorizer = CountVectorizer(stop_words="english", max_df=0.95, min_df=10)
    articles = df['text']
    doc_term_matrix = vectorizer.fit_transform(articles)
    lda = LatentDirichletAllocation(n_components=k_components, random_state=42)
    lda.fit(doc_term_matrix)
    topic_distributions = lda.transform(doc_term_matrix)
    topic_columns = [f"Topic_{i}" for i in range(lda.n_components)]
    topic_df = pd.DataFrame(topic_distributions, columns=topic_columns)
    final_df = pd.concat([df, topic_df], axis=1)
    final_df['dominant_topic'] = np.argmax(topic_distributions, axis=1)
    return final_df


def find_best_k(df):
    articles = df['text']

    # 1. Lock in sensible defaults for the vectorizer
    pipeline_step1 = Pipeline([
        ('vect', CountVectorizer(stop_words="english", max_df=0.95, min_df=10)),
        ('lda', LatentDirichletAllocation(random_state=42, learning_decay=0.7, max_iter=10))
    ])

    # 2. ONLY search for the best number of topics
    # This results in just 4 combinations * 3 folds = 12 total fits
    search_params_step1 = {
        'lda__n_components': [5, 10, 15, 20]
    }

    grid_search_1 = GridSearchCV(
        pipeline_step1,
        param_grid=search_params_step1,
        cv=3,
        n_jobs=-1,
        verbose=1
    )

    print("Starting Phase 1: Finding optimal number of topics...")
    grid_search_1.fit(articles)

    best_k = grid_search_1.best_params_['lda__n_components']
    print(f"Phase 1 Complete! Best n_components: {best_k}")

    return best_k

def fine_tune_parameters(df, best_k):
    articles = df['text']

    # 1. Pipeline using the best_k we found in Step 1
    pipeline_step2 = Pipeline([
        ('vect', CountVectorizer(stop_words="english")),
        ('lda', LatentDirichletAllocation(n_components=best_k, random_state=42, max_iter=10))
    ])

    # 2. Search the vectorizer and learning parameters
    # 3 options * 3 options * 3 options * 3 folds = 81 total fits
    search_params_step2 = {
        'vect__max_df': [0.85, 0.90, 0.95],
        'vect__min_df': [5, 10, 15],
        'lda__learning_decay': [0.5, 0.7, 0.9]
    }

    grid_search_2 = GridSearchCV(
        pipeline_step2,
        param_grid=search_params_step2,
        cv=3,
        n_jobs=-1,
        verbose=1
    )

    print(f"Starting Phase 2: Fine-tuning parameters for k={best_k}...")
    grid_search_2.fit(articles)

    print("Phase 2 Complete! Final Optimal Parameters:")
    for param_name in sorted(search_params_step2.keys()):
        print(f"{param_name}: {grid_search_2.best_params_[param_name]}")

    # Returns the fully optimized model ready to use
    return grid_search_2.best_estimator_

# Run this second using the result from the first function
# final_best_model = fine_tune_parameters(df, optimal_k)

def logisticRegression(df, llm = True):


    if llm:
        features_to_use = ['llm_sentiment_score', 'dominant_topic', 'llm_sentiment']
        X = df[features_to_use].copy()
        y = df['is_fake_news'].astype(int)
        X['dominant_topic'] = X['dominant_topic'].astype(str)
        X = pd.get_dummies(X, columns=['dominant_topic', 'llm_sentiment'], drop_first=True)


    else:
        features_to_use = [
        'vader_positive', 'vader_negative', 'vader_neutral', 'dominant_topic']
        X = df[features_to_use].copy()
        y = df['is_fake_news'].astype(int)
        X['dominant_topic'] = X['dominant_topic'].astype(str)
        X = pd.get_dummies(X, columns=['dominant_topic'], drop_first=True)






    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}\n")
    print("Classification Report:\n", classification_report(y_test, y_pred))
