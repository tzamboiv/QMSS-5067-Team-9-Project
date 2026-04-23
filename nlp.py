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
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from scipy.stats import loguniform, randint

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

def rf(df, llm = True):
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
    model = RandomForestClassifier(random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}\n")
    print("Classification Report:\n", classification_report(y_test, y_pred))

def gnb(df, llm = True):
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
    model = GaussianNB(var_smoothing=1e-9)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}\n")
    print("Classification Report:\n", classification_report(y_test, y_pred))

def trainEnsemble(df, llm = True, var_smoothing = 2.348881295853314e-06, C = 14.528246637516036, max_iter = 1000, solver='liblinear', max_depth=10, min_samples_leaf=3, min_samples_split=4, n_estimators=255):
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
    gnb = GaussianNB(var_smoothing = var_smoothing)
    gnb.fit(X_train, y_train)
    lr = LogisticRegression(C = C, max_iter = 1000, solver = solver)
    lr.fit(X_train, y_train)
    rf = RandomForestClassifier(max_depth = max_depth, min_samples_leaf = min_samples_leaf, min_samples_split = min_samples_split, n_estimators = n_estimators)
    rf.fit(X_train, y_train)

    voting_ensemble = VotingClassifier(
    estimators=[
        ('GNB', gnb),
        ('Logistic', lr),
        ('Random Forest', rf)
    ],
    voting='hard' )
    voting_ensemble.fit(X_train, y_train)
    y_pred_voting = voting_ensemble.predict(X_test)

    print(f"Voting Ensemble Accuracy: {accuracy_score(y_test, y_pred_voting):.4f}")
    print("Classification Report:\n", classification_report(y_test, y_pred_voting))
    return voting_ensemble

def hyperparameterTuneAllThreeModels(df, llm = True):
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
    gnb = GaussianNB()
    lr = LogisticRegression(max_iter=1000)
    rf = RandomForestClassifier()

    param_dist_gnb = {
    'var_smoothing': loguniform(1e-9, 1e0)
    }

    param_dist_lr = {
    'solver': ['liblinear'],
    'penalty': ['l1', 'l2'],
    'C': loguniform(1e-3, 1e2)
    }

    param_dist_rf = {
    'n_estimators': randint(50, 300),
    'max_depth': randint(3, 20),
    'min_samples_split': randint(2, 11),
    'min_samples_leaf': randint(1, 5)
    }
    models_and_params = {
    'Gaussian Naive Bayes': (gnb, param_dist_gnb, 20), # (model, parameters, n_iter)
    'Logistic Regression': (lr, param_dist_lr, 20),
    'Random Forest': (rf, param_dist_rf, 30) # Giving RF a few more iterations since it's more complex
    }
    best_estimators = {}
    print("--- Starting Randomized Search ---")
    for name, (model, params, n_iters) in models_and_params.items():
        print(f"Tuning {name}...")

        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=params,
            n_iter=n_iters,   # How many random combinations to test
            cv=5,             # 5-fold cross-validation
            scoring='accuracy',
            n_jobs=-1,        # Use all CPU cores
            random_state=42   # For reproducibility
            )

        random_search.fit(X_train, y_train)

            # Save the best trained model
        best_estimators[name] = random_search.best_estimator_

        print(f"  Best Accuracy: {random_search.best_score_:.4f}")
        print(f"  Best Params: {random_search.best_params_}\n")

            # 5. Evaluate the tuned models on the test set
        print("--- Final Test Set Performance ---")
        for name, best_model in best_estimators.items():
            y_pred = best_model.predict(X_test)
            test_acc = accuracy_score(y_test, y_pred)

            print(f"{name}: {test_acc:.4f}")
            print("Classification Report:\n", classification_report(y_test, y_pred))
    return best_estimators
