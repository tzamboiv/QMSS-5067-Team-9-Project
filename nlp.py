import pandas as pd

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')
from transformers import pipeline

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
