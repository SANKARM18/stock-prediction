import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')

# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()

def analyze_sentiment(tweet_data, date):
    # Filter tweets based on the provided date
    filtered_tweets = tweet_data.filter(tweet_data['Date'] == date).collect()

    # Aggregate sentiment score for all tweets on the given date
    sentiment_score = 0
    for tweet in filtered_tweets:
        score = sia.polarity_scores(tweet['Tweet'])['compound']
        sentiment_score += score

    return sentiment_score / len(filtered_tweets) if filtered_tweets else 0
