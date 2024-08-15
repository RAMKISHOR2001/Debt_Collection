import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import spacy

# Download necessary NLTK data
nltk.download('vader_lexicon')
nltk.download('punkt')

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

def generate_sample_communications(n_samples=1000):
    communications = []
    for _ in range(n_samples):
        comm_type = np.random.choice(['email', 'phone', 'letter'])
        if comm_type == 'email':
            communications.append(f"Dear customer, this is a reminder about your outstanding debt of ${np.random.randint(100, 10000)}. Please contact us to discuss payment options.")
        elif comm_type == 'phone':
            communications.append(f"Hello, I'm calling about your unpaid balance of ${np.random.randint(100, 10000)}. Can we schedule a payment?")
        else:
            communications.append(f"Official notice: Your account is overdue by ${np.random.randint(100, 10000)}. Please remit payment immediately to avoid further action.")
    return communications

def perform_sentiment_analysis(texts):
    sia = SentimentIntensityAnalyzer()
    sentiments = [sia.polarity_scores(text)['compound'] for text in texts]
    return sentiments

def extract_key_phrases(texts):
    key_phrases = []
    for text in texts:
        doc = nlp(text)
        phrases = [chunk.text for chunk in doc.noun_chunks if len(chunk.text.split()) > 1]
        key_phrases.append(', '.join(phrases))
    return key_phrases

def perform_topic_modeling(texts, n_topics=5):
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(texts)
    
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(tfidf_matrix)
    
    feature_names = vectorizer.get_feature_names_out()
    topics = []
    for topic_idx, topic in enumerate(lda.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-10 - 1:-1]]
        topics.append(f"Topic {topic_idx + 1}: {', '.join(top_words)}")
    
    return topics

def analyze_communications(communications):
    sentiments = perform_sentiment_analysis(communications)
    key_phrases = extract_key_phrases(communications)
    topics = perform_topic_modeling(communications)
    
    results = pd.DataFrame({
        'communication': communications,
        'sentiment': sentiments,
        'key_phrases': key_phrases
    })
    
    return results, topics

if __name__ == "__main__":
    sample_communications = generate_sample_communications()
    results, topics = analyze_communications(sample_communications)
    
    print("Sample results:")
    print(results.head())
    
    print("\nIdentified topics:")
    for topic in topics:
        print(topic)