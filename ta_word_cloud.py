import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
import matplotlib.pyplot as plt
from scipy import stats


# Load the transcript
transcript_df = pd.read_csv("C:/Users/camer/Downloads/inside_out_TA.csv")

# Filter for top four characters
top_four_df = transcript_df[transcript_df['character'].isin(["JOY", "ANXIETY", "DISGUST", "ANGER"])]

# 1. Group lines by character
grouped_df = top_four_df.groupby('character')['line'].apply(lambda x: ' '.join(x)).reset_index()

# Initialize lemmatizer and stop words
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))  # Use NLTK's list of stop words

# Function to preprocess text
def preprocess_text(text):
    # Remove punctuation, lowercase, tokenize, remove stop words, and lemmatize
    text = re.sub(r'[^\w\s]', '', text).lower()
    words = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]
    return ' '.join(words)

# Apply preprocessing to each character's dialogue
grouped_df['line'] = grouped_df['line'].apply(preprocess_text)

# 2. Apply TF-IDF
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(grouped_df['line'])

# Convert the matrix to a DataFrame for easier viewing
tfidf_df = pd.DataFrame(tfidf_matrix.todense(), columns=tfidf_vectorizer.get_feature_names_out(), index=grouped_df['character'])

# Display the TF-IDF values
print("TF-IDF Values:")
print(tfidf_df)

# Function to get sentiment
def get_sentiment(text):
    return TextBlob(text).sentiment.polarity 

# Apply sentiment analysis to each character's processed dialogue
grouped_df['sentiment'] = grouped_df['line'].apply(get_sentiment)

import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Function to create a word cloud from TF-IDF data for a single character
def create_wordcloud(character_name, tfidf_df, character_index):
    # Get the TF-IDF scores for the specific character
    character_tfidf = tfidf_df.iloc[character_index]
    
    # Convert to a dictionary for the word cloud
    word_freq = character_tfidf.to_dict()
    
    # Create the word cloud
    wordcloud = WordCloud(width=400, height=400, background_color='white').generate_from_frequencies(word_freq)
    
    # Plot the word cloud
    plt.figure(figsize=(6, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')  # No axis for a cleaner look
    plt.title(f"Word Cloud for {character_name}", fontsize=16)
    plt.show()

# Assuming `tfidf_df` is the DataFrame with the TF-IDF scores for each character's dialogue
characters = grouped_df['character']  # This is your character name column

# Loop over each character and create their word cloud
for index, character_name in enumerate(characters):
    create_wordcloud(character_name, tfidf_df, index)

