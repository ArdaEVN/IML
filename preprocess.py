import sys
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup

# Download necessary NLTK datasets
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('omw-1.4') 

def preprocess_text(text):
    try:
        # Initialize the lemmatizer
        lemmatizer = WordNetLemmatizer()
        
        # Convert to lowercase
        text = text.lower()
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        # Remove HTML tags
        text = BeautifulSoup(text, "html.parser").text
        # Remove punctuation and special characters
        text = re.sub(r'[^\w\s]', '', text)
        # Tokenize the text
        word_tokens = word_tokenize(text)
        # Load stopwords
        stop_words = set(stopwords.words('english'))
        # Remove stopwords and lemmatize the words
        filtered_text = [lemmatizer.lemmatize(word) for word in word_tokens if word not in stop_words]
        # Join words back to string
        return ' '.join(filtered_text)
    except Exception as e:
        print(f"Error during text preprocessing: {e}")
        return ''


if __name__ == "__main__":
    input_filename = "230000_monopoly.csv"  # Use your dataset's name directly
    output_filename = "updated_reviews_230000_monopoly.csv"  # Output file name

    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    try:
        df = pd.read_csv(input_filename)
        # Assuming the reviews are in the 'content' column
        df['processed_reviews'] = df['content'].fillna('').apply(preprocess_text)
        df.to_csv(output_filename, index=False)
        print(f"Processed reviews saved to {output_filename}")
    except FileNotFoundError:
        print(f"File not found: {input_filename}")
    except pd.errors.EmptyDataError:
        print(f"No data: {input_filename} is empty.")
    except Exception as e:
        print(f"An error occurred: {e}")
