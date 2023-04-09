import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
# Alternative way: Use lemmatization
from nltk.stem import WordNetLemmatizer

# Download required resources
nltk.download('punkt')
nltk.download('stopwords')

# Download required resources
nltk.download('wordnet')

# Define a sample text
sample_text = "This is an example of text preprocessing using Python and NLTK."

def preprocess_text(text, method = "lemmatizer"):
    # Step 1: Lowercase the text
    lowercase_text = text.lower()
    
    # Step 2: Remove punctuation and special characters
    import string
    cleaned_text = "".join([char for char in lowercase_text if char not in string.punctuation])
    
    # Step 3: Tokenize the text
    tokens = word_tokenize(cleaned_text)
    
    # Step 4: Remove stop words
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token not in stop_words]
    
    if method == "stemmer":
        # Step 5: Stemming or lemmatizing
        stemmer = PorterStemmer()
        stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]
        return ' '.join(stemmed_tokens)
    elif method == "lemmatizer":
        lemmatizer = WordNetLemmatizer()
        lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
        return ' '.join(lemmatized_tokens)

if __name__=="__main__":
    print(preprocess_text(sample_text))
    print(preprocess_text(sample_text, method="lemmatizer"))
