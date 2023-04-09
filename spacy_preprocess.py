import spacy

# Load the English language model
nlp = spacy.load("en_core_web_sm")

# Define a sample text
sample_text = "This is an example of text preprocessing using Python and spaCy."

def preprocess_text(text):
    # Process the text with spaCy
    doc = nlp(text)

    # Filter tokens and lemmatize
    filtered_tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]

    # Print the final preprocessed tokens
    return ' '.join(filtered_tokens)

if __name__=="__main__":
    print(preprocess_text(sample_text))
