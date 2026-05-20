import pandas as pd
import pickle
import spacy
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
import numpy as np
import json

nlp = spacy.load("en_core_web_sm")

# Loading dataset and checking for nulls
df = pd.read_csv("merged_synthetic_symptoms.csv")
print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Nulls: {df.isnull().sum().sum()}")
df.dropna()
df.reset_index(drop=True)

#Cleaning text
def clean_text(text):
    text = text.lower()
    
    cleaned_chars = []
    for char in text:
        if char.isalpha() or char == " ":
            cleaned_chars.append(char)
        else:
            cleaned_chars.append(" ")   #replace punctuation and numbers with a space
    
    text = "".join(cleaned_chars)
    text = " ".join(text.split()) # making sure its single spaced
    
    return text

df["cleaned"] = df["symptom_description"].apply(clean_text)

print("Cleaning complete")
print("Before cleaning: ", df["symptom_description"][0])
print("After cleaning: ", df["cleaned"][0])

#Lemmatise and stop word removal while keeping negations

negations = {"no", "not", "without", "never", "nor", "neither", "cannot"}

print("Lemmatising started...")

docs = list(nlp.pipe(df["cleaned"], batch_size=256))

processed = []

for doc in docs:
    tokens = []
    
    for token in doc:
        #keeping negations as it is
        if token.text in negations:
            tokens.append(token.text)
            
        #otherwise, convert and keep lemma
        else:
            if len(token.text) > 1 and not token.is_stop:
                tokens.append(token.lemma_)
            
    processed_text = " ".join(tokens)
    processed.append(processed_text)

df["processed"] = processed
print(df["processed"][0])
print("Lemma and stopword removal complete")

#Check if negations are kept

test_clean = clean_text("i have a runny nose and no fever and not puking")
test_doc = nlp(test_clean)

tokens = []
for token in test_doc:
    if token.text in negations:
        tokens.append(token.text)
    else:
        if (not token.is_stop and len(token.text)>1):
            tokens.append(token.lemma_)
            
test_result = " ".join(tokens)
print(test_result)
# should keep no and not

#Encoding the labels

le = LabelEncoder() #creating the encoder

df["num_label"] = le.fit_transform(df["disease_label"]) #convert diseasenames into numbers

#saving the encoder to a file to use later
with open("label_encoder.pickle", "wb") as f:
    pickle.dump(le, f) 

#printing all the diseases classes
print(f"{len(le.classes_)}")

#example
print(f"Example: Migraine Transformed is: {le.transform(["Migraine"])[0]}")

# Splitting the dataset into 70/15/15 for training testing and validation

X = df["processed"] # cleaned text
y = df["num_label"] # disease numbers

#Two splits to hide test as train_test_split cannot split 3 way

#First split: Training and Validation against Test 
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y,
    test_size=0.15,  #15% for test set
    stratify=y,  #balance diseases
    random_state=1  #same split every run
)

#Second split: Train against Val
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval,
    test_size=0.176,  #17.6% for validation
    stratify=y_trainval,  #balance diseases
    random_state = 1  #same split every run
)

#showing sizes
print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

#Feature input preparations for 3 architectures
print("Feature input preparations starting...")

# TF-IDF for (BiLSTM)
tfidf = TfidfVectorizer(
    max_features=5000, #using the top 5000 most import words
    ngram_range=(1,2) #using single words and 2-word phrases
)

#Learning from training data and convert to numbers
X_train_tfidf = tfidf.fit_transform(X_train)

#Convert validation and test data using tfidf object learned from train (don't need to learn from these directly)
X_val_tfidf = tfidf.transform(X_val)
X_test_tfidf = tfidf.transform(X_test)

# save the convertor to use later (user input)
with open("tfidf_vectorizer.pickle", "wb") as f:
    pickle.dump(tfidf, f)
    
#showing the result
print(f"TF-IDF shape: {X_train_tfidf.shape}")

#Word2Vec embeddings (CNN-LSTM)
tokenised = [t.split() for t in X_train] #split training text into words for Word2Vec

#Training Word2Vec model on training words
w2v = Word2Vec(
    sentences=tokenised, # the word lists from training
    vector_size=100, #each word is 100 numbers
    window=5, #5 words near it
    min_count=1, #using rare words too 
    workers=4,  #4 CPU cores
    seed=1 #same results every time
)

w2v.save("word2vec.model") #saving model

#convert text to numbers
word_vector_size = 100 #word vector size 
max_len = 30 #max 30 words per sentence

def to_w2v(texts, model, max_len= max_len, size=word_vector_size):
    res = []
    for text in texts:
        tokens = text.split()[:max_len]
        matrix = np.zeros((max_len, size)) #empty matrix
        for i, token in enumerate(tokens):
            if token in model.wv:
                matrix[i] = model.wv[token]
        res.append(matrix)
    return np.array(res)
        
#Converting all the datasets
X_train_w2v = to_w2v(X_train, w2v)
X_val_w2v = to_w2v(X_val, w2v)
X_test_w2v = to_w2v(X_test, w2v)

print(f"W2V shape: {X_train_w2v.shape}")

#BERT text (cleaned only, lemmatised it not ideal)
X_train_bert = df.loc[X_train.index, "cleaned"].tolist()
X_val_bert = df.loc[X_val.index, "cleaned"].tolist()
X_test_bert = df.loc[X_test.index, "cleaned"].tolist()

print(f"BERT text: {len(X_train_bert)}")

#Saving everything as numpy file to train
df.to_csv("symptom_disease_processed.csv", index=False)

np.save("X_train_tfidf.npy", X_train_tfidf.toarray())
np.save("X_val_tfidf.npy", X_val_tfidf.toarray())
np.save("X_test_tfidf.npy", X_test_tfidf.toarray())

np.save("X_train_w2v.npy", X_train_w2v)
np.save("X_val_w2v.npy", X_val_w2v)
np.save("X_test_w2v.npy", X_test_w2v)

np.save("y_train.npy", y_train.values)
np.save("y_val.npy", y_val.values)
np.save("y_test.npy", y_test.values)

with open("bert_splits.json", "w") as f:
    json.dump({
        "X_train": X_train_bert,
        "X_val": X_val_bert,
        "X_test": X_test_bert
    }, f)
    
print("All files processed and saved, ready to train.")

#####
print("Processing external test set.")

ext_df = pd.read_csv("heldout_synthetic_gemini.csv")
print(f"External test rows: {len(ext_df)}")
print(f"Label mismatch check: {set(ext_df["disease_label"].unique()) - set(le.classes_)}")

#Clean
ext_df["cleaned"] = ext_df["symptom_description"].apply(clean_text)

#Lemmatise
ext_docs = list(nlp.pipe(ext_df["cleaned"], batch_size=256))
ext_processed = []
for doc in ext_docs:
    tokens = []
    for token in doc:
        if token.text in negations:
            tokens.append(token.text)
        else:
            if len(token.text) > 1 and not token.is_stop:
                tokens.append(token.lemma_)
    ext_processed.append(" ".join(tokens))
    
ext_df["processed"] = ext_processed

#Encoding labels
y_ext = le.transform(ext_df["disease_label"])

#TF-IDF (BiLSTM)
X_ext_tfidf = tfidf.transform(ext_df["processed"]).toarray()

#Word2Vec (CNN-LSTM)
X_ext_w2v = to_w2v(ext_df["processed"], w2v)

#BERT - cleaned only
X_ext_bert = ext_df["cleaned"].tolist()

#save
np.save("X_ext_tfidf.npy", X_ext_tfidf)
np.save("X_ext_w2v.npy", X_ext_w2v)
np.save("y_ext.npy", y_ext)

with open("bert_ext.json", "w") as f:
    json.dump({"X_ext": X_ext_bert}, f)

print(f"External test saved - TF-IDF: {X_ext_tfidf.shape} W2V: {X_ext_w2v.shape} BERT: {len(X_ext_bert)}")
print("All external test files are ready")
