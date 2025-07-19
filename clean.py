import pandas as pd
import re
import csv
import sys
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Increase max CSV field size to handle long text
csv.field_size_limit(2**25)

# Load CSV using Python engine and handle multi-line quoted fields
df = pd.read_csv('TREC-05.csv', engine='python', quotechar='"')

# Remove 'receiver' and 'date' columns
df = df.drop(['receiver', 'date'], axis=1)

# Remove duplicate rows
df = df.drop_duplicates()

# Drop rows with any missing values
df = df.dropna()

# Define text cleaning function
def clean_text(text):
    text = str(text)
    text = text.encode('ascii', errors='ignore').decode('ascii')  # Remove non-ASCII characters
    text = text.lower()                                           # Make lowercase
    text = re.sub(r'[^\w\s]', '', text)                           # Remove punctuation
    text = re.sub(r'\d+', '', text)                               # Remove numbers
    words = text.split()
    words = [word for word in words if word not in ENGLISH_STOP_WORDS]  # Remove stop words
    cleaned_text = ' '.join(words)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()      # Remove extra whitespace
    return cleaned_text

# Apply text cleaning to 'subject' and 'body'
df['subject'] = df['subject'].apply(clean_text)
df['body'] = df['body'].apply(clean_text)

# 🔥 Remove rows where 'subject' or 'body' is blank after cleaning
df = df[(df['subject'].str.strip() != '') & (df['body'].str.strip() != '')]

# Preview cleaned dataset
print(df.head())

# Save cleaned dataset
df.to_csv('cleansed_TREC-05.csv', index=False)
