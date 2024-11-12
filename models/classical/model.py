import pandas as pd
import pickle
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

# Step 1: Load your dataset
df = pd.read_csv('train.csv')
df = df.dropna(subset=['crimeaditionalinfo'])

# Step 2: Remove punctuation from the 'processed_info' column
df['crimeaditionalinfo'] = df['crimeaditionalinfo'].str.translate(str.maketrans('', '', string.punctuation))

# Step 3: Encode the target columns (category and sub_category) as numeric values
category_encoder = LabelEncoder()
sub_category_encoder = LabelEncoder()

df['category'] = category_encoder.fit_transform(df['category'])
df['sub_category'] = sub_category_encoder.fit_transform(df['sub_category'])

# Step 4: Split the data into train and test sets
X = df['crimeaditionalinfo']  # Input text
y = df[['category', 'sub_category']]  # Output labels (multi-label)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)  # Limit to top 5000 words

# Step 6: Build the pipeline
# We will use a pipeline that first transforms the data (TF-IDF) and then applies the classifier
pipeline = Pipeline([
    ('tfidf', vectorizer),
    ('clf', MultiOutputClassifier(LogisticRegression(max_iter=1000)))
])

# Step 7: Train the model
pipeline.fit(X_train, y_train)

# Step 8: Save the model, vectorizer, and encoders
with open('crime_model.pkl', 'wb') as f:
    pickle.dump(pipeline, f)

with open('category_encoder.pkl', 'wb') as f:
    pickle.dump(category_encoder, f)

with open('sub_category_encoder.pkl', 'wb') as f:
    pickle.dump(sub_category_encoder, f)

print("Model and encoders have been saved!")
