import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import make_pipeline

def preprocess_text(text):
    # Assuming text is already preprocessed in the 'processed_reviews' column
    return text

# Load dataset
df = pd.read_csv("updated_reviews_230000_monopoly.csv")

# Preprocess the reviews (if needed, we are assuming it's already done)
df['processed_reviews'] = df['processed_reviews'].fillna('')

# Split data into training and testing sets
X = df['processed_reviews']
y = df['score']  # assuming 'score' is the label you want to predict
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a k-Nearest Neighbors classifier pipeline
pipeline = make_pipeline(TfidfVectorizer(stop_words='english'), KNeighborsClassifier())

# Train the model
pipeline.fit(X_train, y_train)

# Predict on the test set
y_pred = pipeline.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"k-Nearest Neighbors Model Accuracy: {accuracy * 100:.2f}%")
print(classification_report(y_test, y_pred))

# Save the trained model (optional)
import joblib
joblib.dump(pipeline, 'knn_model.pkl')
