# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Step 1: Load Dataset
ds = pd.read_csv(r'/content/spam_labelled.csv')

# Step 2: Explore and Clean the Data
print(ds.info())  # Dataset info
print(ds.columns)  # List of columns
print(ds.isna().sum())  # Check for missing values

# Create binary labels for spam classification
ds['Spam'] = ds['Category'].apply(lambda x: 1 if x == 'spam' else 0)

# Visualizing the Data
plt.pie(ds['Category'].value_counts(), labels=['Not Spam', 'Spam'], autopct="%0.2f%%")
plt.title("Spam vs Not Spam Distribution")
plt.show()

# Step 3: Splitting the Data
X_train, X_test, y_train, y_test = train_test_split(ds.Message, ds.Spam, test_size=0.25, random_state=42)

# Step 4: Building the Model
clf = Pipeline([
    ('vectorizer', CountVectorizer()),  # Convert text to numerical features
    ('nb', MultinomialNB())             # Naive Bayes classifier
])

# Step 5: Train the Model
clf.fit(X_train, y_train)

# Step 6: Testing the Model with Example Emails
emails = [
    'Dear Sir/Madam, I am interested in internship opportunities in machine learning at your university',
    'Will u meet ur dream partner soon? Is ur career off 2 a flyng start? 2 find out free, txt HORO followed by ur star sign, e. g. HORO ARIES',
    'Internship offer! Submit your details to be considered for a premium internship program.',
    'Unlock your career! Click here for quick internship approvals. Limited time only!',
    'Hurry! Limited-time offer. Get a 50% discount on all products. Visit [website].'
]

predictions = clf.predict(emails)
print("Predictions for test emails:")
for email, label in zip(emails, predictions):
    print(f"Email: {email}\nSpam: {'Yes' if label == 1 else 'No'}\n")

# Step 7: Validating the Model on Dataset Rows
dc = pd.read_csv(r"/content/test_unlabelled.csv")
asc = []  # List to hold rows as lists
for _, row in dc.iterrows():
    asc.append(row.tolist())

print("\nPredictions for first 20 rows in dataset:")
for i in range(20):
    print(f"Message: {asc[i][1]} - Prediction: {'Spam' if clf.predict([asc[i][1]])[0] == 1 else 'Not Spam'}")

# Step 8: Evaluating the Model
accuracy = clf.score(X_test, y_test) * 100
print(f"\nModel Accuracy on Test Set: {accuracy:.2f}%")
