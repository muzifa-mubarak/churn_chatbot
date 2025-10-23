import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib
from sklearn.metrics import accuracy_score
import pickle
###added
from transformers import pipeline


#######
# Load data
df = pd.read_csv("churn_data_cleaned.csv",encoding='latin1')

# Drop ID & text
df = df.drop(columns=["CustomerID","Mobile","Email"]) ##FeedbackText is not dropped here

# Encode target
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

###added
sentiment_pipeline = pipeline("sentiment-analysis")

# Analyze text and assign score to a new column
def transformers_sentiment_score(text):
    if isinstance(text, str):
        # The pipeline returns a list of dictionaries, we take the score of the first result
        result = sentiment_pipeline(text)[0]
        # Map sentiment labels to numerical scores (you can adjust this mapping)
        if result['label'] == 'POSITIVE':
            return result['score']
        elif result['label'] == 'NEGATIVE':
            return -result['score']
        else: # Neutral or other labels
            return 0.0
    else:
        return 0.0 # Return 0 for non-string values or missing data

df['SentimentScore'] = df['FeedbackText'].apply(transformers_sentiment_score)

#now drop 
df = df.drop(['FeedbackText'], axis=1, errors='ignore')

# Encode categoricals
categorical_cols = ["Gender", "SubscriptionType", "PaymentMethod", "AutoRenew"]
encoder = LabelEncoder()
for col in categorical_cols:
    df[col] = encoder.fit_transform(df[col])

# Split
X = df.drop(columns=["Churn"])
y = df["Churn"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")


joblib.dump(model, open("model_name.pkl", "wb"))
joblib.dump(encoder, open("encoder_name.pkl", "wb"))