import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

data = pd.read_csv('C:/Users/Mrlaptop/Downloads/Email Classification/Email Classification/email.csv')

data.isnull().sum()

label_encoder = LabelEncoder()
data['Category'] = label_encoder.fit_transform(data['Category'])

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data['Message'])

y = data['Category']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "SVM": SVC()
}

for model_name, model in models.items():

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)*100
    precision = precision_score(y_test, y_pred)*100
    recall = recall_score(y_test, y_pred)*100
    f1 = f1_score(y_test, y_pred)*100
    
    print(f"{model_name}:\n Accuracy: {accuracy}\n Precision: {precision}\n Recall: {recall}\n F1-score: {f1}\n")

    cv_scores = cross_val_score(model, X, y, cv=5)
    print(f"Cross-validation scores for {model_name}: {cv_scores}")
    print(f"Mean cross-validation score for {model_name}: {cv_scores.mean()}\n")
