import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import pickle

df = pd.read_csv('data/dataset.csv')
df.columns = df.columns.str.strip()
symptom_cols = [col for col in df.columns if col.startswith('Symptom')]

all_symptoms = []
for col in symptom_cols:
    all_symptoms.extend(df[col].dropna().astype(str).tolist())
all_symptoms = sorted(set([s.strip().lower() for s in all_symptoms]))

severity_df = pd.read_csv('data/Symptom-severity.csv')
severity_map = dict(zip(
    severity_df['Symptom'].astype(str).str.strip().str.lower(),
    severity_df['weight']
))

binary_df = pd.DataFrame(0, index=df.index, columns=all_symptoms)
for col in symptom_cols:
    for idx, val in df[col].dropna().items():
        symptom = str(val).strip().lower()
        binary_df.at[idx, symptom] = severity_map.get(symptom, 1)

binary_df['disease'] = df['Disease'].str.strip()
le = LabelEncoder()
binary_df['disease_encoded'] = le.fit_transform(binary_df['disease'])

X = binary_df[all_symptoms]
y = binary_df['disease_encoded']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

rf = RandomForestClassifier(random_state=42)
nb = GaussianNB()

rf.fit(X_train, y_train)
nb.fit(X_train, y_train)

params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}

grid = GridSearchCV(RandomForestClassifier(random_state=42), params, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)
rf_best = grid.best_estimator_

dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

voting_clf = VotingClassifier(
    estimators=[('dt', dt), ('rf', rf_best), ('nb', nb)],
    voting='soft'
)
voting_clf.fit(X_train, y_train)

rf_best_acc = accuracy_score(y_test, rf_best.predict(X_test))
voting_acc = accuracy_score(y_test, voting_clf.predict(X_test))
best_model = voting_clf if voting_acc > rf_best_acc else rf_best

with open('model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

with open('columns.pkl', 'wb') as f:
    pickle.dump(all_symptoms, f)

with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)
