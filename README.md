 SVM – Email Spam Classification

 App link: https://spamdetection-gmnjudr9kn7e6xnz5doytz.streamlit.app/

 
📌 Problem Statement

We want to build an SVM classifier to detect whether an email is Spam (1) or Not Spam (0).

Features used:

word_freq_free – frequency of the word “free” in the email

word_freq_money – frequency of the word “money”

word_freq_offer – frequency of the word “offer”

email_length – total length of the email

Target:

Spam (1)

Not Spam (0)

⚙️ Steps Performed

Load Dataset

import pandas as pd
data = pd.read_csv("email_spam_svm.csv")


Split Features and Target

X = data[["word_freq_free", "word_freq_money", "word_freq_offer", "email_length"]]
y = data["Spam"]


Preprocessing (Scaling)

Since SVM is sensitive to feature scales, we standardize features.

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


Train-Test Split

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


Model Training with 5-Fold Cross Validation

from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

model = SVC(probability=True, kernel="rbf", random_state=42)

acc_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
auc_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="roc_auc")

print("CV Accuracy:", acc_scores.mean())
print("CV ROC-AUC:", auc_scores.mean())


Final Model Fit

model.fit(X_train, y_train)


Evaluation on Test Set

from sklearn.metrics import accuracy_score, roc_auc_score

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:,1]

print("Test Accuracy:", accuracy_score(y_test, y_pred))
print("Test ROC-AUC:", roc_auc_score(y_test, y_prob))


Prediction on New Email
Example email: (free=2.1, money=1.3, offer=0.7, length=180)

new_email = [[2.1, 1.3, 0.7, 180]]
new_email_scaled = scaler.transform(new_email)

prediction = model.predict(new_email_scaled)
prob = model.predict_proba(new_email_scaled)[0,1]

print(f"Prediction: {'Spam' if prediction[0]==1 else 'Not Spam'} (Prob={prob:.2f})")

📊 Expected Output

5-Fold CV Accuracy ≈ (depends on dataset)

5-Fold CV ROC-AUC ≈ (depends on dataset)

Test Accuracy & ROC-AUC reported.

New Email Prediction Example

Prediction: Spam (Prob=0.92)

Confusion Matrix Heatmap
<img width="501" height="393" alt="image" src="https://github.com/user-attachments/assets/71ad9705-109d-4860-b0a9-9be82b2835d9" />


ROC Curve

<img width="536" height="470" alt="image" src="https://github.com/user-attachments/assets/697f14a6-9940-4acd-96a9-1c91bb3b3eac" />


✅ Conclusion

SVM with RBF kernel performs well for spam classification.

Both Accuracy and ROC-AUC should be reported for robust evaluation.

The given new email (free=2.1, money=1.3, offer=0.7, length=180) is predicted as Spam with high probability.

