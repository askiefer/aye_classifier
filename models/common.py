import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    f1_score,
)


def print_metrics(y_test, y_pred, y_train, ytrain_pred):
    # Analyzing
    cm = confusion_matrix(y_test, y_pred)
    pd.DataFrame(confusion_matrix(y_test, y_pred, labels=[0, 1]))
    print(f"Confusion Matrix :\n {cm}")
    print(f"Test Set Accuracy Score :\n {accuracy_score(y_test, y_pred)}\n")
    print(f"Train Set Accuracy Score :\n {accuracy_score(y_train, ytrain_pred)}\n")
    print("F1 Score -> ", f1_score(y_pred, y_test), "\n")
    print(f"Classification Report :\n {classification_report(y_test, y_pred)}\n")


def test_sentences(model, tfidf_vect):
    sentences = [
        "lets call the votes",
        "lost my sight",
        "call the role",
        "66 to seven nays",
        "are on my head",
        "sb 221 has 33",
    ]
    labels = ["aye", "eye"]
    transformed = sentences
    if tfidf_vect:
        transformed = tfidf_vect.transform(sentences)
    prediction = model.predict(transformed)
    prediction_probabilities = model.predict_proba(transformed)

    for idx, sentence in enumerate(sentences):
        prediction_label = labels[prediction[idx]]
        print(f"Sentence: {sentence}")
        print(f"aye probability: {prediction_probabilities[idx][0]}")
        print(f"eye probability: {prediction_probabilities[idx][1]}")
        print(f"Prediction: {prediction_label}\n")
