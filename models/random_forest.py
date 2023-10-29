from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import TfidfVectorizer
from models.common import print_metrics, test_sentences


def run_random_forest(X_train, X_test, y_train, y_test):

    classifier_rf = Pipeline(
        [
            ("tfidf", TfidfVectorizer()),
            (
                "clf",
                RandomForestClassifier(
                    bootstrap=False, criterion="entropy", n_estimators=100
                ),
            ),
        ]
    )
    classifier_rf.fit(X_train, y_train)

    # Predicting
    ytest_pred = classifier_rf.predict(X_test)
    ytrain_pred = classifier_rf.predict(X_train)
    print("Random Forest Metrics:\n")
    print_metrics(y_test, ytest_pred, y_train, ytrain_pred)
    test_sentences(classifier_rf, None)
