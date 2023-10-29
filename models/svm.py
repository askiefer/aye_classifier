from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from models.common import print_metrics, test_sentences


def run_svm(X_train, X_test, y_train, y_test):

    Tfidf_vect = TfidfVectorizer()

    Tfidf_vect.fit(X_train)
    Train_X_Tfidf = Tfidf_vect.transform(X_train)
    Test_X_Tfidf = Tfidf_vect.transform(X_test)

    SVM = svm.SVC(C=1.0, kernel="linear", degree=3, gamma="scale", probability=True)
    SVM.fit(Train_X_Tfidf, y_train)

    # predict the labels on validation dataset
    ytrain_pred = SVM.predict(Train_X_Tfidf)
    ytest_pred = SVM.predict(Test_X_Tfidf)

    print("SVM Metrics:\n")
    print_metrics(y_test, ytest_pred, y_train, ytrain_pred)
    test_sentences(SVM, Tfidf_vect)
