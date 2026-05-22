"""
Classical ML ensemble: RF + DT + SVM with soft voting.
Used by all three ensemble training scripts.
"""
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC


def build_classical_ensemble():
    """
    Return an unfitted VotingClassifier(RF + DT + SVM, voting='soft').
    Call .fit(X_train, y_train) after receiving features from a deep extractor.
    """
    rf  = RandomForestClassifier(n_estimators=100, random_state=42)
    dt  = DecisionTreeClassifier(random_state=42)
    svm = SVC(probability=True, random_state=42)

    return VotingClassifier(
        estimators=[('rf', rf), ('dt', dt), ('svm', svm)],
        voting='soft',
    )
