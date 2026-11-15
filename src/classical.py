"""Random Forest, Decision Tree and SVM on the 256 dense-layer features of each deep model.

usage: python classical.py cnn | inceptionv3 | xception
"""

import sys
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import config as C
from metrics import log_result, save_probs, load_probs


def main(kind):
    d = load_probs(kind)
    models = {
        "rf": RandomForestClassifier(n_estimators=300, random_state=C.SEED, n_jobs=-1),
        "dt": DecisionTreeClassifier(random_state=C.SEED),
        "svm": make_pipeline(StandardScaler(), SVC(probability=True, random_state=C.SEED)),
    }
    p_val, p_test = [], []
    for name, clf in models.items():
        clf.fit(d["f_train"], d["y_train"])
        joblib.dump(clf, f"{C.OUT_DIR}/models/{kind}_feat_{name}.pkl")
        log_result(f"{kind}_feat_{name}", d["y_test"], clf.predict(d["f_test"]))
        p_val.append(clf.predict_proba(d["f_val"]))
        p_test.append(clf.predict_proba(d["f_test"]))
    vote_val, vote_test = np.mean(p_val, axis=0), np.mean(p_test, axis=0)  # soft vote
    log_result(f"{kind}_feat_softvote", d["y_test"], vote_test.argmax(1))
    save_probs(f"{kind}_feat_softvote", val=vote_val, test=vote_test, y_val=d["y_val"], y_test=d["y_test"])


if __name__ == "__main__":
    main(sys.argv[1])
