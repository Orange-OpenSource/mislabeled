# %%
import warnings

import matplotlib.pyplot as plt
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import balanced_accuracy_score

from mislabeled.datasets.wrench import fetch_wrench
from mislabeled.detect.detectors import LinearVoSG
from mislabeled.preprocessing import WeakLabelEncoder

seed = 42

all = fetch_wrench("sms")
train = fetch_wrench("sms", split="train")
validation = fetch_wrench("sms", split="validation")
test = fetch_wrench("sms", split="test")

tfidf = TfidfVectorizer(
    strip_accents="unicode", stop_words="english", min_df=5, max_df=0.5
).fit(all["data"])

X_train = tfidf.transform(train["data"]).astype(np.float32)
X_validation = tfidf.transform(validation["data"]).astype(np.float32)
X_test = tfidf.transform(test["data"]).astype(np.float32)

y_train = train["target"]
y_validation = validation["target"]
y_test = test["target"]

wle = WeakLabelEncoder(random_state=seed).fit(train["weak_targets"])
y_noisy_train = wle.transform(train["weak_targets"])
y_noisy_validation = wle.transform(validation["weak_targets"])
y_noisy_test = wle.transform(test["weak_targets"])

classifier = SGDClassifier(loss="log_loss", random_state=seed)

vosg = LinearVoSG(classifier)
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    trust_scores = vosg.trust_score(X_train, y_noisy_train)

n_runs = 5
n_splits = 200
splits = np.linspace(0, 1.0, endpoint=False, num=n_splits)

val_scores = np.empty((n_runs, n_splits))
noisy_val_scores = np.empty((n_runs, n_splits))
test_scores = np.empty((n_runs, n_splits))

gold_scores = np.empty(n_runs)
silver_scores = np.empty(n_runs)
none_scores = np.empty(n_runs)

for i in range(n_runs):
    classifier.set_params(random_state=seed + i)

    gold_scores[i] = balanced_accuracy_score(
        y_test, classifier.fit(X_train, y_train).predict(X_test)
    )

    clean = y_train == y_noisy_train
    silver_scores[i] = balanced_accuracy_score(
        y_test, classifier.fit(X_train[clean, :], y_noisy_train[clean]).predict(X_test)
    )

    none_scores[i] = balanced_accuracy_score(
        y_test, classifier.fit(X_train, y_noisy_train).predict(X_test)
    )

    for j, split in enumerate(splits):
        classifier.set_params(random_state=seed + i * n_splits + j)
        filtered = trust_scores >= np.quantile(trust_scores, split)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            classifier.fit(X_train[filtered], y_noisy_train[filtered])
        val_scores[i, j] = balanced_accuracy_score(
            y_validation, classifier.predict(X_validation)
        )
        noisy_val_scores[i, j] = balanced_accuracy_score(
            y_noisy_validation, classifier.predict(X_validation)
        )
        test_scores[i, j] = balanced_accuracy_score(y_test, classifier.predict(X_test))


plt.axhline(np.mean(gold_scores), color="gold", label="gold")
plt.axhline(np.mean(silver_scores), color="silver", label="silver")
plt.axhline(np.mean(none_scores), color="red", label="none")
for scores, label, color in zip(
    (val_scores, noisy_val_scores, test_scores),
    ("clean validation", "noisy validation", "test"),
    ("green", "purple", "blue"),
):
    plt.plot(splits, np.mean(scores, axis=0), label=label, color=f"tab:{color}")
    plt.fill_between(
        splits,
        np.mean(scores, axis=0) + np.std(scores, axis=0),
        np.mean(scores, axis=0) - np.std(scores, axis=0),
        alpha=0.5,
        color=f"tab:{color}",
    )
plt.axvline(
    splits[np.argmax(np.mean(val_scores, axis=0))],
    linestyle="--",
    color="black",
    label="clean threshold",
)
plt.axvline(
    splits[np.argmax(np.mean(noisy_val_scores, axis=0))],
    linestyle=":",
    color="black",
    label="noisy threshold",
)
plt.xlabel("threshold")
plt.ylabel("balanced accuracy")
plt.title("Threshold selection on SMS dataset")
plt.legend()
plt.show()

# %%
