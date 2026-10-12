"""
train_motion_model.py

Trains the motion-only RandomForestClassifier ({J, Z, NONE}) from
motion_only_data.csv (produced by collect_motion_data.py) and saves it as
motion_only_model.joblib.

Run after collecting at least ~60-100 examples per class.
"""

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from hand_features import MOTION_FEATURE_NAME_ORDER

MOTION_CSV = "motion_only_data.csv"


def main():
    df = pd.read_csv(MOTION_CSV)
    print(f"Motion data: {len(df)} rows, classes: "
          f"{sorted(df['label'].unique())}")

    X = df[MOTION_FEATURE_NAME_ORDER].values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=200,
        min_samples_leaf=2,
        class_weight="balanced",  # guards against uneven J/Z/NONE counts
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    print("\n=== Motion (J/Z/NONE) model ===")
    print(classification_report(y_test, preds))

    joblib.dump(clf, "motion_only_model.joblib")
    print("Saved motion_only_model.joblib")


if __name__ == "__main__":
    main()