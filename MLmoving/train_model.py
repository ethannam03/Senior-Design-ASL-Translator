
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
 
from hand_features import STATIC_FEATURE_NAMES, MOTION_FEATURE_NAME_ORDER
 
STATIC_CSV = "static_data.csv"
MOTION_CSV = "motion_data.csv"
 
 
def train_static():
    df = pd.read_csv(STATIC_CSV)
    print(f"Static data: {len(df)} rows, classes: "
          f"{sorted(df['label'].unique())}")
 
    X = df[STATIC_FEATURE_NAMES].values
    y = df["label"].values
 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
 
    clf = RandomForestClassifier(
        n_estimators=150,       
        max_depth=None,         
        min_samples_leaf=2,     
        class_weight="balanced", 
        random_state=42,
        n_jobs=-1,              
    )
    clf.fit(X_train, y_train)
 
    preds = clf.predict(X_test)
    print("\nStatic letter model")
    print(classification_report(y_test, preds))
 
    joblib.dump(clf, "static_model.joblib")
    print("Saved static_model.joblib")
    return clf
 
 
def train_motion():
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
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
 
    preds = clf.predict(X_test)
    print("\nMotion (J/Z/NONE) model")
    print(classification_report(y_test, preds))
 
    joblib.dump(clf, "motion_model.joblib")
    print("Saved motion_model.joblib")
    return clf
 
 
if __name__ == "__main__":
    train_static()
    train_motion()