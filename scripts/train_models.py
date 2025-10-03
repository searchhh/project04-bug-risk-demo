import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import joblib

DATA_PATH = Path(".\data\data.csv")
assert DATA_PATH.exists(), "NO DATA, PLEASE GENERATE DATA FIRSTG"

# 1, 读数据
df = pd.read_csv(DATA_PATH)
feature_names = ["churn", "dev_count", "sync_keywords"]
X = df[feature_names].values
y = df["is_buggy"].values

# 2, 训练测试划分
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 3，训练Random Forest
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_leaf=2,
    n_jobs=-1,
    random_state=42,
    class_weight="balanced"
)
rf.fit(X_train, y_train)

# 4, 训练XGBoost
neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
scale_pos_weight = neg / max(pos, 1)

xgb = XGBClassifier(
    n_estimators=500,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_lambda=1.0,
    objective="binary:logistic",
    eval_metric="logloss",
    n_jobs=-1,
    random_state=42,
    scale_pos_weight=scale_pos_weight
)
xgb.fit(X_train, y_train)


def eval_model(name, model):
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)  # 如需提高召回可以调小阈值
    print(f"\n==== {name} ====")
    print("Accuracy :", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred, zero_division=0))
    print("Recall   :", recall_score(y_test, y_pred))
    print("F1       :", f1_score(y_test, y_pred))
    try:
        print("ROC AUC  :", roc_auc_score(y_test, y_prob))
    except ValueError:
        print("ROC AUC  : (需要测试集中同时包含正负样本)")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n",
          classification_report(y_test, y_pred, digits=4))


# 5, 评估
eval_model("RandomForest", rf)
eval_model("XGBoost", xgb)

# 6， 保存模型与特征
joblib.dump({"model": rf, "features": feature_names}, "rf_model.joblib")
joblib.dump({"model": xgb, "features": feature_names}, "xgb_model.joblib")
print("\n模型已保存：rf_model.joblib, xgb_model.joblib")
