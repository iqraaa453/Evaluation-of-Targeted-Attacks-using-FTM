from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

def preprocess(df):
    # =========================
    # 🧹 CLEAN DATA
    # =========================
    df = df.replace("?", np.nan)
    df = df.dropna()

    # =========================
    # 🎯 TARGET ENCODING
    # =========================
    df["income"] = df["income"].apply(
        lambda x: 1 if str(x).strip() == ">50K" else 0
    )

    # =========================
    # 🔀 SPLIT FEATURES / LABEL
    # =========================
    X = df.drop("income", axis=1)
    y = df["income"]

    # =========================
    # 🧠 ONE-HOT ENCODING
    # =========================
    X = pd.get_dummies(X)

    # =========================
    # ⚖️ FEATURE SCALING
    # =========================
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y