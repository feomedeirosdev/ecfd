from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


def load_flags_dataframe(df):
    """Retorna somente as flags e o target."""
    cols = ['cvv_result', 'avs_match', 'three_ds_flag', 'promo_used', 'is_fraud']
    return df[cols].copy()

def xy_train_test_split(df, target):
    X = df.drop(target, axis=1)
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y, 
        test_size=0.3, 
        random_state=42, 
        stratify=y
    )

    print("Shapes:")
    print("X_train:", X_train.shape)
    print("X_test :", X_test.shape)
    print(f"y_train (sum): {y_train.sum()} de {y_train.shape[0]} - pct: {y_train.sum()/y_train.shape[0]}")
    print(f"y_test  (sum): {y_test.sum()} de {y_test.shape[0]} - pct: {y_test.sum()/y_test.shape[0]}")
    
    return X_train, X_test, y_train, y_test

# 1. Regressão Logística simples (sem interações)
def logistic_simple(df_flags):
    X = df_flags.drop('is_fraud', axis=1)
    y = df_flags['is_fraud']

    X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y, 
        test_size=0.3, 
        random_state=42, 
        stratify=y
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    return model, classification_report(y_test, preds, output_dict=True)

def fraud_prob(cvv, avs, ds, promo, coef):
    """
    Retorna a probabilidade estimada de fraude para qualquer combinação das 4 flags.
    
    Parâmetros:
        cvv   : 0 ou 1
        avs   : 0 ou 1
        ds    : 0 ou 1 (three_ds_flag)
        promo : 0 ou 1 (promo_used)

    Exemplo:
        fraud_prob(0, 0, 0, 1)
    """
    z = (
        coef['const']
        + coef['cvv_result']  * cvv
        + coef['avs_match']   * avs
        + coef['three_ds_flag'] * ds
        + coef['promo_used']  * promo
    )
    
    p = 1 / (1 + np.exp(-z))
    return float(p)

def confusion_at(th, y_proba, y_test):
    pred = (y_proba >= th).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    return {
        'threshold': th,
        'TP': tp, 'FP': fp,
        'FN': fn, 'TN': tn,
        'precision': tp / (tp + fp + 1e-9),
        'recall': tp / (tp + fn + 1e-9)
    }