# %% CARREGANDO MÓDULOS

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import numpy as np
import itertools
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import classification_report


# %% CARREGANDO FUNÇÕES

def get_project_root() -> Path:
    """
    Retorna a raiz do projeto (pasta que contém o arquivo .projroot).
    Funciona independetemente de onde o script for executado.
    """
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent/'.projroot').exists():
            return parent       
    raise RuntimeError("Arquivo .projroot não encontrado."
                       "Verifique se ele está na raiz do projeto.")

def load_flags_dataframe(df):
    """Retorna somente as flags e o target."""
    cols = ['cvv_result', 'avs_match', 'three_ds_flag', 'promo_used', 'is_fraud']
    return df[cols].copy()

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

def confusion_at(th):
    pred = (y_proba >= th).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    return {
        'threshold': th,
        'TP': tp, 'FP': fp,
        'FN': fn, 'TN': tn,
        'precision': tp / (tp + fp + 1e-9),
        'recall': tp / (tp + fn + 1e-9)
    }

# %% CARREGANDO DADOS

print('1. carregamento')
root = get_project_root()
csv_path = root/'data'/'raw'/'transactions.csv'
df = pd.read_csv(csv_path)
df_flags = load_flags_dataframe(df)
print(df_flags)
print()

# Checagem Inicial
print("2. dtypes e missing")
print(df_flags.dtypes)
print(df_flags.isnull().sum())
print()

print('3. checar valores únicos por flag (rápido)')
flags = ['cvv_result', 'avs_match', 'three_ds_flag', 'promo_used', 'is_fraud']
for c in flags:
    print(df_flags[c].value_counts(dropna=False).sort_index())
print()

print("4. garantir que são 0/1 inteiros (forçar conversão segura)")
print("   se existirem outros códigos, vamos listar antes de converter")
non_binary = {c: sorted([v for v in df_flags[c].unique() if v not in (0,1)]) for c in flags}
print("=== valores não-binários (esperamos listas vazias) ===")
print(non_binary)

# Se as listas acima estiverem vazias, podemos converter com segurança:
df_flags = df_flags.fillna(-1)  # só por segurança; falaremos sobre missings se aparecerem
for c in flags:
    df_flags[c] = df_flags[c].astype(int)
print()

print('"5. garantir que são 0/1 inteiros (forçar conversão segura)"')
print("=== balanceamento do target (is_fraud) ===")
print(df_flags['is_fraud'].value_counts(dropna=False))
print(df_flags['is_fraud'].value_counts(normalize=True))
print()

print('6. split estratificado (apenas criar e mostrar shapes')
X = df_flags.drop('is_fraud', axis=1)
y = df_flags['is_fraud']

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
print(f"y_train (sum frauds): {y_train.sum()} de {y_train.shape[0]} - pct: {y_train.sum()/y_train.shape[0]}")
print(f"y_test  (sum frauds): {y_test.sum()} de {y_test.shape[0]} - pct: {y_test.sum()/y_test.shape[0]}")

# %% MODELAGEM ESTTÍSTICA

# Adiciona intercepto
X = sm.add_constant(X, has_constant='add')

# Ajustando o modelo Logit
model_sm = sm.Logit(y, X)
result = model_sm.fit(disp=False) # disp=False para não poluir a saída

# Mostara sumário estatísticos
print(f'\nSumário Estatístico\n{result.summary()}\n')

# Calcular ratios e intervalos de confiança
coef = result.params
conf = result.conf_int()
conf.columns = ['2.5%', '97.5%']

odds = np.exp(coef)
odds_ci = np.exp(conf)

df_or = pd.DataFrame({
    'coef': coef,
    'odds_ratio': odds,
    'ci_lower': odds_ci['2.5%'],
    'ci_upper': odds_ci['97.5%'],
    'p_value': result.pvalues
})

print("\nOdds ratios e IC 95%:")
print(df_or.sort_values('odds_ratio', ascending=False))

# %% (A) Gera uma tabela com a probabilidade estimada para todas as 16 combinações possíveis das 4 flags (0/1) e ordeno por risco

# Coeficientes do modelo Logit (use exatamente os seus)
coef = {
    'const':       -1.772458,
    'cvv_result':  -0.748623,
    'avs_match':   -1.866811,
    'three_ds_flag': -1.066433,
    'promo_used':   0.969340
}

# Gera as 16 combinações possíveis das 4 flags (0/1)
combos = list(itertools.product([0, 1], repeat=4))

# Montar lista de resultados
linhas = []

for cvv, avs, ds, promo in combos:
    z = (
        coef['const']
        + coef['cvv_result'] * cvv
        + coef['avs_match'] * avs
        + coef['three_ds_flag'] * ds
        + coef['promo_used'] * promo
    )
    p = 1 / (1 + np.exp(-z))   # probabilidade
    
    linhas.append({
        'combo': f"{cvv}{avs}{ds}{promo}",
        'cvv_result': cvv,
        'avs_match': avs,
        'three_ds_flag': ds,
        'promo_used': promo,
        'prob_fraud': p
    })

# DataFrame final
df_combos = pd.DataFrame(linhas).sort_values('prob_fraud', ascending=False)
df_combos[['combo', 'prob_fraud']]

# %% (B) Entrega uma função pronta para rodar no seu ambiente e consultar qualquer combo
# Coeficientes do modelo Logit ajustado
COEF = {
    'const':       -1.772458,
    'cvv_result':  -0.748623,
    'avs_match':   -1.866811,
    'three_ds_flag': -1.066433,
    'promo_used':   0.969340
}

fraud_prob(
    cvv = 0, 
    avs = 0, 
    ds = 0, 
    promo = 1, 
    coef=COEF)

# %% (C) Aplica o modelo no conjunto de teste para mostrar calibragem (predicted probs vs observed), matriz de confusão em vários cutoffs e AUC.

# C1 - Calibração: predicted vs observed
model_lr = LogisticRegression(max_iter=1000)
model_lr.fit(X_train, y_train)

y_proba = model_lr.predict_proba(X_test)[:,1]

prob_true, prob_pred = calibration_curve(y_test, y_proba, n_bins=10)

df_calibration = pd.DataFrame({
    'bin_predicted': prob_pred,
    'bin_observed': prob_true
})

print("\n=== Calibração (predito vs observado) ===")
print(df_calibration)


# C2 - Matrizes de confusão em vários thresholds

thresholds = [0.01, 0.02, 0.05, 0.10, 0.20]
df_thresholds = pd.DataFrame([confusion_at(t) for t in thresholds])

print("\n=== Matrizes em vários thresholds ===")
print(df_thresholds)

# C3 - AUC/ROC
auc = roc_auc_score(y_test, y_proba)
fpr, tpr, _ = roc_curve(y_test, y_proba)

print(f"\nAUC = {auc:.4f}")

# %%
