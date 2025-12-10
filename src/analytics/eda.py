# %% CARREGANDO MÓDULOS

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# %% FUNÇÕES DEFINIDAS
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

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.edgecolor": "#333",
    "axes.linewidth": 0.8,
    "figure.dpi": 120
})

def create_hist_box(col, df, n_bins=100, y_scale='linear'):
    """
        Gera uma figura contendo um histograma e um boxplot horizontal
        para uma variável numérica.

        Parameters
        ----------
        col : str
            Nome da coluna numérica do DataFrame.
        df : pandas.DataFrame
            DataFrame contendo os dados.
        n_bins : int, default=100
            Número de bins do histograma.
        y_scale : {'linear', 'log'}, default='linear'
            Escala do eixo y do histograma.

        Saves
        -----
        ../../img/hist_box_<col>.png
            Arquivo PNG da figura com resolução de 300 dpi.

        Notes
        -----
        O título do gráfico é omitido para uso em relatórios (caption externo).
        O tamanho do bin (Δ) é exibido discretamente no canto superior direito
        do histograma.
    """
    delta_bin = (df[col].max() - df[col].min()) / n_bins

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(7.5, 5.2), # proporção mais de paper
        height_ratios=[4, 1]
    )

    # --- HISTOGRAMA ---
    ax1.hist(
        df[col],
        bins=n_bins,
        edgecolor="#222",
        linewidth=0.5,
        color="#4a90e2"  # azul discreto
    )

    ax1.text(
        0.98, 0.95,
        f"Δ = {delta_bin:.3g}",
        ha="right",
        va="top",
        transform=ax1.transAxes,
        fontsize=9,
        color="#333",
        bbox=dict(
            facecolor="white",
            edgecolor="none",
            alpha=0.6,
            boxstyle="round,pad=0.3"
        )
    )

    ax1.set_yscale(y_scale)
    # ax1.set_title(f"{col} — histograma + boxplot  (Δ={delta_bin:.3g})")
    ax1.set_ylabel("Contagem")

    # ax1.grid(
    #     True,
    #     which="both",
    #     linestyle="--",
    #     linewidth=0.4,
    #     alpha=0.4
    # )

    # Remove xticks do histograma para não poluir
    ax1.tick_params(axis="x", bottom=False, labelbottom=False)

    # --- BOXPLOT ---
    bp = ax2.boxplot(
        df[col],
        vert=False,
        patch_artist=True,
        boxprops=dict(facecolor="#d9e6f5", color="#222", linewidth=0.8),
        medianprops=dict(color="#d0021b", linewidth=1.2),
        whiskerprops=dict(color="#222", linewidth=0.8),
        capprops=dict(color="#222", linewidth=0.8)
    )

    ax2.set_xlabel(col)
    ax2.set_yticks([])

    # Margens mais justas
    plt.tight_layout()

    # plt.savefig(f'../../img/hist_box_{col}.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_flag(col):
    """
    Plota uma barra vertical para flags (0/1) com:
    - valores absolutos no eixo y
    - porcentagens exibidas nas barras (posição adaptativa)
    """

    abs_counts = df[col].value_counts().sort_index()
    rel_counts = df[col].value_counts(normalize=True).sort_index() * 100

    fig, ax = plt.subplots(figsize=(4, 3))

    bars = ax.bar(
        abs_counts.index.astype(str),
        abs_counts.values,
        edgecolor="black"
    )

    ax.set_ylabel("Contagem")
    ax.set_xlabel(col)

    # limiar baseado na barra maior
    limiar = abs_counts.max() * 0.15  # 15% da maior barra

    # adiciona porcentagens (adaptativo)
    for bar, pct, val in zip(bars, rel_counts, abs_counts):
        height = bar.get_height()
        x = bar.get_x() + bar.get_width() / 2

        if height >= limiar:
            # dentro da barra
            ax.text(
                x,
                height * 0.92,
                f"{pct:.1f}%",
                ha="center",
                va="top",
                color="white",
                fontsize=9
            )
        else:
            # fora da barra
            ax.text(
                x,
                height * 1.02,
                f"{pct:.1f}%",
                ha="center",
                va="bottom",
                color="black",
                fontsize=9
            )

    plt.tight_layout()
    plt.show()


def plot_barh(col, df, top_n=None):
    """
    Plota barras horizontais para uma variável categórica com:
    - valores absolutos no eixo x
    - porcentagem exibida em cada barra (posição adaptativa)

    A porcentagem aparece:
    - dentro da barra, alinhada à direita, se a barra for grande
    - fora da barra, à direita, se a barra for pequena
    """

    counts = df[col].value_counts(ascending=True)

    if top_n is not None:
        counts = counts.head(top_n)

    total = counts.sum()
    percentages = counts / total * 100

    fig, ax = plt.subplots(
        figsize=(8, max(3, len(counts) * 0.35))
    )

    ax.barh(
        counts.index.astype(str),
        counts.values,
        edgecolor="black"
    )

    ax.set_xlabel("Contagem")
    ax.set_ylabel(col)

    # threshold para decidir se a porcentagem fica dentro ou fora
    limiar = counts.max() * 0.10  # 10% da barra maior

    # Adiciona o texto das porcentagens
    for i, (value, pct) in enumerate(zip(counts.values, percentages)):
        y = i  # posição da categoria

        if value >= limiar:
            # texto dentro da barra
            ax.text(
                value * 0.98,  # um pouco antes do final da barra
                y,
                f"{pct:.1f}%",
                ha="right",
                va="center",
                color="white",
                fontsize=9,
            )
        else:
            # texto fora da barra
            ax.text(
                value * 1.02,
                y,
                f"{pct:.1f}%",
                ha="left",
                va="center",
                color="black",
                fontsize=9,
            )

    plt.tight_layout()
    plt.show()


# %% CARREGANDO DADOS
root = get_project_root()
csv_path = root/'data'/'raw'/'transactions.csv'
df = pd.read_csv(csv_path)


# %% INFORMAÇÕES PRELIMINARES DO DATASET

print('[DATASET]')
print(f'{df.head()}\n')

print('[INFO DATASET]')
print(f'{df.info()}\n')

# IDs
print('[IDs]')
print(f"[transaction_id] lines: {df[['transaction_id']].shape[0]} type: {df['transaction_id'].dtypes}")
print(f"[user_id] lines: {df[['user_id']].shape[0]} type: {df['user_id'].dtypes}\n")

# Colunas Numéricas
print('[COLUNAS NUMÉRICAS]\n')
num_cols = ['account_age_days', 'total_transactions_user','avg_amount_user','amount','shipping_distance_km']

for col in num_cols:
    print(f'["{col}" describe:]')
    print(f'{df[col].describe()}\n')

# Colunas Categóricas
print('[COLUNAS CATEGÓRICAS]\n')
cat_cols = ['country', 'bin_country', 'channel', 'merchant_category',]
for col in cat_cols:
    print(f'[value counts:]')
    print(f'{df[col].value_counts(normalize=True)}\n')

# Flags
print('[FLAGS]\n')
flag_cols = ['promo_used', 'avs_match', 'cvv_result', 'three_ds_flag']
for col in flag_cols:
    print(f'[value counts:]')
    print(f'{df[col].value_counts(normalize=True)}\n')

# Datetimes
print('[DATETIMES]')
print(f"[transaction_time] lines: {df[['transaction_time']].shape[0]} type: {df['transaction_time'].dtypes}\n")

# Target
print('[TARGET]')
print(f'{df['is_fraud'].value_counts(normalize=True)}\n')

# %% DESCRIPITION OF VARIABLES
data_dict = {
    "column": [
        "transaction_id",
        "user_id",
        "account_age_days",
        "total_transactions_user",
        "avg_amount_user",
        "amount",
        "country",
        "bin_country",
        "channel",
        "merchant_category",
        "promo_used",
        "avs_match",
        "cvv_result",
        "three_ds_flag",
        "transaction_time",
        "shipping_distance_km",
        "is_fraud"
    ],
    
    "type": [
        "int64", "int64", "int64", "int64", "float64", "float64",
        "object", "object", "object", "object",
        "int64", "int64", "int64", "int64", "object",
        "float64", "int64"
    ],
    
    "description": [
        "ID único da transação",
        "ID único do usuário",
        "Idade da conta (dias desde a criação)",
        "Total de transações históricas do usuário",
        "Valor médio histórico das transações do usuário",
        "Valor da transação atual",
        "País do usuário",
        "País do BIN do cartão",
        "Canal da transação (web, app, etc.)",
        "Categoria do estabelecimento",
        "Indicador do uso de cupom promocional",
        "Indicador do AVS (Address Verification Service)",
        "Indicador do CVV (Card Verification Value)",
        "Indicador do uso de 3DSecure",
        "Timestamp da transação (ISO 8601)",
        "Distância estimada entre endereço de entrega e billing",
        "Label da transação (0 legítima, 1 fraude)"
    ]
}
df_dict = pd.DataFrame(data_dict)
print(df_dict)

# %% TARGET PROPORTION
plot_flag('is_fraud')

# %% HISTOGRAMAS/BOXPLOTS DAS VARIÁVEIS NUMÉRICAS
create_hist_box(col='account_age_days', df=df)
create_hist_box(col='total_transactions_user', df=df, n_bins=20)
create_hist_box(col='avg_amount_user', df=df, y_scale='log')
create_hist_box(col='amount', df=df, y_scale='log')
create_hist_box(col='shipping_distance_km', df=df, y_scale='log')

# %% BARRAS HORIZONTAIS DAS VARIÁVEIS CATEGÓRICAS
plot_barh(col='country', df=df)
plot_barh(col='bin_country', df=df)
plot_barh(col='channel', df=df)
plot_barh(col='merchant_category', df=df)

# %% BARRAS VERTICATS DAS FLAGS
plot_flag('promo_used')
plot_flag('avs_match')
plot_flag('cvv_result')
plot_flag('three_ds_flag')

# %% TRANSACTION TIME

# Cópia de segurança
df_tt = df[['transaction_id', 'user_id', 'transaction_time', 'is_fraud']].copy()

# Reatribuições
df_tt['new_transaction_time'] = pd.to_datetime(df_tt['transaction_time'], utc=True)
df_tt['date'] = df_tt['new_transaction_time'].dt.strftime('%Y-%m-%d')
df_tt['weekday'] = df_tt['new_transaction_time'].dt.day_name()
df_tt['month'] = df_tt['new_transaction_time'].dt.month
df_tt['day'] = df_tt['new_transaction_time'].dt.strftime('%d')
df_tt['hour'] = df_tt['new_transaction_time'].dt.hour

df_tt = df_tt[[
    'transaction_id', 'user_id',
    'transaction_time', 'new_transaction_time',
    'date', 'weekday', 'month', 'day', 'hour',
    'is_fraud']]

print(df_tt)

# Separa legítimas e fraudes
df_tt_0 = df_tt[df_tt['is_fraud'] == 0].copy()
df_tt_1 = df_tt[df_tt['is_fraud'] == 1].copy()

# Contagem Mensal
df_monthly     = df_tt.groupby('month').size()
df_monthly_0   = df_tt_0.groupby('month').size()
df_monthly_1   = df_tt_1.groupby('month').size()
df_monthly_summary = pd.DataFrame({
    'transactions': df_monthly,
    'legit': df_monthly_0,
    'fraud': df_monthly_1,
    'pct_legit': (df_monthly_0 / df_monthly),
    'pct_fraud': (df_monthly_1 / df_monthly)
}).reset_index()
print('[Contagem Mensal]')
print(f'{df_monthly_summary}\n')

# Contagem Diária
df_daily = df_tt.groupby('date').size()
df_daily_0 = df_tt_0.groupby('date').size()
df_daily_1 = df_tt_1.groupby('date').size()
df_daily_summary = pd.DataFrame({
    'transactions': df_daily,
    'legit': df_daily_0,
    'fraud': df_daily_1,
    'pct_legit': (df_daily_0 / df_daily),
    'pct_fraud': (df_daily_1 / df_daily)
}).reset_index()
print('[Contagem Diária]')
print(f'{df_daily_summary}\n')

# Contagens por Hora
df_hourly = df_tt.groupby('hour').size()
df_hourly_0 = df_tt_0.groupby('hour').size()
df_hourly_1 = df_tt_1.groupby('hour').size()
df_hourly_summary = pd.DataFrame({
    'tranactions': df_hourly,
    'legit': df_hourly_0,
    'fraud': df_hourly_1,
    'pct_legit': df_hourly_0 / df_hourly,
    'pct_fraud': df_hourly_1 / df_hourly
}).reset_index()
print('[Contagens por Hora]')
print(f'{df_hourly_summary}\n')

# Contagens por Dia da Semana
df_weekday = df_tt.groupby('weekday').size()
df_weekday_0 = df_tt_0.groupby('weekday').size()
df_weekday_1 = df_tt_1.groupby('weekday').size()
df_weekday_summary = pd.DataFrame({
    'tranactions': df_weekday,
    'legit': df_weekday_0,
    'fraud': df_weekday_1,
    'pct_legit': df_weekday_0 / df_weekday,
    'pct_fraud': df_weekday_1 / df_weekday
}).reset_index()
print('[Contagens por Dia da Semana]')
print(f'{df_weekday_summary}\n')

# Contagem por semana sequencial
# criar índice semanal sequencial
df_tt_widx = df_tt.sort_values('new_transaction_time').copy()
df_tt_widx['week_index'] = ((df_tt_widx['new_transaction_time'] - df_tt_widx['new_transaction_time'].iloc[0]).dt.days // 7)
# número total de semanas no dataset
total_weeks = df_tt_widx['week_index'].max() + 1  # geralmente 44 no seu caso
# manter apenas as 43 semanas completas
df_tt_weeks = df_tt_widx[df_tt_widx['week_index'] < 43].copy()
# criar resumo semanal
df_weekly_summary = df_tt_weeks.groupby('week_index').agg(
    transactions = ('is_fraud','size'),
    legit = ('is_fraud', lambda x: (x == 0).sum()),
    fraud = ('is_fraud', lambda x: (x == 1).sum())
).reset_index()
# adicionar taxas
df_weekly_summary['pct_legit'] = df_weekly_summary['legit'] / df_weekly_summary['transactions']
df_weekly_summary['pct_fraud'] = df_weekly_summary['fraud'] / df_weekly_summary['transactions']
print('[Contagem por Semana Sequencial]')
print(f'{df_weekly_summary}\n')


# %% CORRELAÇÕES NUMÉRICAS
cols = [
    "account_age_days",
    "total_transactions_user",
    "avg_amount_user",
    "amount",
    "shipping_distance_km",
    "is_fraud"
]
df_num = df[cols].copy()

# Colunas Logarítimicas
df_num['log_amount'] = np.log1p(df_num['amount'])
df_num['log_avg_amount_user'] = np.log1p(df_num['avg_amount_user'])
df_num['log_shipping_distance_km'] = np.log1p(df_num['shipping_distance_km'])

df_num = df_num[[
    "account_age_days",
    "total_transactions_user",
    "avg_amount_user", 
    "amount",
    "shipping_distance_km",
    "log_avg_amount_user",
    "log_amount",
    "log_shipping_distance_km",
    "is_fraud"
]]

# Correlações
corr = df_num.drop(columns=["log_avg_amount_user", "log_amount", "log_shipping_distance_km"]).corr()
corr_log = df_num.drop(columns=["avg_amount_user", "amount", "shipping_distance_km"]).corr()
corr_comp = df_num.corr()

# %% Corr
plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, fmt=".3f", cmap="viridis", linewidths=0.5)
plt.title("Correlação entre Variáveis Numéricas")
plt.show()

# %% Corr Log
plt.figure(figsize=(8,6))
sns.heatmap(corr_log, annot=True, fmt=".3f", cmap="viridis", linewidths=0.5)
plt.title("Correlação entre Variáveis Numéricas Logarítimicas")
plt.show()

# %% Corr Compose
plt.figure(figsize=(8,6))
sns.heatmap(corr_comp, annot=True, fmt=".3f", cmap="viridis", linewidths=0.5)
plt.title("Correlação entre Variáveis Numéricas Compostas")
plt.show()

# %% Coments
# df_cvv_result_0 = df[df['cvv_result'] == 0]
# print('cvv_result NEGATIVO')
# print(df_cvv_result_0['is_fraud'].value_counts(normalize=True))

# df_three_ds_flag_0 = df[df['three_ds_flag'] == 0]
# print('three_ds_flag NEGATIVO')
# print(df_three_ds_flag_0['is_fraud'].value_counts(normalize=True))

# df_avs_match_0 = df[df['avs_match'] == 0]
# print('avs_match NEGATIVO')
# print(df_avs_match_0['is_fraud'].value_counts(normalize=True))

# df_promo_used_0 = df[df['promo_used'] == 0]
# print('promo_used NEGATIVO')
# print(df_promo_used_0['is_fraud'].value_counts(normalize=True))

# df_cvv_result_1 = df[df['cvv_result'] == 1]
# print('cvv_result POSITIVOS')
# print(df_cvv_result_1['is_fraud'].value_counts(normalize=True))

# df_three_ds_flag_1 = df[df['three_ds_flag'] == 1]
# print('three_ds_flag POSITIVOS')
# print(df_three_ds_flag_1['is_fraud'].value_counts(normalize=True))

# df_avs_match_1 = df[df['avs_match'] == 1]
# print('avs_match POSITIVOS')
# print(df_avs_match_1['is_fraud'].value_counts(normalize=True))

# df_promo_used_1 = df[df['promo_used'] == 1]
# print('promo_used POSITIVOS')
# print(df_promo_used_1['is_fraud'].value_counts(normalize=True))

# %% ANÁLISE DAS FLAGS
flags = ['cvv_result', 'three_ds_flag', 'avs_match', 'promo_used']

for flag in flags:
    print(f'\n=== {flag.upper()} ===')
    
    for value, label in [(0, 'NEGATIVO'), (1, 'POSITIVO')]:
        subset = df[df[flag] == value]['is_fraud']
        proporcao = subset.value_counts(normalize=True)
        
        print(f'\n{label}:')
        print(proporcao)
print()
rows = []
for flag in flags:
    for value in [0, 1]:
        subset = df[df[flag] == value]['is_fraud']
        proporcao = subset.value_counts(normalize=True)
        fraud_0 = proporcao.get(0, 0)
        fraud_1 = proporcao.get(1, 0)
        rows.append([flag, value, fraud_0, fraud_1])

df_flags_summary = pd.DataFrame(rows, columns=['flag', 'value', 'prop_legit', 'prop_fraud'])
df_flags_summary

# criar combinações
df_flags = df[['cvv_result','avs_match','three_ds_flag','promo_used','is_fraud']].copy()

df_flags['combo'] = (
    df_flags['cvv_result'].astype(str) +
    df_flags['avs_match'].astype(str) +
    df_flags['three_ds_flag'].astype(str) +
    df_flags['promo_used'].astype(str)
)

S_combo_prob = df_flags.groupby('combo')['is_fraud'].mean().sort_values(ascending=False)

df_combo_prob = pd.DataFrame(S_combo_prob).reset_index()

print('[PROPORÇÕES OBSERVADAS]')
print(df_combo_prob)

