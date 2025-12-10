# E-Commerce Fraud Detection

Projeto voltado à detecção de fraudes em transações de comércio eletrônico.  
O repositório contém o pipeline completo de ingestão, análise exploratória, preparação de dados, engenharia de atributos e modelagem supervisionada e não supervisionada para identificação de padrões fraudulentos.

## 1. Visão Geral

Este projeto simula o fluxo usado por equipes antifraude em ambientes reais, cobrindo:

- identificação de padrões comportamentais de usuários e transações;
- criação de variáveis derivadas (velocity features, device patterns, geolocalização, etc.);
- tratamento de forte desbalanceamento entre classes;
- experimentação com modelos supervisionados e algoritmos de detecção de anomalias;
- avaliação com métricas apropriadas para fraude (AUC, Recall, Precision@k, etc.).

A documentação completa e a análise científica detalhada estarão disponíveis no diretório **/docs** (relatório em LaTeX).

---

## 2. Estrutura do Repositório
.
├── data/ # Dados brutos e processados (não versionados no Git)
├── src/ # Código-fonte do pipeline
│ ├── eda/ # Scripts de análise exploratória
│ ├── features/ # Feature engineering
│ ├── models/ # Treinamento e avaliação
│ └── utils/ # Funções auxiliares
├── notebooks/ # Notebooks de experimentação
├── docs/ # Relatório, diagramas e documentação adicional
├── img/ # Figuras usadas no README e/ou relatório
└── README.md

---

## 3. Como Executar

### Pré-requisitos
- Python 3.x  
- pip ou conda  
- requirements list (em `requirements.txt`)

### Instalação
```sh
pip install -r requirements.txt
```

### Execução do pipeline
```
python src/run_pipeline.py
```
### Uso via notebooks
Os notebooks em notebooks/ podem ser executados no Jupyter ou Google Colab.

## 4. Dataset

O conjunto de dados contém transações legítimas e fraudulentas.
A descrição completa, incluindo dicionário de variáveis, será disponibilizada em:

docs/data_dictionary.md

### Principais colunas incluem:
- dados da conta e histórico do usuário;
- atributos da transação;
- categorias comerciais;
- informações de risco (CVV, AVS, 3DS);
- rótulo binário indicando fraude.

## 5. Roadmap do Projeto

- [ ] Validação inicial do dataset
- [ ] Data Dictionary
- [ ] EDA com foco em padrões de fraude
- [ ] Criação das features
- [ ] Preparação para modelagem
- [ ] Modelos supervisionados
- [ ] Modelos de anomalia
- [ ] Avaliação e comparação
- [ ] Geração de relatório científico
- [ ] Documentação final

# Data Dictionary — E-Commerce Fraud Detection

Este documento descreve o significado de cada variável do dataset utilizado no projeto.  
Os tipos referem-se ao formato original de carregamento em `pandas`.

> Observação: campos derivados (engineered features) serão incluídos após a etapa de feature engineering.

---

## 1. Variáveis Originais

| Coluna                     | Tipo      | Descrição                                                                 |
|---------------------------|-----------|---------------------------------------------------------------------------|
| transaction_id            | int64     | Identificador único da transação.                                         |
| user_id                   | int64     | Identificador único do usuário.                                           |
| account_age_days          | int64     | Idade da conta em dias.                                                   |
| total_transactions_user   | int64     | Total de transações históricas do usuário.                                |
| avg_amount_user           | float64   | Valor médio gasto pelo usuário em transações passadas.                    |
| amount                    | float64   | Valor da transação atual.                                                 |
| country                   | object    | País onde a transação foi registrada.                                     |
| bin_country               | object    | País associado ao BIN do cartão.                                          |
| channel                   | object    | Canal de origem da transação (web, mobile, api...).                       |
| merchant_category         | object    | Categoria do estabelecimento/comércio.                                    |
| promo_used                | int64     | 1 se um cupom/promo foi usado; 0 caso contrário.                          |
| avs_match                 | int64     | Resultado do Address Verification System.                                 |
| cvv_result                | int64     | Resultado do CVV no processamento.                                        |
| three_ds_flag             | int64     | 1 se teve autenticação 3D-Secure; 0 caso contrário.                        |
| transaction_time          | object    | Timestamp da transação.                                                   |
| shipping_distance_km      | float64   | Distância estimada entre o endereço do cliente e o endereço de entrega.   |
| is_fraud                  | int64     | 1 = fraude, 0 = legítima.                                                 |

---

## 2. Variáveis Derivadas (a serem adicionadas)

Estas serão criadas durante o pipeline:

- velocity features (n transações nos últimos X min)
- desvio de comportamento por usuário
- razões monetárias (amount / avg_amount_user)
- discrepância geográfica (country vs bin_country)
- sinalizações de risco (heurísticas)
- embeddings categóricos, se usados

Um novo documento será gerado quando o conjunto final estiver definido.
