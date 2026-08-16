# QuantB3 — Sistema de Geração de Sinais e Gestão de Risco para Ações da B3

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB)](https://www.python.org/) [![Streamlit](https://img.shields.io/badge/Streamlit-dashboard-FF4B4B)](https://streamlit.io/) [![LightGBM](https://img.shields.io/badge/LightGBM-model-02569B)](https://lightgbm.readthedocs.io/) [![yfinance](https://img.shields.io/badge/yfinance-data-6001D2)](https://github.com/ranaroussi/yfinance) [![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## Visão geral

O **QuantB3** é um projeto end-to-end de Data Science para o mercado brasileiro. Ele combina:

- classificação da direção do retorno de cada ação nos próximos cinco pregões;
- controle quantitativo de risco com VaR, CVaR, volatilidade, beta e correlação;
- backtest fora da amostra e dashboard Streamlit para exploração interativa.

O público natural são gestoras, mesas proprietárias e fintechs. O diferencial técnico é a preocupação explícita com dados financeiros: pipeline reprodutível, ajuste de splits/dividendos, validação temporal, defasagem da posição no backtest e métricas de risco não paramétricas.

> ⚠️ **Disclaimer:** este repositório é educacional e de portfólio. Não constitui recomendação de investimento, oferta de valores mobiliários ou aconselhamento financeiro.

## Arquitetura

```text
yfinance API → OHLCV Raw Data → Feature Engineering (10+ features)
     ↓
TimeSeriesSplit CV → LightGBM (Optuna tuning) → Pipeline salvo (.pkl)
     ↓                                                    ↓
Backtest Estratégia                            Streamlit Multi-página
(Equity Curve, Sharpe)                    (Mercado | Sinais | Risco | Predição)
     ↓
Modelo de Risco (VaR/CVaR + Backtesting Kupiec)
```

## Estrutura

```text
quantb3/
├── app.py                         # entrada Streamlit
├── config.py                      # universo, período e parâmetros
├── data.py                        # yfinance e features sem leakage
├── modeling.py                    # CV temporal, métricas e backtest
├── risk.py                        # VaR, CVaR e teste de Kupiec
├── eda.py                         # quatro visualizações analíticas
├── train.py                       # treino, Optuna e persistência
├── pages/
│   ├── 1_📊_Mercado.py
│   ├── 2_🤖_Sinais.py
│   ├── 3_⚠️_Risco.py
│   ├── 4_🔍_Predicao.py
│   └── common.py
├── model/                         # gerado pelo treino: .pkl e feature_names.json
└── artifacts/                     # gerado: métricas e backtest
requirements.txt
README.md
```

## Dados e features

Os preços ajustados são coletados gratuitamente de `yfinance`, para PETR4, VALE3, ITUB4, BBDC4, WEGE3, RENT3, ABEV3, MGLU3, B3SA3, TOTS3, ELET3, PRIO3, RADL3, HAPV3 e CSAN3, além do `^BVSP`, entre 2015 e 2024. `auto_adjust=True` trata splits e dividendos. A base inclui RSI(14), MACD (linha/sinal/histograma), BB %B, ATR, EMAs 9/21/50 e cruzamentos, retornos de 5/10/21 dias, volatilidades 10/21 dias, volume relativo, retorno do Ibovespa, força relativa e beta rolling de 60 dias.

O target vale 1 se `Close[t+5] / Close[t] - 1 > 0.5%`. O `shift(-5)` é usado somente para rotular o passado e as últimas cinco observações são removidas. Portanto, nenhuma feature contém informação futura.

## Como executar

```bash
git clone <seu-repositorio>
cd Projeto-Data-Science-B3
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
python -m quantb3.train
streamlit run quantb3/app.py
```

O treinamento baixa os dados, realiza os **50 trials** do Optuna, salva `quantb3/model/pipeline_lgbm.pkl`, `feature_names.json`, `quantb3/artifacts/metrics.json` e `backtest.csv`. O dashboard mostra uma mensagem orientativa antes de esses artefatos existirem.

## Avaliação e resultados

O split é estritamente temporal: treino (2015–2021), validação walk-forward (2022) e teste final out-of-sample (2023–2024). Os valores reais da execução ficam em `quantb3/artifacts/metrics.json`: **ROC-AUC, PR-AUC, F1, retorno acumulado, Sharpe, Calmar, maximum drawdown e threshold**. Essa escolha evita publicar números fabricados — resultados variam com a revisão histórica do provedor e a data de execução.

Compare a estratégia com o Ibovespa no `backtest.csv`: a estratégia entra comprada quando a probabilidade supera o threshold selecionado na validação e aplica a posição somente no pregão seguinte. O painel de risco também permite comparar o número de violações de VaR com a cobertura esperada pelo teste de Kupiec.

## Decisões técnicas

- **Split temporal, não K-Fold aleatório:** um split aleatório permite que padrões do futuro apareçam no treino, inflando métricas. `TimeSeriesSplit` preserva a ordem de chegada da informação.
- **LightGBM:** funciona bem em relações não lineares e interações entre indicadores tabulares, mantendo inferência rápida. Regressão logística e Random Forest são referências de baseline.
- **Desbalanceamento:** o modelo utiliza `class_weight='balanced'`. SMOTE não é usado porque pode criar exemplos sintéticos temporalmente indevidos se não for aplicado dentro de cada janela de treino.
- **VaR histórico:** retornos financeiros apresentam assimetria e caudas pesadas. O quantil empírico e o CVaR não impõem a hipótese gaussiana do VaR paramétrico.
- **Interpretabilidade:** a página de previsão gera waterfall SHAP e o pipeline pode ser estendido para beeswarm de todo o conjunto de teste; isso revela direção e magnitude das features, não apenas uma métrica agregada.

## Limitações e próximos passos

O backtest é simplificado: não incorpora corretagem, emolumentos, spread, slippage, imposto ou limites de liquidez; todos reduzem retornos em produção. Também há risco de overfitting, revisões na fonte pública e **concept drift** (mudança de regimes de mercado). Próximas etapas: custos e execução realistas, walk-forward rolling com reentreino, MLflow, monitoramento de drift/PSI, dados de notícias via NLP, validação por setor e orquestração de MLOps.

Deploy: [Streamlit Community Cloud — configurar URL](https://projeto-data-science-b3.streamlit.app/)

## Perguntas prováveis de entrevista

1. **Como evitar look-ahead bias?** Features são rolling até `t`; `shift(-5)` só produz o rótulo. As últimas cinco linhas são descartadas e a posição do backtest é defasada um dia.
2. **Por que TimeSeriesSplit, não K-Fold?** K-Fold embaralha cronologia e permite treino em datas posteriores à validação; TimeSeriesSplit respeita o fluxo de informação disponível.
3. **Como interpretar Sharpe?** É retorno anualizado excedente por unidade de volatilidade anualizada. Maior é melhor, mas não substitui análise de drawdown e assimetria.
4. **Por que VaR histórico em vez de gaussiano?** O método histórico usa a distribuição observada e absorve caudas pesadas e assimetrias presentes nos retornos.
5. **Como SHAP valida o modelo?** Ele confere se as decisões decorrem de relações economicamente plausíveis e detecta dependência excessiva de uma feature, mesmo que ROC-AUC pareça boa.
6. **O que é concept drift?** É a mudança da relação entre features e retorno por novos regimes. Pode ser detectada por queda de métricas, PSI e mudanças na distribuição; trata-se com reentreino/alertas rolling.
7. **Como colocar em produção?** Versionar o(s) dados/modelo, agendando ingestão e reentreinando, registrando experimentos, monitoraria qualidade/drift/performance e teria rollback e aprovação humana.
8. **Quais limitações do backtest?** Ausência de custos, slippage, liquidez, impacto de mercado, atrasos de execução, vieses de seleção e possível sobreajuste de hiperparâmetros.
