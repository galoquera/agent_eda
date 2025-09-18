# Agente EDA — LangChain + Gemini (+ LangSmith) — Streamlit

App de **Análise Exploratória de Dados (EDA)** com interface web em **Streamlit**, usando **LangChain** (ferramentas + memória), **Gemini** (LLM da Google) e suporte a **LangSmith** (tracing/observabilidade).  
Permite **upload de CSV**, perguntas em linguagem natural e gráficos como **histograma, correlação, dispersões, crosstab, outliers** e **tendências temporais** (reamostragem por H/D/W/M com média móvel opcional).

---

## ✨ Funcionalidades
- Upload dinâmico de **CSV**
- Agente com ferramentas:
  - `listar_colunas`, `descricao_geral_dados`, `estatisticas_descritivas`
  - `plotar_histograma`, `plotar_mapa_correlacao`, `plotar_dispersao`, `matriz_dispersao`
  - `tabela_cruzada`
  - Outliers: `detectar_outliers_iqr`, `detectar_outliers_zscore`
  - Tempo: `converter_time_para_datetime`
  - **Novo**: `tendencias_temporais` (H/D/W/M, `sum/mean/...`, `rolling`)
- Memória de sessão com `RunnableWithMessageHistory`
- Integração opcional com **LangSmith** (tracing)

---

## 🧩 Requisitos

- **Python 3.11** (recomendado)
- Instalar dependências:
  ```bash
  pip install -r requirements.txt
