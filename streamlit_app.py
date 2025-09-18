# streamlit_app.py
# -*- coding: utf-8 -*-
"""
App Streamlit (somente UI) — Agente EDA com LangChain + Gemini (+ LangSmith)
- Upload dinâmico de CSV
- Ferramentas de EDA + tendências temporais (H/D/W/M + rolling)
- Memória conversacional (RunnableWithMessageHistory)
"""

import os
import io
import tempfile
from typing import Optional, List

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.tools import StructuredTool
from pydantic import BaseModel, Field

# (Opcional) LangSmith — habilita tracing se LANGSMITH_API_KEY estiver nas Secrets
def _enable_langsmith(project: str = "EDA-Agent"):
    ls_key = os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")
    if ls_key and not os.getenv("LANGCHAIN_API_KEY"):
        os.environ["LANGCHAIN_API_KEY"] = ls_key
    if not os.getenv("LANGCHAIN_ENDPOINT") and (ls_key or os.getenv("LANGCHAIN_API_KEY")):
        os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    if os.getenv("LANGCHAIN_TRACING_V2", "").lower() not in ("1", "true", "yes"):
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
    if project and not os.getenv("LANGCHAIN_PROJECT"):
        os.environ["LANGCHAIN_PROJECT"] = project

load_dotenv()
_enable_langsmith()

# -------------------------------------------------------------
# Agente
# -------------------------------------------------------------
class AgenteDeAnalise:
    def __init__(self, caminho_arquivo_csv: str, session_id: str = "ui_streamlit"):
        # --- API Key LLM ---
        google_api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            raise ValueError("Defina GEMINI_API_KEY ou GOOGLE_API_KEY nas Secrets.")

        # --- Dados ---
        if not os.path.exists(caminho_arquivo_csv):
            raise FileNotFoundError(f"Erro: O arquivo '{caminho_arquivo_csv}' não foi encontrado.")
        self.df = pd.read_csv(caminho_arquivo_csv)
        self.memoria_analises: List[str] = []
        self.ultima_coluna: Optional[str] = None
        self.session_id = session_id

        # --- LLM ---
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0,
            google_api_key=google_api_key,
        )

        # --- Ferramentas ---
        tools = self._definir_ferramentas()

        # --- Prompt ---
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system",
                 "Você é um assistente de análise de dados. Use as ferramentas quando necessário.\n"
                 "- Se o usuário pedir relações entre variáveis, ofereça 'plotar_mapa_correlacao', 'plotar_dispersao' e 'matriz_dispersao'.\n"
                 "- Para 'tabela cruzada', use 'tabela_cruzada'.\n"
                 "- Para 'valores mais/menos frequentes', use 'frequencias_coluna'.\n"
                 "- Para 'outliers' no dataset, use 'resumo_outliers_dataset'; para coluna específica, IQR/Z-score.\n"
                 "- Para padrões no tempo, use 'tendencias_temporais'.\n"
                 "- Se o usuário omitir a coluna (ex.: 'histograma', 'frequências', 'moda', 'outliers'), use a última coluna mencionada.\n"
                 "- Se o usuário pedir para converter 'Time' em datetime, use 'converter_time_para_datetime'.\n"
                 "- Só pergunte de volta se não houver coluna clara no histórico.\n"
                 "- Quando gerar gráfico, explique brevemente o que ele mostra."),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
                MessagesPlaceholder("agent_scratchpad"),
            ]
        )

        # --- Agente + Executor ---
        base_agent = create_tool_calling_agent(self.llm, tools, prompt)
        self.base_executor = AgentExecutor(
            agent=base_agent,
            tools=tools,
            verbose=False,
            max_iterations=5,
            handle_parsing_errors=True,
        )

        # --- Memória conversacional ---
        self._store = {}  # session_id -> ChatMessageHistory

        def _get_history(session_id: str):
            if session_id not in self._store:
                self._store[session_id] = ChatMessageHistory()
            return self._store[session_id]

        self.agent = RunnableWithMessageHistory(
            self.base_executor,
            _get_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="output",
        )

    # -------------------------
    # Definição das ferramentas
    # -------------------------
    def _definir_ferramentas(self):
        class HistogramaInput(BaseModel):
            coluna: str = Field(description="Coluna numérica para histograma.")

        class FrequenciasInput(BaseModel):
            coluna: str = Field(description="Coluna para calcular frequências.")
            top_n: int = Field(default=10)
            bottom_n: int = Field(default=10)

        class ModaInput(BaseModel):
            coluna: str = Field(description="Coluna (qualquer tipo) para a(s) moda(s).")

        class DispersaoInput(BaseModel):
            x: str = Field(description="Coluna X (numérica).")
            y: str = Field(description="Coluna Y (numérica).")
            hue: Optional[str] = Field(default=None, description="Categórica para colorir.")

        class PairplotInput(BaseModel):
            colunas: Optional[str] = Field(default=None, description="Lista separada por vírgula. Se vazio, escolhe até 6 numéricas.")
            hue: Optional[str] = Field(default=None)

        class CrosstabInput(BaseModel):
            linhas: str = Field(description="Coluna para linhas (categórica).")
            colunas: str = Field(description="Coluna para colunas (categórica).")
            normalizar: bool = Field(default=True)
            heatmap: bool = Field(default=True)

        class OutlierIQRInput(BaseModel):
            coluna: str = Field(description="Coluna numérica.")
            plot: bool = Field(default=False)

        class OutlierZInput(BaseModel):
            coluna: str = Field(description="Coluna numérica.")
            threshold: float = Field(default=3.0)
            plot: bool = Field(default=False)

        class TimeConvertInput(BaseModel):
            origem: Optional[str] = Field(default=None, description="YYYY-MM-DD HH:MM:SS")
            unidade: str = Field(default="s")
            nova_coluna: Optional[str] = Field(default=None)
            criar_features: bool = Field(default=True)

        class TendenciasInput(BaseModel):
            coluna_valor: str = Field(description="Coluna numérica (ex.: Amount).")
            freq: str = Field(default="D", description="H/D/W/M")
            agg: str = Field(default="sum", description="sum/mean/median/count/max/min")
            timestamp_col: Optional[str] = Field(default=None)
            origem: Optional[str] = Field(default=None)
            unidade: str = Field(default="s")
            rolling: Optional[int] = Field(default=None)

        return [
            StructuredTool.from_function(
                func=self.listar_colunas,
                name="listar_colunas",
                description="Lista as colunas do dataset.",
            ),
            StructuredTool.from_function(
                func=self.obter_descricao_geral,
                name="descricao_geral_dados",
                description="Resumo: linhas, colunas, tipos e nulos.",
            ),
            StructuredTool.from_function(
                func=self.obter_estatisticas_descritivas,
                name="estatisticas_descritivas",
                description="Describe numérico.",
            ),
            StructuredTool.from_function(
                func=self.plotar_histograma,
                name="plotar_histograma",
                description="Histograma de uma coluna numérica.",
                args_schema=HistogramaInput,
            ),
            StructuredTool.from_function(
                func=self.mostrar_correlacao,
                name="plotar_mapa_correlacao",
                description="Mapa de calor de correlação.",
            ),
            StructuredTool.from_function(
                func=self.frequencias_coluna,
                name="frequencias_coluna",
                description="Frequências top/bottom.",
                args_schema=FrequenciasInput,
            ),
            StructuredTool.from_function(
                func=self.moda_coluna,
                name="moda_coluna",
                description="Moda(s) da coluna.",
                args_schema=ModaInput,
            ),
            StructuredTool.from_function(
                func=self.plotar_dispersao,
                name="plotar_dispersao",
                description="Dispersão X vs Y.",
                args_schema=DispersaoInput,
            ),
            StructuredTool.from_function(
                func=self.matriz_dispersao,
                name="matriz_dispersao",
                description="Pairplot de colunas.",
                args_schema=PairplotInput,
            ),
            StructuredTool.from_function(
                func=self.tabela_cruzada,
                name="tabela_cruzada",
                description="Crosstab entre duas categóricas.",
                args_schema=CrosstabInput,
            ),
            StructuredTool.from_function(
                func=self.detectar_outliers_iqr,
                name="detectar_outliers_iqr",
                description="Outliers por IQR.",
                args_schema=OutlierIQRInput,
            ),
            StructuredTool.from_function(
                func=self.detectar_outliers_zscore,
                name="detectar_outliers_zscore",
                description="Outliers por Z-score.",
                args_schema=OutlierZInput,
            ),
            StructuredTool.from_function(
                func=self.converter_time_para_datetime,
                name="converter_time_para_datetime",
                description="Converte 'Time' p/ datetime e cria features.",
                args_schema=TimeConvertInput,
            ),
            StructuredTool.from_function(
                func=self.tendencias_temporais,
                name="tendencias_temporais",
                description="Reamostra série temporal e plota.",
                args_schema=TendenciasInput,
            ),
            StructuredTool.from_function(
                func=self.mostrar_conclusoes,
                name="mostrar_conclusoes",
                description="Mostra memória de análises.",
            ),
        ]

    # -------------------------
    # Ferramentas (implementação)
    # -------------------------
    def listar_colunas(self) -> str:
        return f"As colunas são: {', '.join(self.df.columns.tolist())}"

    def obter_descricao_geral(self) -> str:
        buffer = io.StringIO()
        self.df.info(buf=buffer)
        resumo = f"{self.df.shape[0]} linhas x {self.df.shape[1]} colunas\n\n{buffer.getvalue()}"
        self.memoria_analises.append("Descrição geral executada.")
        return resumo

    def obter_estatisticas_descritivas(self) -> str:
        txt = self.df.describe().to_string()
        self.memoria_analises.append("Estatísticas descritivas executadas.")
        return txt

    def plotar_histograma(self, coluna: str) -> str:
        if coluna not in self.df.columns:
            return f"Erro: a coluna '{coluna}' não existe."
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(self.df[coluna], kde=True, stat="density", linewidth=0, ax=ax)
        ax.set_title(f"Histograma: {coluna}")
        ax.set_xlabel(coluna); ax.set_ylabel("Densidade"); ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        self.memoria_analises.append(f"Histograma exibido para {coluna}.")
        self.ultima_coluna = coluna
        return f"Histograma da coluna '{coluna}' exibido."

    def mostrar_correlacao(self) -> str:
        df_num = self.df.select_dtypes(include="number")
        if df_num.empty:
            return "Sem colunas numéricas para correlação."
        fig, ax = plt.subplots(figsize=(12, 9))
        sns.heatmap(df_num.corr(), annot=False, cmap="coolwarm", ax=ax)
        ax.set_title("Mapa de Calor da Correlação")
        st.pyplot(fig)
        self.memoria_analises.append("Mapa de correlação exibido.")
        return "Mapa de correlação exibido."

    def frequencias_coluna(self, coluna: str, top_n: int = 10, bottom_n: int = 10) -> str:
        if coluna not in self.df.columns:
            return f"Erro: a coluna '{coluna}' não existe."
        s = self.df[coluna].dropna()
        partes = []
        if pd.api.types.is_numeric_dtype(s) and s.nunique() > 50:
            try:
                bins = pd.qcut(s, q=min(20, s.nunique()), duplicates="drop")
                cont = bins.value_counts().sort_values(ascending=False)
                top = cont.head(top_n); bottom = cont.tail(bottom_n)
                partes.append("Numérica contínua: usando faixas (quantis).")
                partes.append("\n-- Mais frequentes --")
                partes.extend([f"{idx}: {val}" for idx, val in top.items()])
                partes.append("\n-- Menos frequentes --")
                partes.extend([f"{idx}: {val}" for idx, val in bottom.items()])
            except Exception as e:
                return f"Falha nos quantis: {e}"
        else:
            cont = s.value_counts(dropna=False)
            top = cont.head(top_n)
            bottom = cont[cont > 0].sort_values().head(bottom_n)
            partes.append("\n-- Valores mais frequentes --")
            partes.extend([f"{idx}: {val}" for idx, val in top.items()])
            partes.append("\n-- Valores menos frequentes (não-zero) --")
            partes.extend([f"{idx}: {val}" for idx, val in bottom.items()])
        self.memoria_analises.append(f"Frequências calculadas para '{coluna}'.")
        self.ultima_coluna = coluna
        return "\n".join(partes)

    def moda_coluna(self, coluna: str) -> str:
        if coluna not in self.df.columns:
            return f"Erro: a coluna '{coluna}' não existe."
        modos = self.df[coluna].mode(dropna=True)
        if modos.empty:
            return f"Não foi possível calcular a moda de '{coluna}'."
        valores = ", ".join(map(str, modos.tolist()))
        self.memoria_analises.append(f"Moda de '{coluna}': {valores}")
        self.ultima_coluna = coluna
        return f"Moda(s) de '{coluna}': {valores}"

    def plotar_dispersao(self, x: str, y: str, hue: Optional[str] = None) -> str:
        for col in [x, y] + ([hue] if hue else []):
            if col and col not in self.df.columns:
                return f"Erro: a coluna '{col}' não existe."
        df_plot = self.df[[x, y] + ([hue] if hue else [])].dropna()
        for col in [x, y]:
            if not pd.api.types.is_numeric_dtype(df_plot[col]):
                df_plot[col] = pd.to_numeric(df_plot[col], errors="coerce")
        df_plot = df_plot.dropna(subset=[x, y])
        if df_plot.empty:
            return "Não há dados válidos após limpeza para o gráfico."
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(data=df_plot, x=x, y=y, hue=hue if hue else None, s=20, alpha=0.7, ax=ax)
        ax.set_title(f"Dispersão: {x} vs {y}" + (f" (hue={hue})" if hue else ""))
        ax.grid(True, linestyle="--", alpha=0.3)
        st.pyplot(fig)
        self.memoria_analises.append(f"Dispersão exibida: {x} vs {y}.")
        self.ultima_coluna = y
        return "Gráfico de dispersão exibido."

    def matriz_dispersao(self, colunas: Optional[str] = None, hue: Optional[str] = None) -> str:
        if colunas:
            cols = [c.strip() for c in colunas.split(",") if c.strip()]
        else:
            df_num = self.df.select_dtypes(include="number")
            if df_num.shape[1] == 0:
                return "Sem colunas numéricas."
            vari = df_num.var(numeric_only=True).sort_values(ascending=False)
            cols = vari.index.tolist()[:6]
        for c in cols + ([hue] if hue else []):
            if c and c not in self.df.columns:
                return f"Erro: a coluna '{c}' não existe."
        use_cols = cols + ([hue] if hue else [])
        df_plot = self.df[use_cols].dropna()
        if len(cols) < 2:
            return "Selecione pelo menos 2 colunas."
        g = sns.pairplot(df_plot, vars=cols, hue=hue, corner=True, diag_kind="hist", plot_kws=dict(s=15, alpha=0.7))
        st.pyplot(g.fig)
        self.memoria_analises.append(f"Matriz de dispersão: {cols}")
        return "Matriz de dispersão exibida."

    def tabela_cruzada(self, linhas: str, colunas: str, normalizar: bool = True, heatmap: bool = True) -> str:
        for col in [linhas, colunas]:
            if col not in self.df.columns:
                return f"Erro: a coluna '{col}' não existe."
        ct = pd.crosstab(self.df[linhas].astype(str), self.df[colunas].astype(str))
        tabela = (ct / ct.values.sum()) if normalizar else ct
        if heatmap:
            fig, ax = plt.subplots(figsize=(max(6, 0.4 * len(tabela.columns)), max(4, 0.4 * len(tabela.index))))
            sns.heatmap(tabela, cmap="Blues", ax=ax, fmt=".2f" if normalizar else "d")
            ax.set_title(f"Tabela Cruzada: {linhas} x {colunas}" + (" (normalizada)" if normalizar else ""))
            ax.set_xlabel(colunas); ax.set_ylabel(linhas)
            st.pyplot(fig)
        self.memoria_analises.append(f"Crosstab gerada: {linhas} x {colunas}.")
        self.ultima_coluna = colunas
        return "Tabela cruzada exibida."

    def detectar_outliers_iqr(self, coluna: str, plot: bool = False) -> str:
        if coluna not in self.df.columns:
            return f"Erro: a coluna '{coluna}' não existe."
        s = self.df[coluna].dropna()
        if not pd.api.types.is_numeric_dtype(s):
            return f"A coluna '{coluna}' não é numérica."
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        mask = (s < low) | (s > high)
        n_out, n = int(mask.sum()), int(s.shape[0])
        pct = (n_out / n * 100) if n else 0.0
        if plot:
            fig, ax = plt.subplots(figsize=(8, 1.8))
            sns.boxplot(x=s, ax=ax)
            ax.set_title(f"Boxplot {coluna} (IQR)")
            st.pyplot(fig)
        self.memoria_analises.append(f"IQR em '{coluna}': {pct:.3f}%")
        self.ultima_coluna = coluna
        return f"Outliers (IQR) em '{coluna}': {n_out}/{n} = {pct:.3f}%."

    def detectar_outliers_zscore(self, coluna: str, threshold: float = 3.0, plot: bool = False) -> str:
        if coluna not in self.df.columns:
            return f"Erro: a coluna '{coluna}' não existe."
        s = self.df[coluna].dropna()
        if not pd.api.types.is_numeric_dtype(s):
            return f"A coluna '{coluna}' não é numérica."
        mu, sigma = s.mean(), s.std(ddof=0)
        if sigma == 0 or pd.isna(sigma):
            return f"Desvio padrão zero/NaN em '{coluna}'."
        z = (s - mu) / sigma
        mask = z.abs() > threshold
        n_out, n = int(mask.sum()), int(s.shape[0])
        pct = (n_out / n * 100) if n else 0.0
        if plot:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(z, kde=True, stat="density", linewidth=0, ax=ax)
            ax.set_title(f"Distribuição de Z-scores - {coluna}")
            st.pyplot(fig)
        self.memoria_analises.append(f"Z-score em '{coluna}': {pct:.3f}%")
        self.ultima_coluna = coluna
        return f"Outliers (Z>|{threshold}|) em '{coluna}': {n_out}/{n} = {pct:.3f}%."

    def converter_time_para_datetime(self, origem: Optional[str] = None, unidade: str = "s",
                                     nova_coluna: Optional[str] = None, criar_features: bool = True) -> str:
        col = "Time"
        if col not in self.df.columns:
            return "Erro: coluna 'Time' não encontrada."
        s = pd.to_numeric(self.df[col], errors="coerce")
        if s.isna().all():
            return "Erro: 'Time' não é numérico."
        try:
            td = pd.to_timedelta(s, unit=unidade)
        except Exception as e:
            return f"Erro ao converter para Timedelta: {e}"
        created = []
        if origem:
            base = pd.to_datetime(origem)
            target = nova_coluna or "Time_dt"
            self.df[target] = base + td
            created.append(target)
            modo = f"ancorado em '{origem}'"
        else:
            target = nova_coluna or "Time_delta"
            self.df[target] = td
            created.append(target)
            modo = "relativo (Timedelta)"
        if criar_features:
            seconds_total = td.dt.total_seconds()
            self.df["Time_hour"] = ((seconds_total // 3600) % 24).astype(int)
            self.df["Time_day"] = (seconds_total // 86400).astype(int)
            bins = (seconds_total // 3600).astype(int)
            self.df["Time_bin_1h"] = bins.astype(str) + "h-" + (bins + 1).astype(str) + "h"
            created += ["Time_hour", "Time_day", "Time_bin_1h"]
        msg = f"Conversão concluída ({modo}). Criadas: {', '.join(created)}."
        self.memoria_analises.append(msg)
        return msg

    def tendencias_temporais(self, coluna_valor: str, freq: str = "D", agg: str = "sum",
                             timestamp_col: Optional[str] = None, origem: Optional[str] = None,
                             unidade: str = "s", rolling: Optional[int] = None) -> str:
        if coluna_valor not in self.df.columns:
            return f"Erro: a coluna '{coluna_valor}' não existe."
        if not pd.api.types.is_numeric_dtype(self.df[coluna_valor]):
            return f"A coluna '{coluna_valor}' precisa ser numérica."
        ts_col = None
        if timestamp_col and timestamp_col in self.df.columns:
            if not pd.api.types.is_datetime64_any_dtype(self.df[timestamp_col]):
                self.df[timestamp_col] = pd.to_datetime(self.df[timestamp_col])
            ts_col = timestamp_col
        elif "Time_dt" in self.df.columns:
            ts_col = "Time_dt"
        elif "Time" in self.df.columns:
            origem_padrao = origem or "1970-01-01 00:00:00"
            self.converter_time_para_datetime(origem=origem_padrao, unidade=unidade)
            ts_col = "Time_dt" if "Time_dt" in self.df.columns else None
        if ts_col is None:
            return "Não há coluna temporal. Informe 'timestamp_col' ou inclua 'Time'/'Time_dt'."
        df_ts = self.df[[ts_col, coluna_valor]].dropna().sort_values(ts_col).set_index(ts_col)
        if agg not in {"sum", "mean", "median", "count", "max", "min"}:
            return "agg inválido. Use: sum, mean, median, count, max, min."
        series = getattr(df_ts[coluna_valor].resample(freq), agg)()
        if series.empty:
            return "Série vazia após reamostragem."
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(series.index, series.values, label=f"{agg}({coluna_valor})")
        if rolling and isinstance(rolling, int) and rolling > 1:
            roll = series.rolling(rolling, min_periods=1).mean()
            ax.plot(roll.index, roll.values, linestyle="--", label=f"rolling({rolling})")
        ax.set_title(f"Tendência temporal: {coluna_valor} por {freq} (agg={agg})")
        ax.set_xlabel("Tempo"); ax.set_ylabel(coluna_valor); ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend()
        st.pyplot(fig)
        self.memoria_analises.append(f"Tendência exibida: {coluna_valor}, freq={freq}, agg={agg}.")
        return "Tendência temporal exibida."

    def mostrar_conclusoes(self) -> str:
        return "Nenhuma análise foi realizada ainda." if not self.memoria_analises \
            else "\n--- Conclusões ---\n" + "\n".join(self.memoria_analises)

    # leve pré-processador p/ perguntas
    def _preprocessar_pergunta(self, pergunta: str) -> str:
        t = pergunta.strip()
        if t in self.df.columns:
            self.ultima_coluna = t
            return f"Use a coluna '{t}' como foco. Gere um histograma e calcule outliers por IQR."
        low = t.lower()
        if any(k in low for k in ["histograma", "histogram"]):
            if self.ultima_coluna:
                return f"Plote histograma de '{self.ultima_coluna}' e descreva."
        if any(k in low for k in ["frequenc", "frequências", "frequencias"]):
            if self.ultima_coluna:
                return f"Mostre frequências (top/bottom) de '{self.ultima_coluna}'."
        if "moda" in low and self.ultima_coluna:
            return f"Calcule a moda de '{self.ultima_coluna}'."
        if any(k in low for k in ["outlier", "atípic"]):
            if self.ultima_coluna:
                return f"Detecte outliers por IQR em '{self.ultima_coluna}'."
        if any(k in low for k in ["tendên", "tendenc", "temporal"]):
            if "Amount" in self.df.columns:
                return "Gere tendências temporais agregando 'Amount' por dia."
        if "converter time" in low or ("time" in low and "datetime" in low):
            return "Converta 'Time' para datetime (segundos) e crie features."
        return pergunta

# -------------------------------------------------------------
# UI Streamlit
# -------------------------------------------------------------
st.set_page_config(page_title="Agente EDA (Streamlit)", layout="wide")
st.title("Agente EDA — LangChain + Gemini (+ LangSmith)")
st.caption("Envie um CSV e faça perguntas. Experimente a ferramenta de **tendências temporais**.")

with st.sidebar:
    st.subheader("Upload do CSV")
    uploaded = st.file_uploader("Selecione o arquivo", type=["csv"])

if "agente" not in st.session_state:
    st.session_state.agente = None
    st.session_state.csv_path = None

if uploaded is not None:
    fd, tmp_path = tempfile.mkstemp(suffix=".csv")
    with os.fdopen(fd, "wb") as f:
        f.write(uploaded.getbuffer())
    st.session_state.csv_path = tmp_path
    try:
        st.session_state.agente = AgenteDeAnalise(caminho_arquivo_csv=tmp_path)
        st.success("CSV carregado com sucesso!")
        st.write("**Colunas:**", ", ".join(st.session_state.agente.df.columns.tolist()))
        st.dataframe(st.session_state.agente.df.head())
    except Exception as e:
        st.error(str(e))
else:
    st.info("📄 Faça upload de um CSV para iniciar.")

if st.session_state.agente is not None:
    agente = st.session_state.agente

    st.markdown("### Pergunte ao agente")
    pergunta = st.text_input("Digite sua pergunta (ex.: 'Quais são as correlações?', 'Gerar tendências de Amount por dia')")
    if st.button("Perguntar") and pergunta:
        proc = agente._preprocessar_pergunta(pergunta)
        try:
            resposta = agente.agent.invoke(
                {"input": proc},
                config={
                    "configurable": {"session_id": "ui_streamlit"},
                    "tags": ["ui", "streamlit"],
                    "metadata": {"origin": "ui"},
                },
            )
            st.markdown("#### Resposta")
            st.write(resposta.get("output", resposta))
        except Exception as e:
            st.error(str(e))

    st.markdown("---")
    st.subheader("Tendências temporais (execução direta)")
    df = agente.df
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if num_cols:
        valor_col = st.selectbox("Coluna numérica", options=num_cols, index=num_cols.index("Amount") if "Amount" in num_cols else 0)
        col1, col2, col3 = st.columns(3)
        with col1:
            freq = st.selectbox("Frequência", options=["H", "D", "W", "M"], index=1)
        with col2:
            agg = st.selectbox("Agregação", options=["sum", "mean", "median", "count", "max", "min"], index=0)
        with col3:
            rolling = st.number_input("Janela média móvel (opcional)", min_value=0, value=0, step=1)
        timestamp_col = st.selectbox("Coluna de tempo (opcional)", options=["(auto)"] + df.columns.tolist(), index=0)
        origem = st.text_input("Origem para 'Time' (opcional, ex.: 2013-01-01 00:00:00)")
        if st.button("Gerar tendência"):
            ts_col = None if timestamp_col == "(auto)" else timestamp_col
            msg = agente.tendencias_temporais(
                coluna_valor=valor_col, freq=freq, agg=agg,
                timestamp_col=ts_col, origem=origem or None, unidade="s",
                rolling=(rolling if rolling and rolling > 1 else None),
            )
            st.success(msg)
    else:
        st.info("O dataset não possui colunas numéricas para tendências.")

    st.markdown("### Conclusões acumuladas")
    st.text(agente.mostrar_conclusoes())

