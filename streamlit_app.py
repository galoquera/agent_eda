# streamlit_app.py
# -*- coding: utf-8 -*-
"""
Agente EDA (Streamlit) — LangChain + Gemini (+ LangSmith)
Atende ao enunciado:
- Interface de pergunta + resposta com gráficos quando aplicável
- Upload de CSV genérico
- Memória ("mostrar_conclusoes")
- Tools: descrição, tendências temporais, frequências/moda, relações (dispersão/correlação/crosstab),
  outliers (IQR/Z-score + resumo por dataset) e clusters (k-means)
"""

import os
import io
import tempfile
from typing import List

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

# ---------- LangSmith (opcional, se estiver nas Secrets) ----------
def _enable_langsmith(project: str = "EDA-Agent"):
    ls_key = os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")
    if ls_key and not os.getenv("LANGCHAIN_API_KEY"):
        os.environ["LANGCHAIN_API_KEY"] = ls_key
    if (ls_key or os.getenv("LANGCHAIN_API_KEY")) and not os.getenv("LANGCHAIN_ENDPOINT"):
        os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    if os.getenv("LANGCHAIN_TRACING_V2", "").lower() not in ("1", "true", "yes"):
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
    if project and not os.getenv("LANGCHAIN_PROJECT"):
        os.environ["LANGCHAIN_PROJECT"] = project

load_dotenv()
_enable_langsmith()

# ===================================================================

class AgenteDeAnalise:
    def __init__(self, caminho_arquivo_csv: str, session_id: str = "ui_streamlit"):
        # --- API Key ---
        google_api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            raise ValueError("Defina GEMINI_API_KEY ou GOOGLE_API_KEY nas Secrets do Streamlit.")

        # --- Dados ---
        if not os.path.exists(caminho_arquivo_csv):
            raise FileNotFoundError(f"Arquivo '{caminho_arquivo_csv}' não encontrado.")
        self.df = pd.read_csv(caminho_arquivo_csv)
        self.memoria_analises: List[str] = []
        self.ultima_coluna: str | None = None
        self.session_id = session_id

        # --- LLM ---
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0,
            google_api_key=google_api_key,
        )

        # --- Tools ---
        tools = self._definir_ferramentas()

        # --- Prompt ---
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system",
                 "Você é um agente de EDA. Responda às perguntas do usuário usando as ferramentas quando necessário.\n"
                 "- Descrição dos dados: use 'descricao_geral_dados' e 'estatisticas_descritivas'.\n"
                 "- Distribuições e valores frequentes: 'plotar_histograma', 'frequencias_coluna', 'moda_coluna'.\n"
                 "- Tendências temporais: 'tendencias_temporais' (a coluna 'Time' indica segundos desde a 1ª transação).\n"
                 "- Relações entre variáveis: 'plotar_mapa_correlacao', 'plotar_dispersao', 'tabela_cruzada'.\n"
                 "- Outliers: por coluna ('detectar_outliers_iqr' / 'detectar_outliers_zscore') e resumo geral ('resumo_outliers_dataset').\n"
                 "- Clusters: 'kmeans_clusterizar' e explique brevemente a figura.\n"
                 "- Se o usuário disser apenas o nome de uma coluna, considere-a como foco para histogramas/outliers.\n"
                 "- Se ele perguntar por 'conclusões', use 'mostrar_conclusoes'.\n"
                 "- Quando gerar gráfico, explique o que mostra e registre uma anotação em memória."),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
                MessagesPlaceholder("agent_scratchpad"),
            ]
        )

        base_agent = create_tool_calling_agent(self.llm, tools, prompt)
        self.base_executor = AgentExecutor(
            agent=base_agent,
            tools=tools,
            verbose=False,
            max_iterations=5,
            handle_parsing_errors=True,
        )

        # --- Memória conversacional ---
        self._store = {}
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

    # ------------------------ Tools ------------------------
    def _definir_ferramentas(self):
        # IMPORTANTe: sem Optional nos schemas (compatível com Gemini tool-calling)
        class HistogramaInput(BaseModel):
            coluna: str = Field(description="Coluna numérica para histograma.")

        class FrequenciasInput(BaseModel):
            coluna: str = Field(description="Coluna para frequências top/bottom.")
            top_n: int = Field(default=10)
            bottom_n: int = Field(default=10)

        class ModaInput(BaseModel):
            coluna: str = Field(description="Coluna para calcular moda.")

        class DispersaoInput(BaseModel):
            x: str = Field(description="Coluna X (numérica).")
            y: str = Field(description="Coluna Y (numérica).")
            hue: str = Field(default="", description="Coluna categórica para colorir (opcional).")

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

        class ResumoOutInput(BaseModel):
            method: str = Field(default="iqr", description="iqr ou zscore")
            top_k: int = Field(default=10)

        class KMeansInput(BaseModel):
            colunas: str = Field(default="", description="Lista separada por vírgula (vazio = numéricas).")
            clusters: int = Field(default=3, description="k (min 2)")

        class TimeConvertInput(BaseModel):
            origem: str = Field(default="", description="YYYY-MM-DD HH:MM:SS (vazio = relativo).")
            unidade: str = Field(default="s", description="s/ms/m/h")
            nova_coluna: str = Field(default="", description="Nome da coluna criada (opcional).")
            criar_features: bool = Field(default=True)

        class TendenciasInput(BaseModel):
            coluna: str = Field(description="Coluna numérica (ex.: Amount).")
            freq: str = Field(default="D", description="H/D/W/M")

        return [
            StructuredTool.from_function(self.listar_colunas, name="listar_colunas",
                                         description="Lista as colunas do dataset."),
            StructuredTool.from_function(self.obter_descricao_geral, name="descricao_geral_dados",
                                         description="Resumo (linhas/colunas/tipos/nulos)."),
            StructuredTool.from_function(self.obter_estatisticas_descritivas, name="estatisticas_descritivas",
                                         description="Estatísticas descritivas numéricas."),
            StructuredTool.from_function(self.plotar_histograma, name="plotar_histograma",
                                         description="Histograma de uma coluna numérica.", args_schema=HistogramaInput),
            StructuredTool.from_function(self.frequencias_coluna, name="frequencias_coluna",
                                         description="Top/bottom frequências (bins para numéricas contínuas).", args_schema=FrequenciasInput),
            StructuredTool.from_function(self.moda_coluna, name="moda_coluna",
                                         description="Moda(s) da coluna.", args_schema=ModaInput),
            StructuredTool.from_function(self.mostrar_correlacao, name="plotar_mapa_correlacao",
                                         description="Mapa de calor de correlação entre numéricas."),
            StructuredTool.from_function(self.plotar_dispersao, name="plotar_dispersao",
                                         description="Dispersão X vs Y (hue opcional).", args_schema=DispersaoInput),
            StructuredTool.from_function(self.tabela_cruzada, name="tabela_cruzada",
                                         description="Crosstab entre duas categóricas.", args_schema=CrosstabInput),
            StructuredTool.from_function(self.detectar_outliers_iqr, name="detectar_outliers_iqr",
                                         description="Outliers por IQR.", args_schema=OutlierIQRInput),
            StructuredTool.from_function(self.detectar_outliers_zscore, name="detectar_outliers_zscore",
                                         description="Outliers por Z-score.", args_schema=OutlierZInput),
            StructuredTool.from_function(self.resumo_outliers_dataset, name="resumo_outliers_dataset",
                                         description="Resumo de outliers por coluna (iqr/zscore).", args_schema=ResumoOutInput),
            StructuredTool.from_function(self.kmeans_clusterizar, name="kmeans_clusterizar",
                                         description="K-means e projeção PCA 2D.", args_schema=KMeansInput),
            StructuredTool.from_function(self.converter_time_para_datetime, name="converter_time_para_datetime",
                                         description="Converte 'Time' e cria features.", args_schema=TimeConvertInput),
            StructuredTool.from_function(self.tendencias_temporais, name="tendencias_temporais",
                                         description="Reamostra uma métrica por H/D/W/M e plota.", args_schema=TendenciasInput),
            StructuredTool.from_function(self.mostrar_conclusoes, name="mostrar_conclusoes",
                                         description="Resumo das análises realizadas até agora."),
        ]

    # ---------------- Implementações ----------------
    def listar_colunas(self) -> str:
        return f"Colunas: {', '.join(self.df.columns.tolist())}"

    def obter_descricao_geral(self) -> str:
        buffer = io.StringIO()
        self.df.info(buf=buffer)
        txt = f"{self.df.shape[0]} linhas x {self.df.shape[1]} colunas\n\n{buffer.getvalue()}"
        self.memoria_analises.append("Descrição geral realizada.")
        return txt

    def obter_estatisticas_descritivas(self) -> str:
        txt = self.df.describe().to_string()
        self.memoria_analises.append("Estatísticas descritivas calculadas.")
        return txt

    def plotar_histograma(self, coluna: str) -> str:
        if coluna not in self.df.columns:
            return f"Erro: '{coluna}' não existe."
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.histplot(self.df[coluna], kde=True, stat="density", linewidth=0, ax=ax)
        ax.set_title(f"Histograma: {coluna}"); ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        self.memoria_analises.append(f"Histograma exibido para {coluna}.")
        self.ultima_coluna = coluna
        return f"Histograma de '{coluna}' exibido."

    def frequencias_coluna(self, coluna: str, top_n: int = 10, bottom_n: int = 10) -> str:
        if coluna not in self.df.columns:
            return f"Erro: '{coluna}' não existe."
        s = self.df[coluna].dropna()
        partes = []
        if pd.api.types.is_numeric_dtype(s) and s.nunique() > 50:
            try:
                bins = pd.qcut(s, q=min(20, s.nunique()), duplicates="drop")
                cont = bins.value_counts().sort_values(ascending=False)
                partes.append("Numérica contínua: usando faixas (quantis).")
                partes.append("\n-- Mais frequentes --")
                for idx, val in cont.head(top_n).items():
                    partes.append(f"{idx}: {val}")
                partes.append("\n-- Menos frequentes --")
                for idx, val in cont.tail(bottom_n).items():
                    partes.append(f"{idx}: {val}")
            except Exception as e:
                return f"Falha nos quantis: {e}"
        else:
            cont = s.value_counts(dropna=False)
            partes.append("\n-- Valores mais frequentes --")
            for idx, val in cont.head(top_n).items():
                partes.append(f"{idx}: {val}")
            partes.append("\n-- Valores menos frequentes (não-zero) --")
            for idx, val in cont[cont > 0].sort_values().head(bottom_n).items():
                partes.append(f"{idx}: {val}")
        self.memoria_analises.append(f"Frequências calculadas para '{coluna}'.")
        self.ultima_coluna = coluna
        return "\n".join(partes)

    def moda_coluna(self, coluna: str) -> str:
        if coluna not in self.df.columns:
            return f"Erro: '{coluna}' não existe."
        modos = self.df[coluna].mode(dropna=True)
        if modos.empty:
            return f"Não foi possível calcular a(s) moda(s) de '{coluna}'."
        valores = ", ".join(map(str, modos.tolist()))
        self.memoria_analises.append(f"Moda calculada para '{coluna}': {valores}")
        self.ultima_coluna = coluna
        return f"Moda(s) de '{coluna}': {valores}"

    def mostrar_correlacao(self) -> str:
        df_num = self.df.select_dtypes(include="number")
        if df_num.empty:
            return "Sem colunas numéricas para correlação."
        fig, ax = plt.subplots(figsize=(12, 9))
        sns.heatmap(df_num.corr(), cmap="coolwarm", ax=ax)
        ax.set_title("Mapa de calor da correlação")
        st.pyplot(fig)
        self.memoria_analises.append("Mapa de correlação exibido.")
        return "Mapa de correlação exibido."

    def plotar_dispersao(self, x: str, y: str, hue: str = "") -> str:
        for c in [x, y] + ([hue] if hue else []):
            if c and c not in self.df.columns:
                return f"Erro: '{c}' não existe."
        df_plot = self.df[[x, y] + ([hue] if hue else [])].dropna()
        for c in [x, y]:
            if not pd.api.types.is_numeric_dtype(df_plot[c]):
                df_plot[c] = pd.to_numeric(df_plot[c], errors="coerce")
        df_plot = df_plot.dropna(subset=[x, y])
        if df_plot.empty:
            return "Sem dados válidos após limpeza."
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(data=df_plot, x=x, y=y, hue=(hue if hue else None), s=20, alpha=0.7, ax=ax)
        ax.set_title(f"Dispersão: {x} vs {y}" + (f" (hue={hue})" if hue else ""))
        ax.grid(True, linestyle="--", alpha=0.3)
        st.pyplot(fig)
        self.memoria_analises.append(f"Dispersão exibida: {x} vs {y}.")
        self.ultima_coluna = y
        return "Gráfico de dispersão exibido."

    def tabela_cruzada(self, linhas: str, colunas: str, normalizar: bool = True, heatmap: bool = True) -> str:
        for c in [linhas, colunas]:
            if c not in self.df.columns:
                return f"Erro: '{c}' não existe."
        ct = pd.crosstab(self.df[linhas].astype(str), self.df[colunas].astype(str))
        tabela = (ct / ct.values.sum()) if normalizar else ct
        if heatmap:
            fig, ax = plt.subplots(figsize=(max(6, 0.4 * len(tabela.columns)), max(4, 0.4 * len(tabela.index))))
            sns.heatmap(tabela, cmap="Blues", ax=ax, fmt=".2f" if normalizar else "d")
            ax.set_title(f"Crosstab: {linhas} x {colunas}" + (" (normalizada)" if normalizar else ""))
            ax.set_xlabel(colunas); ax.set_ylabel(linhas)
            st.pyplot(fig)
        self.memoria_analises.append(f"Crosstab gerada: {linhas} x {colunas}.")
        self.ultima_coluna = colunas
        return "Tabela cruzada exibida."

    def detectar_outliers_iqr(self, coluna: str, plot: bool = False) -> str:
        if coluna not in self.df.columns:
            return f"Erro: '{coluna}' não existe."
        s = self.df[coluna].dropna()
        if not pd.api.types.is_numeric_dtype(s):
            return f"'{coluna}' não é numérica."
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
        self.memoria_analises.append(f"IQR '{coluna}': {n_out}/{n} ({pct:.3f}%).")
        self.ultima_coluna = coluna
        return f"Outliers (IQR) em '{coluna}': {n_out}/{n} = {pct:.3f}%."

    def detectar_outliers_zscore(self, coluna: str, threshold: float = 3.0, plot: bool = False) -> str:
        if coluna not in self.df.columns:
            return f"Erro: '{coluna}' não existe."
        s = self.df[coluna].dropna()
        if not pd.api.types.is_numeric_dtype(s):
            return f"'{coluna}' não é numérica."
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
        self.memoria_analises.append(f"Z-score '{coluna}': {n_out}/{n} ({pct:.3f}%).")
        self.ultima_coluna = coluna
        return f"Outliers (Z>|{threshold}|) em '{coluna}': {n_out}/{n} = {pct:.3f}%."

    def resumo_outliers_dataset(self, method: str = "iqr", top_k: int = 10) -> str:
        df_num = self.df.select_dtypes(include="number")
        if df_num.empty:
            return "Sem colunas numéricas."
        linhas = []
        for col in df_num.columns:
            s = df_num[col].dropna()
            n = int(s.shape[0])
            if n == 0:
                linhas.append((col, 0.0, 0, 0)); continue
            if method.lower() == "zscore":
                mu, sigma = s.mean(), s.std(ddof=0)
                if sigma == 0 or pd.isna(sigma):
                    pct, cnt = 0.0, 0
                else:
                    z = (s - mu) / sigma
                    mask = z.abs() > 3.0
                    cnt = int(mask.sum()); pct = (cnt / n * 100)
            else:
                q1, q3 = s.quantile(0.25), s.quantile(0.75)
                iqr = q3 - q1
                low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                mask = (s < low) | (s > high)
                cnt = int(mask.sum()); pct = (cnt / n * 100)
            linhas.append((col, pct, cnt, n))
        linhas.sort(key=lambda x: x[1], reverse=True)
        top = linhas[:max(1, top_k)]
        media_pct = sum(p for _, p, _, _ in linhas) / len(linhas)
        partes = [f"Resumo de outliers por {method.upper()} (top {len(top)}):"]
        for col, pct, cnt, n in top:
            partes.append(f"- {col}: {cnt}/{n} = {pct:.3f}%")
        partes.append(f"Média de % de outliers nas numéricas: {media_pct:.3f}%")
        self.memoria_analises.append("Resumo de outliers calculado.")
        return "\n".join(partes)

    def kmeans_clusterizar(self, colunas: str = "", clusters: int = 3) -> str:
        try:
            from sklearn.preprocessing import StandardScaler
            from sklearn.cluster import KMeans
            from sklearn.decomposition import PCA
        except Exception:
            return "k-means requer scikit-learn. Instale 'scikit-learn' no requirements."
        if clusters < 2:
            clusters = 2
        cols = [c.strip() for c in colunas.split(",") if c.strip()] if colunas else self.df.select_dtypes(include="number").columns.tolist()
        if not cols:
            return "Sem colunas numéricas para clusterização."
        X = self.df[cols].dropna()
        if X.empty:
            return "Sem linhas após remoção de NAs."
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        km = KMeans(n_clusters=clusters, n_init=10, random_state=42)
        labels = km.fit_predict(Xs)
        pca = PCA(n_components=2, random_state=42)
        XY = pca.fit_transform(Xs)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(x=XY[:, 0], y=XY[:, 1], hue=labels, legend=True, ax=ax)
        ax.set_title(f"K-means (k={clusters}) — PCA 2D"); ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        resumo = f"k-means concluído (k={clusters}) nas colunas {cols}. Inércia: {km.inertia_:.2f}."
        self.memoria_analises.append(resumo)
        return resumo

    def converter_time_para_datetime(self, origem: str = "", unidade: str = "s",
                                     nova_coluna: str = "", criar_features: bool = True) -> str:
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
        if origem.strip():
            base = pd.to_datetime(origem)
            target = (nova_coluna or "Time_dt").strip()
            self.df[target] = base + td
            created.append(target)
            modo = f"ancorado em '{origem}'"
        else:
            target = (nova_coluna or "Time_delta").strip()
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
        msg = f"Conversão de 'Time' {modo}. Criadas: {', '.join(created)}."
        self.memoria_analises.append(msg)
        return msg

    def tendencias_temporais(self, coluna: str, freq: str = "D") -> str:
        if coluna not in self.df.columns:
            return f"Erro: '{coluna}' não existe."
        if not pd.api.types.is_numeric_dtype(self.df[coluna]):
            return f"'{coluna}' deve ser numérica."
        ts_col = None
        if "Time_dt" in self.df.columns:
            ts_col = "Time_dt"
        elif "Time" in self.df.columns:
            self.converter_time_para_datetime(origem="1970-01-01 00:00:00", unidade="s", nova_coluna="Time_dt", criar_features=False)
            ts_col = "Time_dt"
        else:
            return "Não há coluna temporal ('Time' ou 'Time_dt')."
        df_ts = self.df[[ts_col, coluna]].dropna().sort_values(ts_col).set_index(ts_col)
        series = df_ts[coluna].resample(freq).mean()
        if series.empty:
            return "Série vazia após reamostragem."
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(series.index, series.values, label=f"mean({coluna})")
        ax.set_title(f"Tendência temporal: {coluna} por {freq}")
        ax.set_xlabel("Tempo"); ax.set_ylabel(coluna); ax.grid(True, linestyle="--", alpha=0.3); ax.legend()
        st.pyplot(fig)
        self.memoria_analises.append(f"Tendência temporal de {coluna} ({freq}) exibida.")
        return "Tendência temporal exibida."

    def mostrar_conclusoes(self) -> str:
        return "Nenhuma análise registrada." if not self.memoria_analises \
            else "\n--- Conclusões ---\n" + "\n".join(self.memoria_analises)

    # Pré-processamento leve (ajuda o agente a usar a última coluna)
    def _preprocessar_pergunta(self, pergunta: str) -> str:
        t = pergunta.strip()
        if t in self.df.columns:
            self.ultima_coluna = t
            return f"Use a coluna '{t}' como foco: gere um histograma e calcule outliers por IQR."
        low = t.lower()
        if "histogram" in low or "histograma" in low:
            if self.ultima_coluna:
                return f"Plote histograma de '{self.ultima_coluna}' e descreva."
        if "frequenc" in low or "frequênc" in low:
            if self.ultima_coluna:
                return f"Mostre frequências (top/bottom) de '{self.ultima_coluna}'."
        if "moda" in low and self.ultima_coluna:
            return f"Calcule a moda de '{self.ultima_coluna}'."
        if "outlier" in low or "atípic" in low:
            if self.ultima_coluna:
                return f"Detecte outliers (IQR) em '{self.ultima_coluna}'."
        if "tendên" in low or "tendenc" in low or "temporal" in low:
            if "Amount" in self.df.columns:
                return "Mostre tendências temporais de 'Amount' por dia."
        if "converter time" in low or ("time" in low and "datetime" in low):
            return "Converta 'Time' para datetime (s) e crie features."
        return pergunta

# ========================= UI Streamlit =========================
st.set_page_config(page_title="Agente EDA (Streamlit)", layout="wide")
st.title("Agente EDA — LangChain + Gemini")
st.caption("Faça upload de um CSV e pergunte ao agente. Ele pode gerar gráficos e registrar conclusões.")

with st.sidebar:
    st.subheader("Upload do CSV")
    uploaded = st.file_uploader("Selecione um arquivo .csv", type=["csv"])

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
        st.success("CSV carregado.")
        with st.expander("Prévia do dataset"):
            st.dataframe(st.session_state.agente.df.head(30))
            st.write("**Colunas:** ", ", ".join(st.session_state.agente.df.columns.tolist()))
    except Exception as e:
        st.error(str(e))
elif st.session_state.agente is None:
    st.info("📄 Envie um CSV para começar.")

if st.session_state.agente is not None:
    agente = st.session_state.agente

    st.markdown("### Faça sua pergunta")
    pergunta = st.text_input("Ex.: 'Quais variáveis estão mais correlacionadas?', 'Gerar clusters k=3', 'mostrar_conclusoes'")
    if st.button("Perguntar") and pergunta:
        proc = agente._preprocessar_pergunta(pergunta)
        try:
            resposta = agente.agent.invoke(
                {"input": proc},
                config={"configurable": {"session_id": "ui_streamlit"},
                        "tags": ["ui", "streamlit"], "metadata": {"origin": "ui"}},
            )
            st.markdown("#### Resposta")
            st.write(resposta.get("output", resposta))
        except Exception as e:
            st.error(str(e))

    st.markdown("### Conclusões do agente")
    st.text(agente.mostrar_conclusoes())
