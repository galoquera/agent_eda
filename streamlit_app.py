# -*- coding: utf-8 -*-
"""
Agente EDA com LangChain + Gemini (+ LangSmith) + UI Web (Streamlit)
- CSV dinâmico: aceita --csv caminho.csv no CLI e upload no UI web
- Nova ferramenta: tendencias_temporais (reamostragem por hora/dia/semana/mês)

Como usar (CLI):
    python agente_eda.py --csv seu_arquivo.csv

Como usar (UI Web):
    streamlit run agente_eda.py

Requisitos (pip):
    pip install -U pandas matplotlib seaborn python-dotenv langchain langchain-google-genai scikit-learn streamlit

Variáveis de ambiente (.env):
    GEMINI_API_KEY=...
    # ou GOOGLE_API_KEY=...
    LANGSMITH_API_KEY=...         # opcional (para tracing)
    LANGCHAIN_TRACING_V2=true     # opcional
    LANGCHAIN_ENDPOINT=https://api.smith.langchain.com  # opcional
    LANGCHAIN_PROJECT=EDA-Agent   # opcional
"""

import os
import io
import sys
import argparse
import tempfile
from typing import Optional, List

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.tools import StructuredTool
from pydantic import BaseModel, Field

# Opcional: cliente LangSmith (para interações extras via API se desejar)
try:
    from langsmith import Client as LangSmithClient  # type: ignore
except Exception:  # pragma: no cover
    LangSmithClient = None

load_dotenv()


# -------------------------------------------------------------
# Utilitário para detectar execução via Streamlit
# -------------------------------------------------------------
def _is_streamlit_runtime() -> bool:
    return (
        os.environ.get("STREAMLIT_SERVER_RUNNING") == "1"
        or os.environ.get("STREAMLIT_RUNTIME") is not None
    )


# -------------------------------------------------------------
# Agente
# -------------------------------------------------------------
class AgenteDeAnalise:
    def __init__(
        self,
        caminho_arquivo_csv: str,
        session_id: str = "default",
        langsmith_project: Optional[str] = "EDA-Agent",
        run_tags: Optional[List[str]] = None,
    ):
        # --- LangSmith (habilita tracing V2, endpoint e projeto) ---
        self._enable_langsmith(langsmith_project)

        # --- API Key LLM ---
        google_api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            raise ValueError("Defina GEMINI_API_KEY ou GOOGLE_API_KEY no .env/ambiente.")

        # --- Dados ---
        if not os.path.exists(caminho_arquivo_csv):
            raise FileNotFoundError(f"Erro: O arquivo '{caminho_arquivo_csv}' não foi encontrado.")
        print(f"Carregando dados de '{caminho_arquivo_csv}'...")
        self.df = pd.read_csv(caminho_arquivo_csv)
        self.memoria_analises: List[str] = []
        print("Dados carregados com sucesso!")

        self.ultima_coluna: Optional[str] = None
        self.session_id = session_id

        # Tags/metadata para aparecerem nos runs do LangSmith
        self.run_tags = run_tags or [
            "eda",
            "langchain",
            "gemini",
            "agente-eda",
            "tendencias-temporais",
        ]
        self.run_metadata = {
            "app": "AgenteDeAnalise",
            "csv": os.path.basename(caminho_arquivo_csv),
            "session_id": session_id,
        }

        # (Opcional) instanciar cliente LangSmith
        self.langsmith_client = None
        if LangSmithClient is not None:
            try:
                self.langsmith_client = LangSmithClient()  # usa variáveis de ambiente
            except Exception:
                self.langsmith_client = None  # segue sem cliente explícito

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
                (
                    "system",
                    "Você é um assistente de análise de dados. Use as ferramentas quando necessário.\n"
                    "- Se o usuário pedir relações entre variáveis, ofereça 'plotar_mapa_correlacao', 'plotar_dispersao' e 'matriz_dispersao'.\n"
                    "- Para 'tabela cruzada', use 'tabela_cruzada'.\n"
                    "- Para 'valores mais/menos frequentes', use 'frequencias_coluna'.\n"
                    "- Para 'outliers' no dataset, use 'resumo_outliers_dataset'; para coluna específica, IQR/Z-score.\n"
                    "- Para 'tendências temporais' (padrões no tempo), use 'tendencias_temporais'.\n"
                    "- Se o usuário omitir a coluna (ex.: 'histograma', 'frequências', 'moda', 'outliers'), use a última coluna mencionada.\n"
                    "- Se o usuário pedir para converter 'Time' em datetime, use 'converter_time_para_datetime'.\n"
                    "- Só pergunte de volta se não houver coluna clara no histórico.\n"
                    "- Quando gerar gráfico, explique brevemente o que ele mostra.",
                ),
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

    # --- LangSmith helpers ---
    def _enable_langsmith(self, project: Optional[str]):
        """Habilita LangSmith tracing V2 para LangChain via variáveis de ambiente."""
        ls_key = os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")
        if ls_key and not os.getenv("LANGCHAIN_API_KEY"):
            os.environ["LANGCHAIN_API_KEY"] = ls_key
        if not os.getenv("LANGCHAIN_ENDPOINT") and (ls_key or os.getenv("LANGCHAIN_API_KEY")):
            os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
        if os.getenv("LANGCHAIN_TRACING_V2", "").lower() not in ("1", "true", "yes"):
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
        if project and not os.getenv("LANGCHAIN_PROJECT"):
            os.environ["LANGCHAIN_PROJECT"] = project

    def _definir_ferramentas(self):
        # Schemas
        class HistogramaInput(BaseModel):
            coluna: str = Field(description="Nome da coluna para gerar o histograma.")

        class FrequenciasInput(BaseModel):
            coluna: str = Field(description="Nome da coluna para calcular frequências.")
            top_n: int = Field(default=10, description="Top N valores/faixas mais frequentes.")
            bottom_n: int = Field(default=10, description="Bottom N valores/faixas menos frequentes.")

        class ModaInput(BaseModel):
            coluna: str = Field(description="Nome da coluna para calcular a(s) moda(s).")

        class KMeansInput(BaseModel):
            colunas: Optional[str] = Field(
                default=None,
                description="Colunas separadas por vírgula. Se vazio, usa todas as numéricas.",
            )
            clusters: int = Field(default=3, description="Número de clusters (k, mínimo 2).")

        class OutlierIQRInput(BaseModel):
            coluna: str = Field(description="Coluna numérica para detecção por IQR.")
            plot: bool = Field(default=False, description="Se True, plota boxplot.")

        class OutlierZScoreInput(BaseModel):
            coluna: str = Field(description="Coluna numérica para detecção por Z-score.")
            threshold: float = Field(default=3.0, description="Limite do |z| (padrão 3.0).")
            plot: bool = Field(default=False, description="Se True, plota histograma.")

        class OutlierIFInput(BaseModel):
            colunas: Optional[str] = Field(
                default=None,
                description="Colunas separadas por vírgula. Se vazio, usa todas as numéricas.",
            )
            contamination: float = Field(default=0.01, description="Proporção esperada de outliers (ex.: 0.01).")

        class ResumoOutliersInput(BaseModel):
            method: str = Field(default="iqr", description="Método: 'iqr' ou 'zscore'.")
            top_k: int = Field(default=10, description="Mostrar top_k colunas com mais outliers.")

        class DispersaoInput(BaseModel):
            x: str = Field(description="Coluna para o eixo X (numérica).")
            y: str = Field(description="Coluna para o eixo Y (numérica).")
            hue: Optional[str] = Field(default=None, description="Coluna categórica para colorir pontos.")
            amostra: int = Field(default=5000, description="Máximo de linhas amostradas para o gráfico.")

        class PairplotInput(BaseModel):
            colunas: Optional[str] = Field(
                default=None,
                description="Colunas separadas por vírgula. Se vazio, usa até 6 numéricas automaticamente."
            )
            hue: Optional[str] = Field(default=None, description="Coluna categórica para colorir.")
            amostra: int = Field(default=3000, description="Máximo de linhas amostradas para o pairplot.")
            corner: bool = Field(default=True, description="Se True, mostra apenas metade inferior da matriz.")

        class CrosstabInput(BaseModel):
            linhas: str = Field(description="Coluna para as linhas (categórica/discreta).")
            colunas: str = Field(description="Coluna para as colunas (categórica/discreta).")
            normalizar: bool = Field(default=True, description="Normaliza a tabela (proporções).")
            normalizar_modo: Optional[str] = Field(
                default="all",
                description="Modo de normalização: 'all' (total), 'index' (por linha), 'columns' (por coluna) ou None."
            )
            top_k: int = Field(default=20, description="Limita categorias por eixo para evitar tabelas gigantes.")
            heatmap: bool = Field(default=True, description="Exibe heatmap da tabela.")
            annot: bool = Field(default=False, description="Anotar valores no heatmap (pode poluir).")

        class TimeConvertInput(BaseModel):
            origem: Optional[str] = Field(
                default=None,
                description="Data/hora inicial 'YYYY-MM-DD HH:MM:SS'. Se omitido, cria colunas relativas."
            )
            unidade: str = Field(
                default="s",
                description="Unidade do Time: 's' (segundos), 'ms', 'm', 'h'. Padrão: 's'."
            )
            nova_coluna: Optional[str] = Field(
                default=None,
                description="Nome da nova coluna. Padrão: 'Time_dt' (com origem) ou 'Time_delta' (sem origem)."
            )
            criar_features: bool = Field(
                default=True,
                description="Se True, cria Time_hour, Time_day e Time_bin_1h."
            )

        class TendenciasInput(BaseModel):
            coluna_valor: str = Field(description="Coluna numérica a ser agregada (ex.: 'Amount').")
            freq: str = Field(
                default="D",
                description="Frequência de reamostragem: 'H' (hora), 'D' (dia), 'W' (semana), 'M' (mês).",
            )
            agg: str = Field(
                default="sum",
                description="Agregação: 'sum', 'mean', 'median', 'count', 'max', 'min'.",
            )
            timestamp_col: Optional[str] = Field(
                default=None,
                description="Coluna datetime a usar (se None, usa 'Time_dt' ou converte 'Time').",
            )
            origem: Optional[str] = Field(
                default=None,
                description="Origem para converter 'Time' em datetime, se necessário (YYYY-MM-DD HH:MM:SS).",
            )
            unidade: str = Field(
                default="s",
                description="Unidade da coluna 'Time' caso precise converter: 's','ms','m','h'.",
            )
            rolling: Optional[int] = Field(
                default=None,
                description="Janela de média móvel (inteiro de períodos) para suavização opcional.",
            )

        return [
            StructuredTool.from_function(
                name="listar_colunas",
                func=self.listar_colunas,
                description="Retorna os nomes das colunas do dataframe.",
            ),
            StructuredTool.from_function(
                name="descricao_geral_dados",
                func=self.obter_descricao_geral,
                description="Resumo do dataset: linhas, colunas, tipos e nulos.",
            ),
            StructuredTool.from_function(
                name="estatisticas_descritivas",
                func=self.obter_estatisticas_descritivas,
                description="Estatísticas descritivas das colunas numéricas.",
            ),
            StructuredTool.from_function(
                name="plotar_histograma",
                func=self.plotar_histograma,
                description="Gera e exibe um histograma para uma coluna numérica.",
                args_schema=HistogramaInput,
            ),
            StructuredTool.from_function(
                name="plotar_mapa_correlacao",
                func=self.mostrar_correlacao,
                description="Mapa de calor de correlação entre colunas numéricas (método: 'pearson', 'spearman', 'kendall').",
            ),
            StructuredTool.from_function(
                name="frequencias_coluna",
                func=self.frequencias_coluna,
                description=(
                    "Top/bottom frequências de uma coluna. Se for numérica contínua, usa faixas (quantis)."
                ),
                args_schema=FrequenciasInput,
            ),
            StructuredTool.from_function(
                name="moda_coluna",
                func=self.moda_coluna,
                description="Calcula a(s) moda(s) da coluna.",
                args_schema=ModaInput,
            ),
            StructuredTool.from_function(
                name="kmeans_clusterizar",
                func=self.kmeans_clusterizar,
                description="Executa k-means e plota PCA 2D. Requer scikit-learn.",
                args_schema=KMeansInput,
            ),
            StructuredTool.from_function(
                name="detectar_outliers_iqr",
                func=self.detectar_outliers_iqr,
                description="Detecta outliers por IQR em uma coluna; opção de boxplot.",
                args_schema=OutlierIQRInput,
            ),
            StructuredTool.from_function(
                name="detectar_outliers_zscore",
                func=self.detectar_outliers_zscore,
                description="Detecta outliers por Z-score em uma coluna.",
                args_schema=OutlierZScoreInput,
            ),
            StructuredTool.from_function(
                name="detectar_outliers_isolation_forest",
                func=self.detectar_outliers_isolation_forest,
                description="Isolation Forest para detectar outliers em múltiplas colunas. Requer scikit-learn.",
                args_schema=OutlierIFInput,
            ),
            StructuredTool.from_function(
                name="resumo_outliers_dataset",
                func=self.resumo_outliers_dataset,
                description="Resumo (todas as colunas numéricas) da % de outliers por IQR/Z-score.",
                args_schema=ResumoOutliersInput,
            ),
            StructuredTool.from_function(
                name="mostrar_conclusoes",
                func=self.mostrar_conclusoes,
                description="Resumo das análises realizadas até agora.",
            ),
            StructuredTool.from_function(
                name="plotar_dispersao",
                func=self.plotar_dispersao,
                description="Gráfico de dispersão entre duas colunas numéricas (com hue opcional).",
                args_schema=DispersaoInput,
            ),
            StructuredTool.from_function(
                name="matriz_dispersao",
                func=self.matriz_dispersao,
                description="Matriz de dispersão (pairplot) para colunas selecionadas (com hue opcional).",
                args_schema=PairplotInput,
            ),
            StructuredTool.from_function(
                name="tabela_cruzada",
                func=self.tabela_cruzada,
                description="Tabela cruzada (crosstab) entre duas colunas categóricas/discretas; heatmap opcional.",
                args_schema=CrosstabInput,
            ),
            StructuredTool.from_function(
                name="converter_time_para_datetime",
                func=self.converter_time_para_datetime,
                description=(
                    "Converte 'Time' (offset numérico) para datetime relativo ou ancorado em uma origem, e cria features temporais."
                ),
                args_schema=TimeConvertInput,
            ),
            StructuredTool.from_function(
                name="tendencias_temporais",
                func=self.tendencias_temporais,
                description=(
                    "Reamostra séries temporais agregando uma coluna numérica por H/D/W/M e plota linha (opcional smoothing)."
                ),
                args_schema=TendenciasInput,
            ),
        ]

    # -------------------------
    # Ferramentas (funções)
    # -------------------------
    def listar_colunas(self) -> str:
        colunas = self.df.columns.tolist()
        return f"As colunas são: {', '.join(colunas)}"

    def obter_descricao_geral(self) -> str:
        buffer = io.StringIO()
        self.df.info(buf=buffer)
        info_str = buffer.getvalue()
        resumo = f"O dataset possui {self.df.shape[0]} linhas e {self.df.shape[1]} colunas.\n\n{info_str}"
        self.memoria_analises.append(f"Análise de Descrição Geral: {resumo}")
        return resumo

    def obter_estatisticas_descritivas(self) -> str:
        descricao = self.df.describe().to_string()
        self.memoria_analises.append(f"Análise Estatística realizada:\n{descricao}")
        return descricao

    def plotar_histograma(self, coluna: str) -> str:
        if coluna not in self.df.columns:
            return f"Erro: a coluna '{coluna}' não existe. Use 'listar_colunas' para ver opções."
        plt.figure(figsize=(10, 6))
        sns.histplot(self.df[coluna], kde=True, stat="density", linewidth=0)
        plt.title(f"Histograma da Coluna: {coluna}")
        plt.xlabel(coluna)
        plt.ylabel("Densidade")
        plt.grid(True)
        plt.show()
        resultado = f"Histograma da coluna '{coluna}' gerado e exibido."
        self.memoria_analises.append(f"Análise Gráfica: {resultado}")
        self.ultima_coluna = coluna
        return resultado

    def mostrar_correlacao(self, method: str = "pearson") -> str:
        """Mapa de calor de correlação; method: 'pearson' (padrão), 'spearman' ou 'kendall'."""
        df_num = self.df.select_dtypes(include="number")
        if df_num.empty:
            return "Nenhuma coluna numérica encontrada para correlação."
        plt.figure(figsize=(12, 9))
        corr = df_num.corr(method=method)
        sns.heatmap(corr, annot=False, cmap="coolwarm")
        plt.title(f"Mapa de Calor da Correlação (método: {method})")
        plt.show()
        resultado = f"Mapa de calor de correlação (método: {method}) gerado e exibido."
        self.memoria_analises.append(f"Análise de Correlação: {resultado}")
        return resultado

    def frequencias_coluna(self, coluna: str, top_n: int = 10, bottom_n: int = 10) -> str:
        if coluna not in self.df.columns:
            return f"Erro: a coluna '{coluna}' não existe. Use 'listar_colunas' para ver opções."
        s = self.df[coluna].dropna()
        partes = []
        if pd.api.types.is_numeric_dtype(s) and s.nunique() > 50:
            try:
                bins = pd.qcut(s, q=min(20, s.nunique()), duplicates="drop")
                cont = bins.value_counts().sort_values(ascending=False)
                top = cont.head(top_n)
                bottom = cont.tail(bottom_n)
                partes.append("Coluna numérica contínua detectada: usando faixas (quantis).")
                partes.append("\n-- Faixas mais frequentes --")
                partes.extend([f"{idx}: {val}" for idx, val in top.items()])
                partes.append("\n-- Faixas menos frequentes --")
                partes.extend([f"{idx}: {val}" for idx, val in bottom.items()])
            except Exception as e:
                return f"Não foi possível calcular quantis para '{coluna}': {e}"
        else:
            cont = s.value_counts(dropna=False)
            top = cont.head(top_n)
            bottom = cont[cont > 0].sort_values().head(bottom_n)
            partes.append("\n-- Valores mais frequentes --")
            partes.extend([f"{idx}: {val}" for idx, val in top.items()])
            partes.append("\n-- Valores menos frequentes (não-zero) --")
            partes.extend([f"{idx}: {val}" for idx, val in bottom.items()])
        self.memoria_analises.append(
            f"Frequências calculadas para '{coluna}' (top {top_n}, bottom {bottom_n})."
        )
        self.ultima_coluna = coluna
        return "\n".join(partes)

    def moda_coluna(self, coluna: str) -> str:
        if coluna not in self.df.columns:
            return f"Erro: a coluna '{coluna}' não existe."
        modos = self.df[coluna].mode(dropna=True)
        if modos.empty:
            return f"Não foi possível calcular a moda de '{coluna}'."
        valores = ", ".join(map(str, modos.tolist()))
        self.memoria_analises.append(f"Moda calculada para '{coluna}': {valores}")
        self.ultima_coluna = coluna
        return f"Moda(s) de '{coluna}': {valores}"

    def kmeans_clusterizar(self, colunas: Optional[str] = None, clusters: int = 3) -> str:
        """Executa k-means e plota uma projeção PCA 2D para visualização."""
        try:
            from sklearn.preprocessing import StandardScaler
            from sklearn.cluster import KMeans
            from sklearn.decomposition import PCA
        except Exception:
            return "A ferramenta de k-means requer scikit-learn. Instale com: pip install scikit-learn"

        if clusters < 2:
            clusters = 2

        if colunas:
            cols = [c.strip() for c in colunas.split(",") if c.strip()]
        else:
            cols = self.df.select_dtypes(include="number").columns.tolist()

        if not cols:
            return "Não há colunas numéricas suficientes para clusterização."
        X = self.df[cols].dropna()
        if X.empty:
            return "Após remover NAs, não sobraram linhas para clusterização."

        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)

        km = KMeans(n_clusters=clusters, n_init=10, random_state=42)
        labels = km.fit_predict(Xs)

        # Preserva rótulos no DataFrame original (útil p/ análises futuras)
        self.df.loc[X.index, "Cluster_kmeans"] = labels

        pca = PCA(n_components=2, random_state=42)
        XY = pca.fit_transform(Xs)

        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=XY[:, 0], y=XY[:, 1], hue=labels, legend=True)
        plt.title(f"K-means (k={clusters}) em PCA 2D")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.grid(True)
        plt.show()

        resumo = (
            f"k-means concluído com k={clusters} nas colunas {cols}.\n"
            f"Inércia: {km.inertia_:.2f}. Gráfico PCA 2D exibido. Rótulos salvos na coluna 'Cluster_kmeans'."
        )
        self.memoria_analises.append(resumo)
        return resumo

    # ----------------- OUTLIERS -----------------
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
        n_out = int(mask.sum())
        n = int(s.shape[0])
        pct = (n_out / n * 100) if n else 0.0
        if plot:
            plt.figure(figsize=(8, 1.8))
            sns.boxplot(x=s)
            plt.title(f"Boxplot {coluna} (IQR)")
            plt.grid(True, axis='x', linestyle='--', alpha=0.4)
            plt.show()
        msg = (
            f"Outliers (IQR) em '{coluna}': {n_out}/{n} = {pct:.3f}% (limites: [{low:.4f}, {high:.4f}])."
        )
        self.memoria_analises.append(msg)
        self.ultima_coluna = coluna
        return msg

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
        n_out = int(mask.sum())
        n = int(s.shape[0])
        pct = (n_out / n * 100) if n else 0.0
        if plot:
            plt.figure(figsize=(10, 6))
            sns.histplot(z, kde=True, stat="density", linewidth=0)
            plt.title(f"Distribuição de Z-scores - {coluna}")
            plt.xlabel("Z-score")
            plt.grid(True)
            plt.show()
        msg = (
            f"Outliers (Z>|{threshold}|) em '{coluna}': {n_out}/{n} = {pct:.3f}% (média={mu:.4f}, desvio={sigma:.4f})."
        )
        self.memoria_analises.append(msg)
        self.ultima_coluna = coluna
        return msg

    def detectar_outliers_isolation_forest(
        self, colunas: Optional[str] = None, contamination: float = 0.01
    ) -> str:
        try:
            from sklearn.preprocessing import StandardScaler
            from sklearn.ensemble import IsolationForest
        except Exception:
            return "Isolation Forest requer scikit-learn. Instale com: pip install scikit-learn"

        if colunas:
            cols = [c.strip() for c in colunas.split(",") if c.strip()]
        else:
            cols = self.df.select_dtypes(include="number").columns.tolist()

        if not cols:
            return "Não há colunas numéricas suficientes."
        X = self.df[cols].dropna()
        if X.empty:
            return "Após remover NAs, não sobraram linhas."

        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)

        used_contamination = max(1e-4, min(contamination, 0.5))
        clf = IsolationForest(contamination=used_contamination, random_state=42)
        labels = clf.fit_predict(Xs)  # -1 = outlier
        n_out = int((labels == -1).sum())
        n = int(len(labels))
        pct = (n_out / n * 100) if n else 0.0
        msg = (
            f"Isolation Forest: {n_out}/{n} = {pct:.3f}% de outliers nas colunas {cols} (contamination={used_contamination})."
        )
        self.memoria_analises.append(msg)
        return msg

    def resumo_outliers_dataset(self, method: str = "iqr", top_k: int = 10) -> str:
        df_num = self.df.select_dtypes(include="number")
        if df_num.empty:
            return "Não há colunas numéricas para sumarizar."
        linhas = []
        for col in df_num.columns:
            s = df_num[col].dropna()
            n = int(s.shape[0]) if s is not None else 0
            if n == 0:
                linhas.append((col, 0.0, 0, 0))
                continue
            if method.lower() == "zscore":
                mu, sigma = s.mean(), s.std(ddof=0)
                if sigma == 0 or pd.isna(sigma):
                    pct, cnt = 0.0, 0
                else:
                    z = (s - mu) / sigma
                    mask = z.abs() > 3.0
                    cnt = int(mask.sum())
                    pct = (cnt / n * 100)
            else:
                q1, q3 = s.quantile(0.25), s.quantile(0.75)
                iqr = q3 - q1
                low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                mask = (s < low) | (s > high)
                cnt = int(mask.sum())
                pct = (cnt / n * 100)
            linhas.append((col, pct, cnt, n))
        linhas.sort(key=lambda x: x[1], reverse=True)
        top = linhas[: max(1, top_k)]
        media_pct = sum(p for _, p, _, _ in linhas) / len(linhas)
        partes = [f"Resumo de outliers por {method.upper()} (top {len(top)} colunas):"]
        for col, pct, cnt, n in top:
            partes.append(f"- {col}: {cnt}/{n} = {pct:.3f}%")
        partes.append(f"Média de % de outliers nas colunas numéricas: {media_pct:.3f}%")
        self.memoria_analises.append("Resumo de outliers calculado.")
        return "\n".join(partes)

    # ----------------- RELAÇÕES -----------------
    def plotar_dispersao(
        self, x: str, y: str, hue: Optional[str] = None, amostra: int = 5000
    ) -> str:
        """Gráfico de dispersão entre duas colunas numéricas; 'hue' opcional."""
        for col in [x, y] + ([hue] if hue else []):
            if col and col not in self.df.columns:
                return f"Erro: a coluna '{col}' não existe."
        df_plot = self.df[[x, y] + ([hue] if hue else [])].dropna()
        for col in [x, y]:
            if not pd.api.types.is_numeric_dtype(df_plot[col]):
                df_plot[col] = pd.to_numeric(df_plot[col], errors="coerce")
        df_plot = df_plot.dropna(subset=[x, y])
        if df_plot.empty:
            return "Não há dados válidos após limpeza para o gráfico de dispersão."
        if len(df_plot) > amostra:
            df_plot = df_plot.sample(n=amostra, random_state=42)
        plt.figure(figsize=(8, 6))
        sns.scatterplot(
            data=df_plot, x=x, y=y, hue=hue if hue else None, s=20, alpha=0.7, edgecolor=None
        )
        plt.title(f"Dispersão: {x} vs {y}" + (f" (hue={hue})" if hue else ""))
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.show()
        msg = f"Gráfico de dispersão {x} vs {y}" + (f" com hue={hue}." if hue else ".")
        self.memoria_analises.append(msg)
        self.ultima_coluna = y
        return msg

    def matriz_dispersao(
        self,
        colunas: Optional[str] = None,
        hue: Optional[str] = None,
        amostra: int = 3000,
        corner: bool = True,
    ) -> str:
        """Pairplot para múltiplas colunas (amostrado)."""
        if colunas:
            cols = [c.strip() for c in colunas.split(",") if c.strip()]
        else:
            df_num = self.df.select_dtypes(include="number")
            if df_num.shape[1] == 0:
                return "Não há colunas numéricas para matriz de dispersão."
            vari = df_num.var(numeric_only=True).sort_values(ascending=False)
            cols = vari.index.tolist()[:6]
        for c in cols + ([hue] if hue else []):
            if c and c not in self.df.columns:
                return f"Erro: a coluna '{c}' não existe."
        use_cols = cols + ([hue] if hue else [])
        df_plot = self.df[use_cols].dropna()
        if len(df_plot) > amostra:
            df_plot = df_plot.sample(n=amostra, random_state=42)
        if len(cols) < 2:
            return "Selecione pelo menos 2 colunas para a matriz de dispersão."
        g = sns.pairplot(
            df_plot, vars=cols, hue=hue, corner=corner, diag_kind="hist", plot_kws=dict(s=15, alpha=0.7)
        )
        g.fig.suptitle("Matriz de Dispersão (amostrada)", y=1.02)
        plt.show()
        msg = f"Matriz de dispersão gerada para colunas: {cols}" + (f" (hue={hue})." if hue else ".")
        self.memoria_analises.append(msg)
        return msg

    def tabela_cruzada(
        self,
        linhas: str,
        colunas: str,
        normalizar: bool = True,
        normalizar_modo: Optional[str] = "all",
        top_k: int = 20,
        heatmap: bool = True,
        annot: bool = False,
    ) -> str:
        """Crosstab entre duas colunas categóricas/discretas (+ heatmap opcional)."""
        for col in [linhas, colunas]:
            if col not in self.df.columns:
                return f"Erro: a coluna '{col}' não existe."
        s_l = self.df[linhas].astype(str)
        s_c = self.df[colunas].astype(str)
        top_l = s_l.value_counts().index[:top_k]
        top_c = s_c.value_counts().index[:top_k]
        df_small = self.df[s_l.isin(top_l) & s_c.isin(top_c)]
        ct = pd.crosstab(df_small[linhas].astype(str), df_small[colunas].astype(str))

        if normalizar:
            mode = (normalizar_modo or "all").lower()
            if mode == "index":
                tabela = ct.div(ct.sum(axis=1), axis=0).fillna(0.0)
            elif mode == "columns":
                tabela = ct.div(ct.sum(axis=0), axis=1).fillna(0.0)
            else:
                tabela = (ct / ct.values.sum()).fillna(0.0)
        else:
            tabela = ct

        if heatmap:
            plt.figure(
                figsize=(max(6, 0.4 * len(tabela.columns)), max(4, 0.4 * len(tabela.index)))
            )
            sns.heatmap(tabela, cmap="Blues", annot=annot, fmt=".2f" if normalizar else "d")
            titulo_norm = " (normalizada " + (mode if normalizar else "") + ")" if normalizar else ""
            plt.title(f"Tabela Cruzada: {linhas} x {colunas}{titulo_norm}")
            plt.xlabel(colunas)
            plt.ylabel(linhas)
            plt.tight_layout()
            plt.show()
        msg = (
            f"Tabela cruzada gerada para {linhas} x {colunas} (normalizada={normalizar}, modo={normalizar_modo}, top_k={top_k})."
        )
        self.memoria_analises.append(msg)
        self.ultima_coluna = colunas
        return msg

    # ----------------- Tempo -----------------
    def converter_time_para_datetime(
        self,
        origem: Optional[str] = None,
        unidade: str = "s",
        nova_coluna: Optional[str] = None,
        criar_features: bool = True,
    ) -> str:
        """Converte a coluna 'Time' (offset) para datetime relativo/ancorado e cria features temporais."""
        col = "Time"
        if col not in self.df.columns:
            return "Erro: coluna 'Time' não encontrada no dataset."

        s = pd.to_numeric(self.df[col], errors="coerce")
        if s.isna().all():
            return "Erro: não foi possível interpretar 'Time' como numérico."

        try:
            td = pd.to_timedelta(s, unit=unidade)
        except Exception as e:
            return f"Erro ao converter para Timedelta com unidade='{unidade}': {e}"

        created_cols = []
        if origem:
            try:
                base = pd.to_datetime(origem)
            except Exception as e:
                return f"Erro ao interpretar 'origem': {e}"
            target_col = nova_coluna or "Time_dt"
            self.df[target_col] = base + td
            created_cols.append(target_col)
            self.ultima_coluna = target_col
            modo = f"ancorado em '{origem}'"
        else:
            target_col = nova_coluna or "Time_delta"
            self.df[target_col] = td
            created_cols.append(target_col)
            self.ultima_coluna = target_col
            modo = "relativo (Timedelta)"

        if criar_features:
            seconds_total = td.dt.total_seconds()
            self.df["Time_hour"] = ((seconds_total // 3600) % 24).astype(int)
            self.df["Time_day"] = (seconds_total // 86400).astype(int)
            bins = (seconds_total // 3600).astype(int)
            self.df["Time_bin_1h"] = bins.astype(str) + "h-" + (bins + 1).astype(str) + "h"
            created_cols += ["Time_hour", "Time_day", "Time_bin_1h"]

        msg = (
            f"Conversão de 'Time' concluída ({modo}). Colunas criadas: {', '.join(created_cols)}."
        )
        try:
            sample = self.df[created_cols].head(3).to_string()
            msg += f"\nExemplo de linhas:\n{sample}"
        except Exception:
            pass

        self.memoria_analises.append(msg)
        return msg

    # ----------------- NOVA: Tendências temporais -----------------
    def tendencias_temporais(
        self,
        coluna_valor: str,
        freq: str = "D",
        agg: str = "sum",
        timestamp_col: Optional[str] = None,
        origem: Optional[str] = None,
        unidade: str = "s",
        rolling: Optional[int] = None,
    ) -> str:
        """
        Reamostra por tempo e plota linha da série agregada.
        - 'freq': 'H' (hora), 'D' (dia), 'W' (semana) ou 'M' (mês)
        - 'agg': 'sum', 'mean', 'median', 'count', 'max', 'min'
        - Se não houver coluna datetime, tenta usar 'Time_dt'; caso não exista, converte 'Time'.
        - 'rolling': janela para média móvel (ex.: 7 para 7 dias se freq='D').
        """
        if coluna_valor not in self.df.columns:
            return f"Erro: a coluna '{coluna_valor}' não existe."
        if not pd.api.types.is_numeric_dtype(self.df[coluna_valor]):
            return f"A coluna '{coluna_valor}' precisa ser numérica para agregação."

        ts_col = None
        if timestamp_col and timestamp_col in self.df.columns:
            # Garante datetime
            if not pd.api.types.is_datetime64_any_dtype(self.df[timestamp_col]):
                try:
                    self.df[timestamp_col] = pd.to_datetime(self.df[timestamp_col])
                except Exception as e:
                    return f"Não foi possível converter '{timestamp_col}' para datetime: {e}"
            ts_col = timestamp_col
        elif "Time_dt" in self.df.columns:
            ts_col = "Time_dt"
        elif "Time" in self.df.columns:
            # tenta converter 'Time' para datetime ancorado
            origem_padrao = origem or "1970-01-01 00:00:00"
            conv_msg = self.converter_time_para_datetime(origem=origem_padrao, unidade=unidade)
            self.memoria_analises.append(f"[Auto] {conv_msg}")
            if "Time_dt" in self.df.columns:
                ts_col = "Time_dt"
            else:
                return (
                    "Não foi possível criar 'Time_dt'. Forneça 'timestamp_col' ou use 'converter_time_para_datetime' antes."
                )
        else:
            return (
                "Não há coluna temporal. Informe 'timestamp_col' ou inclua 'Time'/'Time_dt' no dataset."
            )

        # Série
        df_ts = self.df[[ts_col, coluna_valor]].dropna()
        if df_ts.empty:
            return "Sem dados válidos após limpeza para séries temporais."
        df_ts = df_ts.sort_values(ts_col).set_index(ts_col)

        if agg not in {"sum", "mean", "median", "count", "max", "min"}:
            return "Parâmetro 'agg' inválido. Use: sum, mean, median, count, max, min."

        series = getattr(df_ts[coluna_valor].resample(freq), agg)()
        if series.empty:
            return "Reamostragem resultou em série vazia. Verifique 'freq' e o intervalo temporal."

        plt.figure(figsize=(10, 5))
        plt.plot(series.index, series.values, label=f"{agg}({coluna_valor})")

        if rolling and isinstance(rolling, int) and rolling > 1:
            roll = series.rolling(rolling, min_periods=1).mean()
            plt.plot(roll.index, roll.values, linestyle='--', label=f"rolling({rolling})")

        plt.title(
            f"Tendência temporal de '{coluna_valor}' por {freq} (agg={agg})"
        )
        plt.xlabel("Tempo")
        plt.ylabel(coluna_valor)
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

        msg = (
            f"Tendência temporal gerada para '{coluna_valor}' com freq='{freq}', agg='{agg}'"
            + (f", rolling={rolling}" if rolling else "")
            + f" usando '{ts_col}' como eixo temporal."
        )
        self.memoria_analises.append(msg)
        return msg

    # ----------------- SUPORTE / MEMÓRIA -----------------
    def mostrar_conclusoes(self) -> str:
        if not self.memoria_analises:
            return "Nenhuma análise foi realizada ainda."
        return "\n--- Conclusões Baseadas nas Análises Realizadas ---\n" + "\n".join(
            self.memoria_analises
        )

    # -------------------------
    # Pré-processamento leve para inferir intenção/coluna
    # -------------------------
    def _preprocessar_pergunta(self, pergunta: str) -> str:
        t = pergunta.strip()
        if t in self.df.columns:
            self.ultima_coluna = t
            return f"Use a coluna '{t}' como foco. Gere um histograma e calcule outliers por IQR."
        low = t.lower()
        if any(k in low for k in ["histogram", "histograma"]):
            if self.ultima_coluna:
                return (
                    f"Plote um histograma da coluna '{self.ultima_coluna}' e descreva a distribuição."
                )
        if any(k in low for k in ["frequenc", "frequência", "frequencias", "frequências"]):
            if self.ultima_coluna:
                return (
                    f"Mostre as frequências (top 10 e bottom 10) da coluna '{self.ultima_coluna}'."
                )
        if "moda" in low and self.ultima_coluna:
            return f"Calcule a moda da coluna '{self.ultima_coluna}'."
        if any(k in low for k in ["outlier", "atípic"]) and self.ultima_coluna:
            return (
                f"Detecte outliers por IQR na coluna '{self.ultima_coluna}' e mostre o %."
            )
        if any(k in low for k in ["tendência", "tendencias", "tendências", "temporal", "tempo"]):
            # Sugerir tendência com Amount se existir
            if "Amount" in self.df.columns:
                return (
                    "Gere tendências temporais agregando 'Amount' por dia e descreva o comportamento."
                )
        if "dispers" in low and self.ultima_coluna:
            if "Amount" in self.df.columns and self.ultima_coluna.lower() != "amount":
                return (
                    f"Plote um gráfico de dispersão entre 'Amount' e '{self.ultima_coluna}'."
                )
        if "converter time" in low or ("time" in low and "datetime" in low):
            return (
                "Converta 'Time' para datetime relativo (unidade segundos) e crie features."
            )
        return pergunta

    # -------------------------
    # Loop de chat (CLI)
    # -------------------------
    def iniciar_chat(self):
        print("\nOlá! Sou seu agente de análise de dados. Faça uma pergunta sobre o arquivo CSV.")
        while True:
            try:
                pergunta = input("\nSua pergunta (ou digite 'sair'): ")
            except (EOFError, KeyboardInterrupt):
                print("\nEncerrando o agente. Até mais!")
                break

            if not pergunta:
                continue
            if pergunta.lower() in ["sair", "exit", "quit"]:
                print("Encerrando o agente. Até mais!")
                break

            pergunta_proc = self._preprocessar_pergunta(pergunta)
            try:
                resposta = self.agent.invoke(
                    {"input": pergunta_proc},
                    config={
                        "configurable": {"session_id": self.session_id},
                        "tags": self.run_tags,
                        "metadata": self.run_metadata,
                    },
                )
                print("\n--- Resposta do Agente ---")
                print(resposta.get("output", resposta))
                print("-------------------------\n")
            except Exception as e:
                print(f"Ocorreu um erro ao processar sua pergunta: {e}")


# -------------------------------------------------------------
# UI Web (Streamlit)
# -------------------------------------------------------------
def streamlit_app():
    import streamlit as st

    st.set_page_config(page_title="Agente EDA - CSV dinâmico", layout="wide")
    st.title("Agente EDA (LangChain + Gemini + LangSmith)")
    st.caption("Envie um CSV e faça perguntas. Use a ferramenta de **tendências temporais** para padrões no tempo.")

    # Upload de CSV
    uploaded = st.file_uploader("Envie um arquivo CSV", type=["csv"])

    # Sessão: guarda caminho temporário e agente
    if "agente" not in st.session_state:
        st.session_state.agente = None
        st.session_state.csv_path = None

    col1, col2 = st.columns([2, 1])

    with col1:
        if uploaded is not None:
            # Salva em arquivo temporário
            fd, tmp_path = tempfile.mkstemp(suffix=".csv")
            with os.fdopen(fd, "wb") as f:
                f.write(uploaded.getbuffer())
            st.session_state.csv_path = tmp_path
            try:
                st.session_state.agente = AgenteDeAnalise(
                    caminho_arquivo_csv=tmp_path, session_id="ui_streamlit"
                )
                st.success("CSV carregado com sucesso!")
                st.write("**Colunas:**", ", ".join(st.session_state.agente.df.columns.tolist()))
                st.dataframe(st.session_state.agente.df.head())
            except Exception as e:
                st.error(str(e))
        else:
            st.info("Envie um CSV para iniciar.")

        if st.session_state.agente is not None:
            st.subheader("Pergunte ao agente")
            pergunta = st.text_input(
                "Digite sua pergunta (ex.: 'Quais são as correlações?', 'Gerar tendências temporais de Amount por dia')"
            )
            if st.button("Perguntar", type="primary") and pergunta:
                proc = st.session_state.agente._preprocessar_pergunta(pergunta)
                try:
                    resposta = st.session_state.agente.agent.invoke(
                        {"input": proc},
                        config={
                            "configurable": {"session_id": "ui_streamlit"},
                            "tags": ["ui", "streamlit"],
                            "metadata": {"origin": "ui"},
                        },
                    )
                    st.markdown("### Resposta do agente")
                    st.write(resposta.get("output", resposta))
                except Exception as e:
                    st.error(str(e))

            st.markdown("---")
            st.subheader("Tendências temporais (execução direta da ferramenta)")
            df = st.session_state.agente.df
            num_cols = df.select_dtypes(include="number").columns.tolist()
            valor_col = st.selectbox("Coluna numérica para agregar", options=num_cols, index= num_cols.index("Amount") if "Amount" in num_cols else 0)
            freq = st.selectbox("Frequência", options=["H", "D", "W", "M"], index=1)
            agg = st.selectbox("Agregação", options=["sum", "mean", "median", "count", "max", "min"], index=0)
            timestamp_col = st.selectbox(
                "Coluna de tempo (opcional)", options=["(auto)"] + df.columns.tolist(), index=0
            )
            origem = st.text_input(
                "Origem para converter 'Time' (opcional, se necessário)", value=""
            )
            rolling = st.number_input(
                "Janela da média móvel (opcional)", min_value=0, value=0, step=1
            )
            if st.button("Gerar tendência"):
                ts_col = None if timestamp_col == "(auto)" else timestamp_col
                res = st.session_state.agente.tendencias_temporais(
                    coluna_valor=valor_col,
                    freq=freq,
                    agg=agg,
                    timestamp_col=ts_col,
                    origem=origem or None,
                    unidade="s",
                    rolling=(rolling if rolling and rolling > 1 else None),
                )
                st.success(res)

    with col2:
        st.markdown("### Conclusões acumuladas")
        if st.session_state.agente is not None:
            st.text(st.session_state.agente.mostrar_conclusoes())
        else:
            st.write("Sem análises ainda.")


# -------------------------------------------------------------
# Execução (CLI ou UI)
# -------------------------------------------------------------
if __name__ == "__main__":
    if _is_streamlit_runtime():
        # Executando via: streamlit run agente_eda.py
        streamlit_app()
    else:
        parser = argparse.ArgumentParser(description="Agente EDA - CSV dinâmico")
        parser.add_argument(
            "--csv",
            dest="csv_path",
            type=str,
            required=True,
            help="Caminho para o arquivo CSV de entrada.",
        )
        parser.add_argument(
            "--session",
            dest="session_id",
            type=str,
            default="minha_sessao",
            help="ID da sessão para memória do chat.",
        )
        args = parser.parse_args()

        try:
            agente = AgenteDeAnalise(
                caminho_arquivo_csv=args.csv_path,
                session_id=args.session_id,
                langsmith_project="EDA-Agent",
                run_tags=["eda", "cli"],
            )
            agente.iniciar_chat()
        except (FileNotFoundError, ValueError) as e:
            print(e)
        except Exception as e:
            print(f"Ocorreu um erro inesperado ao inicializar o agente: {e}")
