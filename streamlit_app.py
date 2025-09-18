# -*- coding: utf-8 -*-
"""
Agente EDA com LangChain + LangSmith + UI Web (Streamlit)
- Entrada dinâmica de CSV via upload
- Memória (RunnableWithMessageHistory)
- StructuredTool corrigido
- Ferramentas:
  * listar_colunas, descricao_geral_dados, estatisticas_descritivas
  * plotar_histograma, plotar_mapa_correlacao
  * frequencias_coluna, moda_coluna
  * kmeans_clusterizar
  * detectar_outliers_iqr, detectar_outliers_zscore, detectar_outliers_isolation_forest
  * resumo_outliers_dataset
  * plotar_dispersao, matriz_dispersao, tabela_cruzada
  * converter_time_para_datetime
  * tendencias_temporais (nova)
- UI com Streamlit para upload e interação
"""

import os
import io
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

from langsmith import Client

load_dotenv()


class AgenteDeAnalise:
    def __init__(self, caminho_arquivo_csv: str, session_id: str = "default"):
        google_api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            raise ValueError("Defina GEMINI_API_KEY ou GOOGLE_API_KEY no .env/ambiente.")

        if not os.path.exists(caminho_arquivo_csv):
            raise FileNotFoundError(f"Erro: O arquivo '{caminho_arquivo_csv}' não foi encontrado.")
        self.df = pd.read_csv(caminho_arquivo_csv)
        self.memoria_analises: List[str] = []
        self.ultima_coluna: Optional[str] = None
        self.session_id = session_id

        # LangSmith
        self.client = Client()

        # LLM
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0,
            google_api_key=google_api_key,
        )

        # Ferramentas
        tools = self._definir_ferramentas()

        # Prompt
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system",
                 "Você é um assistente de análise de dados. Use as ferramentas quando necessário.\n"
                 "- Se o usuário pedir relações entre variáveis, ofereça 'plotar_mapa_correlacao', 'plotar_dispersao' e 'matriz_dispersao'.\n"
                 "- Para 'tabela cruzada', use 'tabela_cruzada'.\n"
                 "- Para 'valores mais/menos frequentes', use 'frequencias_coluna'.\n"
                 "- Para 'outliers' no dataset, use 'resumo_outliers_dataset'; para coluna específica, IQR/Z-score.\n"
                 "- Se o usuário omitir a coluna (ex.: 'histograma', 'frequências', 'moda', 'outliers'), use a última coluna mencionada.\n"
                 "- Se o usuário pedir para converter 'Time' em datetime, use 'converter_time_para_datetime'.\n"
                 "- Para tendências ao longo do tempo, use 'tendencias_temporais'.\n"
                 "- Só pergunte de volta se não houver coluna clara no histórico.\n"
                 "- Quando gerar gráfico, explique brevemente o que ele mostra."),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
                MessagesPlaceholder("agent_scratchpad"),
            ]
        )

        # Agente
        base_agent = create_tool_calling_agent(self.llm, tools, prompt)
        self.base_executor = AgentExecutor(
            agent=base_agent,
            tools=tools,
            verbose=False,
            max_iterations=5,
            handle_parsing_errors=True,
        )

        # Memória
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

    def _definir_ferramentas(self):
        class HistogramaInput(BaseModel):
            coluna: str = Field(description="Coluna numérica para histograma.")

        class FrequenciasInput(BaseModel):
            coluna: str = Field(description="Coluna para calcular frequências.")
            top_n: int = Field(default=10)
            bottom_n: int = Field(default=10)

        class ModaInput(BaseModel):
            coluna: str = Field(description="Coluna para calcular moda.")

        class DispersaoInput(BaseModel):
            x: str = Field(description="Coluna X (numérica).")
            y: str = Field(description="Coluna Y (numérica).")
            hue: Optional[str] = Field(default=None)

        class PairplotInput(BaseModel):
            colunas: Optional[str] = Field(default=None)
            hue: Optional[str] = Field(default=None)

        class CrosstabInput(BaseModel):
            linhas: str = Field(description="Coluna para linhas (categórica).")
            colunas: str = Field(description="Coluna para colunas (categórica).")

        class OutlierIQRInput(BaseModel):
            coluna: str = Field(description="Coluna numérica.")
            plot: bool = Field(default=False)

        class OutlierZInput(BaseModel):
            coluna: str = Field(description="Coluna numérica.")
            threshold: float = Field(default=3.0)
            plot: bool = Field(default=False)

        class TimeConvertInput(BaseModel):
            origem: Optional[str] = Field(default=None)
            unidade: str = Field(default="s")
            nova_coluna: Optional[str] = Field(default=None)
            criar_features: bool = Field(default=True)

        class TendenciasInput(BaseModel):
            coluna: str = Field(description="Coluna numérica (ex.: Amount).")
            freq: str = Field(default="D")

        return [
            StructuredTool.from_function(func=self.listar_colunas, name="listar_colunas",
                                         description="Lista as colunas do dataset."),
            StructuredTool.from_function(func=self.obter_descricao_geral, name="descricao_geral_dados",
                                         description="Resumo geral do dataset."),
            StructuredTool.from_function(func=self.obter_estatisticas_descritivas, name="estatisticas_descritivas",
                                         description="Estatísticas descritivas."),
            StructuredTool.from_function(func=self.plotar_histograma, name="plotar_histograma",
                                         description="Histograma de coluna.", args_schema=HistogramaInput),
            StructuredTool.from_function(func=self.mostrar_correlacao, name="plotar_mapa_correlacao",
                                         description="Mapa de calor de correlação."),
            StructuredTool.from_function(func=self.frequencias_coluna, name="frequencias_coluna",
                                         description="Frequências top/bottom.", args_schema=FrequenciasInput),
            StructuredTool.from_function(func=self.moda_coluna, name="moda_coluna",
                                         description="Moda(s) de uma coluna.", args_schema=ModaInput),
            StructuredTool.from_function(func=self.plotar_dispersao, name="plotar_dispersao",
                                         description="Dispersão X vs Y.", args_schema=DispersaoInput),
            StructuredTool.from_function(func=self.matriz_dispersao, name="matriz_dispersao",
                                         description="Pairplot.", args_schema=PairplotInput),
            StructuredTool.from_function(func=self.tabela_cruzada, name="tabela_cruzada",
                                         description="Crosstab entre duas colunas.", args_schema=CrosstabInput),
            StructuredTool.from_function(func=self.detectar_outliers_iqr, name="detectar_outliers_iqr",
                                         description="Outliers por IQR.", args_schema=OutlierIQRInput),
            StructuredTool.from_function(func=self.detectar_outliers_zscore, name="detectar_outliers_zscore",
                                         description="Outliers por Z-score.", args_schema=OutlierZInput),
            StructuredTool.from_function(func=self.converter_time_para_datetime, name="converter_time_para_datetime",
                                         description="Converte 'Time' para datetime.", args_schema=TimeConvertInput),
            StructuredTool.from_function(func=self.tendencias_temporais, name="tendencias_temporais",
                                         description="Tendências temporais.", args_schema=TendenciasInput),
            StructuredTool.from_function(func=self.mostrar_conclusoes, name="mostrar_conclusoes",
                                         description="Mostra memória de análises."),
        ]

    # ---------- Ferramentas ----------
    def listar_colunas(self) -> str:
        return f"Colunas: {', '.join(self.df.columns)}"

    def obter_descricao_geral(self) -> str:
        buffer = io.StringIO()
        self.df.info(buf=buffer)
        return buffer.getvalue()

    def obter_estatisticas_descritivas(self) -> str:
        return self.df.describe().to_string()

    def plotar_histograma(self, coluna: str) -> str:
        if coluna not in self.df.columns:
            return f"Coluna '{coluna}' não encontrada."
        plt.figure(figsize=(8, 5))
        sns.histplot(self.df[coluna], kde=True)
        plt.title(f"Histograma - {coluna}")
        st.pyplot(plt)
        return f"Histograma da coluna '{coluna}' gerado."

    def mostrar_correlacao(self) -> str:
        df_num = self.df.select_dtypes(include="number")
        if df_num.empty:
            return "Nenhuma coluna numérica encontrada."
        plt.figure(figsize=(10, 7))
        sns.heatmap(df_num.corr(), cmap="coolwarm")
        st.pyplot(plt)
        return "Mapa de calor de correlação gerado."

    def frequencias_coluna(self, coluna: str, top_n: int = 10, bottom_n: int = 10) -> str:
        if coluna not in self.df.columns:
            return f"Coluna '{coluna}' não encontrada."
        cont = self.df[coluna].value_counts()
        return f"Top {top_n}:\n{cont.head(top_n)}\n\nBottom {bottom_n}:\n{cont.tail(bottom_n)}"

    def moda_coluna(self, coluna: str) -> str:
        modos = self.df[coluna].mode()
        return f"Moda(s) da coluna {coluna}: {modos.tolist()}"

    def plotar_dispersao(self, x: str, y: str, hue: Optional[str] = None) -> str:
        if x not in self.df.columns or y not in self.df.columns:
            return "Coluna não encontrada."
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=self.df, x=x, y=y, hue=hue)
        st.pyplot(plt)
        return f"Dispersão {x} vs {y} plotada."

    def matriz_dispersao(self, colunas: Optional[str] = None, hue: Optional[str] = None) -> str:
        cols = colunas.split(",") if colunas else self.df.select_dtypes(include="number").columns[:4]
        sns.pairplot(self.df[cols], hue=hue)
        st.pyplot(plt)
        return f"Matriz de dispersão gerada ({cols})."

    def tabela_cruzada(self, linhas: str, colunas: str) -> str:
        if linhas not in self.df.columns or colunas not in self.df.columns:
            return "Coluna não encontrada."
        ct = pd.crosstab(self.df[linhas], self.df[colunas])
        sns.heatmap(ct, cmap="Blues")
        st.pyplot(plt)
        return "Tabela cruzada plotada."

    def detectar_outliers_iqr(self, coluna: str, plot: bool = False) -> str:
        s = self.df[coluna].dropna()
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        low, high = q1 - 1.5*iqr, q3 + 1.5*iqr
        outliers = ((s < low) | (s > high)).sum()
        return f"Outliers em {coluna}: {outliers}"

    def detectar_outliers_zscore(self, coluna: str, threshold: float = 3.0, plot: bool = False) -> str:
        s = self.df[coluna].dropna()
        z = (s - s.mean()) / s.std()
        outliers = (z.abs() > threshold).sum()
        return f"Outliers Z-score em {coluna}: {outliers}"

    def converter_time_para_datetime(self, origem: Optional[str] = None, unidade: str = "s",
                                     nova_coluna: Optional[str] = None, criar_features: bool = True) -> str:
        s = pd.to_numeric(self.df["Time"], errors="coerce")
        td = pd.to_timedelta(s, unit=unidade)
        base = pd.to_datetime(origem) if origem else pd.Timestamp("1970-01-01")
        target_col = nova_coluna or "Time_dt"
        self.df[target_col] = base + td
        return f"Coluna '{target_col}' criada."

    def tendencias_temporais(self, coluna: str, freq: str = "D") -> str:
        if "Time" not in self.df.columns:
            return "Coluna 'Time' não encontrada."
        s = pd.to_numeric(self.df["Time"], errors="coerce")
        td = pd.to_timedelta(s, unit="s")
        base = pd.Timestamp("1970-01-01")
        self.df["Time_dt"] = base + td
        ts = self.df.set_index("Time_dt")[coluna].resample(freq).mean()
        ts.plot(figsize=(10, 5))
        st.pyplot(plt)
        return f"Tendência temporal da coluna '{coluna}' plotada."

    def mostrar_conclusoes(self) -> str:
        return "\n".join(self.memoria_analises) if self.memoria_analises else "Nenhuma análise registrada."


# -------------------------
# UI Streamlit
# -------------------------
st.title("📊 Agente de Análise de Dados")

uploaded_file = st.file_uploader("Carregue um CSV", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    tmp_path = "uploaded.csv"
    df.to_csv(tmp_path, index=False)
    agente = AgenteDeAnalise(caminho_arquivo_csv=tmp_path, session_id="ui")
    pergunta = st.text_input("Faça uma pergunta")
    if st.button("Perguntar") and pergunta:   # <<< sem type="primary"
        resposta = agente.agent.invoke(
            {"input": pergunta},
            config={"configurable": {"session_id": "ui"}},
        )
        st.write(resposta.get("output", resposta))
