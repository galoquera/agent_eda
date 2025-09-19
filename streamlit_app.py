# -*- coding: utf-8 -*-
"""
Agente EDA (Streamlit) — LangChain + Gemini (+ LangSmith opcional)
- Upload de CSV genérico (com persistência de arquivo)
- Pergunta/resposta com ferramentas (gráficos no Streamlit)
- Memória interna (conclusões), exibida apenas quando o usuário pedir
- Tools: descrição, histogramas (1 e múltiplos), frequências, moda,
  correlação, dispersão, crosstab, outliers (IQR/Z-score + resumo dataset),
  tendências temporais, k-means.
"""

# --- REQUISITOS ---
# Para executar este agente, certifique-se de ter as seguintes bibliotecas instaladas:
# pip install streamlit pandas matplotlib seaborn python-dotenv langchain langchain-google-genai pydantic scikit-learn langsmith

import os
import io
import tempfile
from typing import List, Optional

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.tools import StructuredTool
from pydantic import BaseModel, Field
# langchain.hub não é mais necessário

# ---------- LangSmith (opcional) ----------
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
    def __init__(self, caminho_arquivo_csv: str, chat_history_store: dict, session_id: str = "ui_streamlit"):
        google_api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            raise ValueError("Defina GEMINI_API_KEY ou GOOGLE_API_KEY nas Secrets do Streamlit.")

        if not os.path.exists(caminho_arquivo_csv):
            raise FileNotFoundError(f"Arquivo '{caminho_arquivo_csv}' não encontrado.")
        self.df = pd.read_csv(caminho_arquivo_csv)
        self.memoria_analises: List[str] = []
        self.ultima_coluna: str | None = None
        self.session_id = session_id

        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0,
            google_api_key=google_api_key,
        )

        tools = self._definir_ferramentas()

        # --- PROMPT DEFINIDO LOCALMENTE PARA REMOVER DEPENDÊNCIA DO HUB ---
        # Recriação da estrutura do prompt 'hwchase17/react-chat'
        template = """
Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Previous conversation history:
{chat_history}

Question: {input}
Thought:{agent_scratchpad}
"""
        prompt = PromptTemplate.from_template(template)

        base_agent = create_react_agent(self.llm, tools, prompt)
        self.base_executor = AgentExecutor(
            agent=base_agent,
            tools=tools,
            verbose=False,
            max_iterations=5,
            handle_parsing_errors=True,
        )

        def get_session_history(session_id: str):
            if session_id not in chat_history_store:
                chat_history_store[session_id] = ChatMessageHistory()
            return chat_history_store[session_id]

        self.agent = RunnableWithMessageHistory(
            self.base_executor, get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
        )

    # ---- Helper de memória de conclusões ----
    def _lembrar(self, chave: str, texto: str):
        """Guarda conclusão rica, evitando duplicatas por 'chave'."""
        tag = f"[{chave}] "
        item = tag + texto.strip()
        if item not in self.memoria_analises:
            self.memoria_analises.append(item)

    # ------------------------ Tools ------------------------
    def _definir_ferramentas(self):
        class HistogramaInput(BaseModel):
            coluna: str = Field(description="Coluna numérica para histograma.")

        class HistAllInput(BaseModel):
            colunas: str = Field(default="", description="Lista separada por vírgula (vazio = todas numéricas).")
            kde: bool = Field(default=True, description="Exibir curva KDE.")
            bins: int = Field(default=30, description="Número de bins.")
            cols_por_linha: int = Field(default=3, description="Gráficos por linha.")
            max_colunas: int = Field(default=12, description="Limite superior de colunas plotadas.")

        class FrequenciasInput(BaseModel):
            coluna: str = Field(description="Coluna para frequências top/bottom.")
            top_n: int = Field(default=10)
            bottom_n: int = Field(default=10)

        class ModaInput(BaseModel):
            coluna: str = Field(description="Coluna para calcular moda.")
            
        class CorrelacaoInput(BaseModel):
            method: str = Field(default="pearson", description="Método: 'pearson', 'spearman' ou 'kendall'.")

        class DispersaoInput(BaseModel):
            x: str = Field(description="Coluna X (numérica).")
            y: str = Field(description="Coluna Y (numérica).")
            hue: str = Field(default="", description="Coluna categórica (opcional).")
            amostra: int = Field(default=5000, description="Máximo de linhas amostradas para o gráfico.")
            
        class PairplotInput(BaseModel):
            colunas: str = Field(default="", description="Colunas separadas por vírgula. Se vazio, usa até 6 numéricas.")
            hue: str = Field(default="", description="Coluna categórica para colorir.")
            amostra: int = Field(default=3000, description="Máximo de linhas amostradas para o pairplot.")
            corner: bool = Field(default=True, description="Se True, mostra apenas metade inferior da matriz.")

        class CrosstabInput(BaseModel):
            linhas: str = Field(description="Coluna para linhas (categórica).")
            colunas: str = Field(description="Coluna para colunas (categórica).")
            normalizar: bool = Field(default=True)
            heatmap: bool = Field(default=True)
            annot: bool = Field(default=False, description="Anotar valores no heatmap (pode poluir).")
            top_k: int = Field(default=20, description="Limita categorias por eixo para evitar tabelas gigantes.")

        class OutlierIQRInput(BaseModel):
            coluna: str = Field(description="Coluna numérica.")
            plot: bool = Field(default=False)

        class OutlierZInput(BaseModel):
            coluna: str = Field(description="Coluna numérica.")
            threshold: float = Field(default=3.0)
            plot: bool = Field(default=False)
            
        class OutlierIFInput(BaseModel):
            colunas: str = Field(default="", description="Colunas separadas por vírgula. Se vazio, usa todas as numéricas.")
            contamination: float = Field(default=0.01, description="Proporção esperada de outliers (ex.: 0.01).")

        class ResumoOutInput(BaseModel):
            method: str = Field(default="iqr", description="iqr ou zscore")
            top_k: int = Field(default=10)

        class KMeansInput(BaseModel):
            colunas: str = Field(default="", description="Lista separada por vírgula (vazio = numéricas).")
            clusters: int = Field(default=3, description="k (mín. 2)")

        class TimeConvertInput(BaseModel):
            origem: str = Field(default="", description="YYYY-MM-DD HH:MM:SS (vazio = relativo).")
            unidade: str = Field(default="s", description="s/ms/m/h")
            nova_coluna: str = Field(default="", description="Nome da coluna criada (opcional).")
            criar_features: bool = Field(default=True)

        class TendenciasInput(BaseModel):
            coluna: str = Field(description="Coluna numérica (ex.: Amount).")
            freq: str = Field(default="D", description="H/D/W/M")

        return [
            StructuredTool.from_function(
                self.listar_colunas, name="listar_colunas",
                description="Retorna uma lista com os nomes exatos de todas as colunas disponíveis no dataset. Use esta ferramenta quando precisar saber os nomes das colunas para usar em outras ferramentas ou para responder perguntas como 'quais são as colunas do dataset?'."
            ),
            StructuredTool.from_function(
                self.obter_descricao_geral, name="descricao_geral_dados",
                description="Fornece um resumo geral da estrutura do dataset, incluindo o número de linhas e colunas, os tipos de dados de cada coluna (Dtype) e a contagem de valores não nulos. Essencial para uma primeira visão geral e para perguntas como 'me dê um resumo dos dados' ou 'quais colunas têm valores faltando?'."
            ),
            StructuredTool.from_function(
                self.obter_estatisticas_descritivas, name="estatisticas_descritivas",
                description="Calcula e exibe uma tabela completa de estatísticas descritivas (contagem, média, desvio padrão, mínimo, máximo e quartis) para todas as colunas numéricas. Use esta ferramenta para responder perguntas sobre medidas de tendência central, dispersão ou distribuição, como 'qual a média da coluna X?', 'mostre as estatísticas da coluna Y' ou 'descreva as variáveis numéricas'."
            ),
            StructuredTool.from_function(
                self.plotar_histograma, name="plotar_histograma",
                description="Gera e exibe um histograma para uma única coluna numérica, mostrando a distribuição de seus valores. É ideal para visualizar a forma da distribuição, identificar assimetria e picos. Use para perguntas como 'mostre a distribuição da coluna X' ou 'plote um histograma de Y'.",
                args_schema=HistogramaInput
            ),
            StructuredTool.from_function(
                self.plotar_histogramas_dataset, name="plotar_histogramas_dataset",
                description="Gera e exibe histogramas para múltiplas colunas numéricas (ou todas) de uma só vez. Use esta ferramenta para obter uma visão rápida da distribuição de várias variáveis ao mesmo tempo, respondendo a perguntas como 'mostre a distribuição de todas as variáveis' ou 'gere histogramas para as colunas A, B e C'.",
                args_schema=HistAllInput
            ),
            StructuredTool.from_function(
                self.frequencias_coluna, name="frequencias_coluna",
                description="Calcula e exibe as contagens de frequência dos valores em uma coluna. Para colunas categóricas, mostra os valores mais e menos comuns. Para colunas numéricas contínuas, agrupa os dados em faixas (bins) para análise. Use para perguntas como 'quais são os valores mais comuns na coluna X?' ou 'mostre a frequência de Y'.",
                args_schema=FrequenciasInput
            ),
            StructuredTool.from_function(
                self.moda_coluna, name="moda_coluna",
                description="Calcula e retorna o valor (ou valores) que aparece com mais frequência em uma coluna específica. Útil para identificar o resultado mais comum. Responde a perguntas como 'qual é a moda da coluna X?'.",
                args_schema=ModaInput
            ),
            StructuredTool.from_function(
                self.mostrar_correlacao, name="plotar_mapa_correlacao",
                description="Calcula a matriz de correlação entre todas as colunas numéricas e a exibe como um mapa de calor (heatmap). Use esta ferramenta para responder perguntas como 'quais variáveis estão mais correlacionadas?', 'existe uma relação linear entre X e Y?' ou 'mostre o mapa de correlação'.",
                args_schema=CorrelacaoInput
            ),
            StructuredTool.from_function(
                self.plotar_dispersao, name="plotar_dispersao",
                description="Cria um gráfico de dispersão (scatterplot) para visualizar a relação entre duas variáveis numéricas (X e Y). Opcionalmente, pode-se usar uma terceira variável categórica (hue) para colorir os pontos. Use para investigar relações e padrões, respondendo a perguntas como 'mostre a relação entre X e Y' ou 'plote a dispersão de A vs B colorido por C'.",
                args_schema=DispersaoInput
            ),
            StructuredTool.from_function(
                self.matriz_dispersao, name="matriz_dispersao",
                description="Gera uma matriz de gráficos de dispersão (pairplot), mostrando a relação par a par entre várias colunas numéricas. É uma ferramenta poderosa para uma visão geral das inter-relações nos dados. Use para perguntas amplas como 'analise as relações entre as variáveis' ou 'crie uma matriz de dispersão para A, B e C'.",
                args_schema=PairplotInput
            ),
            StructuredTool.from_function(
                self.tabela_cruzada, name="tabela_cruzada",
                description="Cria uma tabela de contingência (crosstab) para analisar a frequência da relação entre duas variáveis categóricas. Pode exibir os resultados como um heatmap. Use para perguntas como 'qual a relação entre a categoria A e a categoria B?' ou 'cruze os dados de X e Y'.",
                args_schema=CrosstabInput
            ),
            StructuredTool.from_function(
                self.detectar_outliers_iqr, name="detectar_outliers_iqr",
                description="Identifica e conta outliers em uma coluna numérica usando o método do Intervalo Interquartil (IQR). Use esta ferramenta para responder a perguntas como 'existem outliers na coluna X com base no IQR?' ou 'quantos outliers a coluna Y possui?'.",
                args_schema=OutlierIQRInput
            ),
            StructuredTool.from_function(
                self.detectar_outliers_zscore, name="detectar_outliers_zscore",
                description="Identifica e conta outliers em uma coluna numérica usando o método Z-score, que mede quantos desvios padrão um ponto de dados está da média. Use para perguntas como 'detecte outliers na coluna X usando Z-score'.",
                args_schema=OutlierZInput
            ),
            StructuredTool.from_function(
                self.detectar_outliers_isolation_forest, name="detectar_outliers_isolation_forest",
                description="Detecta outliers em múltiplas colunas simultaneamente usando o algoritmo Isolation Forest. É útil para encontrar anomalias em um contexto multivariado. Use para perguntas como 'encontre outliers considerando as colunas A e B juntas'.",
                args_schema=OutlierIFInput
            ),
            StructuredTool.from_function(
                self.resumo_outliers_dataset, name="resumo_outliers_dataset",
                description="Calcula e exibe um resumo da porcentagem e contagem de outliers para todas as colunas numéricas do dataset, usando o método IQR ou Z-score. Use para obter uma visão geral de quais colunas são mais afetadas por outliers, respondendo a perguntas como 'quais colunas têm mais outliers?'.",
                args_schema=ResumoOutInput
            ),
            StructuredTool.from_function(
                self.kmeans_clusterizar, name="kmeans_clusterizar",
                description="Aplica o algoritmo de clusterização K-means para agrupar os dados em 'k' clusters e visualiza o resultado em um gráfico 2D usando PCA. Use para segmentar os dados e encontrar grupos naturais, respondendo a perguntas como 'segmente os clientes em 3 grupos' ou 'clusterize os dados com base em A e B'.",
                args_schema=KMeansInput
            ),
            StructuredTool.from_function(
                self.converter_time_para_datetime, name="converter_time_para_datetime",
                description="Converte uma coluna chamada 'Time', que geralmente contém segundos, para um formato de data e hora (datetime). Também pode criar novas colunas úteis como hora do dia e dia. Use esta ferramenta se precisar fazer análises temporais e o dataset tiver uma coluna 'Time'.",
                args_schema=TimeConvertInput
            ),
            StructuredTool.from_function(
                self.tendencias_temporais, name="tendencias_temporais",
                description="Analisa e plota a tendência de uma coluna numérica ao longo do tempo. Requer uma coluna de data/hora (como 'Time_dt', que pode ser criada com `converter_time_para_datetime`). Agrupa os dados por hora (H), dia (D), semana (W) ou mês (M). Use para perguntas como 'mostre a tendência diária da coluna X' ou 'qual a evolução de Y ao longo do tempo?'.",
                args_schema=TendenciasInput
            ),
            StructuredTool.from_function(
                self.mostrar_conclusoes, name="mostrar_conclusoes",
                description="Exibe um resumo consolidado de todas as principais conclusões e insights gerados durante a sessão de análise. Use esta ferramenta quando quiser ver um resumo de tudo o que foi descoberto até agora, respondendo a perguntas como 'quais são as conclusões?' ou 'resuma a análise'."
            ),
        ]

    # ---------------- Implementações ----------------
    def listar_colunas(self) -> str:
        return f"Colunas: {', '.join(self.df.columns.tolist())}"

    def obter_descricao_geral(self) -> str:
        buffer = io.StringIO()
        self.df.info(buf=buffer)
        linhas, colunas = self.df.shape
        n_num = self.df.select_dtypes(include="number").shape[1]
        n_cat = self.df.select_dtypes(exclude="number").shape[1]
        null_pct = (self.df.isna().mean() * 100).round(2).sort_values(ascending=False)
        top_nulls = ", ".join([f"{c}: {p}%" for c, p in null_pct.head(3).items()]) if not null_pct.empty else "sem nulos"
        self._lembrar("descrição",
                      f"Dataset com {linhas} linhas, {colunas} colunas ({n_num} numéricas, {n_cat} categóricas). "
                      f"Colunas com mais nulos: {top_nulls}.")
        return f"{linhas} linhas x {colunas} colunas\n\n{buffer.getvalue()}"

    def obter_estatisticas_descritivas(self) -> str:
        desc = self.df.describe().T
        var_cols = desc.sort_values("std", ascending=False).head(3).index.tolist() if "std" in desc else []
        if var_cols:
            self._lembrar("describe", f"Maior dispersão (std): {', '.join(var_cols)}.")
        return desc.to_string()

    def plotar_histograma(self, coluna: str) -> str:
        if coluna not in self.df.columns:
            return f"Erro: '{coluna}' não existe."
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.histplot(self.df[coluna], kde=True, stat="density", linewidth=0, ax=ax)
        ax.set_title(f"Histograma: {coluna}"); ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        self._lembrar("distribuições", f"Histograma exibido para '{coluna}'.")
        self.ultima_coluna = coluna
        return f"Histograma de '{coluna}' exibido."

    def plotar_histogramas_dataset(self, colunas: str = "", kde: bool = True, bins: int = 30,
                                     cols_por_linha: int = 3, max_colunas: int = 12) -> str:
        if colunas.strip():
            cols = [c.strip() for c in colunas.split(",") if c.strip() and c in self.df.columns]
        else:
            cols = self.df.select_dtypes(include="number").columns.tolist()
        if not cols:
            return "Não há colunas válidas para histograma."
        cols = cols[:max_colunas]
        n = len(cols)
        linhas = (n + cols_por_linha - 1) // cols_por_linha
        fig, axes = plt.subplots(linhas, cols_por_linha, figsize=(cols_por_linha*4.2, linhas*3.4))
        axes = axes.flatten() if n > 1 else [axes]

        for i, c in enumerate(cols):
            ax = axes[i]
            try:
                sns.histplot(self.df[c].dropna(), kde=kde, bins=bins, stat="density", linewidth=0, ax=ax)
                ax.set_title(c)
                ax.grid(True, alpha=0.25)
            except Exception as e:
                ax.text(0.5, 0.5, f"Erro em {c}\n{e}", ha="center", va="center")
                ax.set_axis_off()
        for j in range(len(cols), len(axes)):
            axes[j].set_axis_off()

        fig.suptitle("Distribuição das variáveis (histogramas)", y=0.99)
        plt.tight_layout()
        st.pyplot(fig)

        skews = self.df[cols].skew(numeric_only=True).sort_values(ascending=False)
        mais_assim = ", ".join([f"{c}: {v:.2f}" for c, v in skews.head(3).items()])
        menos_assim = ", ".join([f"{c}: {v:.2f}" for c, v in skews.tail(3).items()])
        self._lembrar("distribuições", f"Maior assimetria positiva: {mais_assim}. Negativa: {menos_assim}.")
        self.ultima_coluna = cols[0]
        return f"Histogramas gerados para: {', '.join(cols)}."

    def frequencias_coluna(self, coluna: str, top_n: int = 10, bottom_n: int = 10) -> str:
        if coluna not in self.df.columns:
            return f"Erro: '{coluna}' não existe."
        s = self.df[coluna].dropna()
        resumo = ""
        if pd.api.types.is_numeric_dtype(s) and s.nunique() > 50:
            try:
                bins = pd.qcut(s, q=min(20, s.nunique()), duplicates="drop")
                cont = bins.value_counts().sort_values(ascending=False)
                resumo = f"Faixa mais comum: {cont.index[0]} ({int(cont.iloc[0])} ocorrências)."
            except Exception as e:
                resumo = f"Não foi possível calcular quantis ({e})."
        else:
            cont = s.value_counts()
            if not cont.empty:
                resumo = f"Valor mais comum: {cont.index[0]} ({int(cont.iloc[0])} ocorrências)."
        if resumo:
            self._lembrar("frequências", f"Em '{coluna}', {resumo}")
        cont_full = s.value_counts()
        return f"Top {top_n}:\n{cont_full.head(top_n)}\n\nBottom {bottom_n}:\n{cont_full.tail(bottom_n)}"

    def moda_coluna(self, coluna: str) -> str:
        if coluna not in self.df.columns:
            return f"Erro: '{coluna}' não existe."
        modos = self.df[coluna].mode(dropna=True)
        if modos.empty:
            return f"Não foi possível calcular a(s) moda(s) de '{coluna}'."
        valores = ", ".join(map(str, modos.tolist()))
        self._lembrar("moda", f"Moda de '{coluna}': {valores}")
        self.ultima_coluna = coluna
        return f"Moda(s) de '{coluna}': {valores}"

    def mostrar_correlacao(self, method: str = "pearson") -> str:
        df_num = self.df.select_dtypes(include="number")
        if df_num.empty:
            return "Sem colunas numéricas para correlação."
        corr = df_num.corr(method=method).abs()
        mask = ~corr.index.to_series().eq(corr.columns.values[:, None])
        top = (
            corr.where(mask)
            .unstack()
            .dropna()
            .sort_values(ascending=False)
            .drop_duplicates()
            .head(3)
        )
        pares = ", ".join([f"{a}~{b}: {v:.2f}" for (a, b), v in top.items()])
        self._lembrar("correlação", f"Pares mais correlacionados ({method}): {pares}.")

        fig, ax = plt.subplots(figsize=(12, 9))
        sns.heatmap(df_num.corr(method=method), cmap="coolwarm", ax=ax)
        ax.set_title(f"Mapa de calor da correlação ({method})")
        st.pyplot(fig)
        return f"Mapa de correlação ({method}) exibido."

    def plotar_dispersao(self, x: str, y: str, hue: str = "", amostra: int = 5000) -> str:
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
        if len(df_plot) > amostra:
            df_plot = df_plot.sample(n=amostra, random_state=42)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(data=df_plot, x=x, y=y, hue=(hue if hue else None), s=20, alpha=0.7, ax=ax, edgecolor=None)
        ax.set_title(f"Dispersão: {x} vs {y}" + (f" (hue={hue})" if hue else ""))
        ax.grid(True, linestyle="--", alpha=0.3)
        st.pyplot(fig)
        self._lembrar("dispersão", f"Dispersão exibida: {x} vs {y}" + (f" (hue={hue})" if hue else ""))
        self.ultima_coluna = y
        return "Gráfico de dispersão exibido."
        
    def matriz_dispersao(self, colunas: str = "", hue: str = "", amostra: int = 3000, corner: bool = True) -> str:
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
        
        g = sns.pairplot(df_plot, vars=cols, hue=hue if hue else None, corner=corner, diag_kind="hist", plot_kws=dict(s=15, alpha=0.7))
        g.fig.suptitle("Matriz de Dispersão (amostrada)", y=1.02)
        st.pyplot(g.fig)
        msg = f"Matriz de dispersão gerada para colunas: {cols}" + (f" (hue={hue})." if hue else ".")
        self._lembrar("dispersão", msg)
        return msg

    def tabela_cruzada(self, linhas: str, colunas: str, normalizar: bool = True, heatmap: bool = True, annot: bool = False, top_k: int = 20) -> str:
        for c in [linhas, colunas]:
            if c not in self.df.columns:
                return f"Erro: '{c}' não existe."
                
        s_l = self.df[linhas].astype(str); s_c = self.df[colunas].astype(str)
        top_l = s_l.value_counts().index[:top_k]; top_c = s_c.value_counts().index[:top_k]
        df_small = self.df[s_l.isin(top_l) & s_c.isin(top_c)]
        ct = pd.crosstab(df_small[linhas].astype(str), df_small[colunas].astype(str))
        
        tabela = (ct / ct.values.sum()) if normalizar else ct
        
        if heatmap:
            fig, ax = plt.subplots(figsize=(max(6, 0.4 * len(tabela.columns)), max(4, 0.4 * len(tabela.index))))
            sns.heatmap(tabela, cmap="Blues", ax=ax, annot=annot, fmt=".2f" if normalizar else "d")
            ax.set_title(f"Crosstab: {linhas} x {colunas}" + (" (normalizada)" if normalizar else ""))
            ax.set_xlabel(colunas); ax.set_ylabel(linhas)
            st.pyplot(fig)
            
        self._lembrar("crosstab", f"Crosstab gerada: {linhas} x {colunas}.")
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
        low, high = q1 - 1.5*iqr, q3 + 1.5*iqr
        mask = (s < low) | (s > high)
        n_out, n = int(mask.sum()), int(s.shape[0])
        pct = (n_out / n * 100) if n else 0.0
        if plot:
            fig, ax = plt.subplots(figsize=(8, 1.8))
            sns.boxplot(x=s, ax=ax); ax.set_title(f"Boxplot {coluna} (IQR)"); st.pyplot(fig)
        self._lembrar("outliers_col", f"'{coluna}': {n_out}/{n} ({pct:.2f}%) fora pelo IQR.")
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
        self._lembrar("outliers_col", f"Z-score '{coluna}': {n_out}/{n} ({pct:.2f}%) com |z|>{threshold}.")
        self.ultima_coluna = coluna
        return f"Outliers (Z>|{threshold}|) em '{coluna}': {n_out}/{n} = {pct:.3f}%."
        
    def detectar_outliers_isolation_forest(self, colunas: str = "", contamination: float = 0.01) -> str:
        try:
            from sklearn.preprocessing import StandardScaler
            from sklearn.ensemble import IsolationForest
        except Exception:
            return "Isolation Forest requer scikit-learn. Adicione 'scikit-learn' ao requirements."

        if colunas:
            cols = [c.strip() for c in colunas.split(",") if c.strip()]
        else:
            cols = self.df.select_dtypes(include="number").columns.tolist()

        if not cols:
            return "Não há colunas numéricas suficientes."
        X = self.df[cols].dropna()
        if X.empty:
            return "Após remover NAs, não sobraram linhas."

        scaler = StandardScaler(); Xs = scaler.fit_transform(X)
        used_contamination = max(1e-4, min(contamination, 0.5))
        clf = IsolationForest(contamination=used_contamination, random_state=42)
        labels = clf.fit_predict(Xs)  # -1 = outlier
        n_out = int((labels == -1).sum()); n = int(len(labels)); pct = (n_out / n * 100) if n else 0.0
        msg = (f"Isolation Forest: {n_out}/{n} = {pct:.3f}% de outliers nas colunas {cols} "
               f"(contamination={used_contamination}).")
        self._lembrar("outliers_ds", msg)
        return msg

    def resumo_outliers_dataset(self, method: str = "iqr", top_k: int = 10) -> str:
        df_num = self.df.select_dtypes(include="number")
        if df_num.empty:
            return "Sem colunas numéricas."
        linhas = []
        for col in df_num.columns:
            s = df_num[col].dropna(); n = int(s.shape[0])
            if n == 0:
                linhas.append((col, 0.0, 0, 0)); continue
            if method.lower() == "zscore":
                mu, sigma = s.mean(), s.std(ddof=0)
                if sigma == 0 or pd.isna(sigma):
                    cnt, pct = 0, 0.0
                else:
                    z = (s - mu) / sigma
                    cnt = int((z.abs() > 3.0).sum()); pct = (cnt / n * 100)
            else:
                q1, q3 = s.quantile(0.25), s.quantile(0.75); iqr = q3 - q1
                low, high = q1 - 1.5*iqr, q3 + 1.5*iqr
                cnt = int(((s < low) | (s > high)).sum()); pct = (cnt / n * 100)
            linhas.append((col, pct, cnt, n))
        linhas.sort(key=lambda x: x[1], reverse=True)
        top = linhas[:max(1, top_k)]
        media_pct = sum(p for _, p, _, _ in linhas) / len(linhas) if linhas else 0
        self._lembrar("outliers_ds",
                      "Top outliers ({}): {}. Média geral: {:.2f}%.".format(
                          method.upper(),
                          ", ".join([f"{c} {p:.2f}%" for c, p, _, _ in top[:3]]),
                          media_pct))
        partes = [f"Resumo de outliers por {method.upper()} (top {len(top)}):"]
        for col, pct, cnt, n in top:
            partes.append(f"- {col}: {cnt}/{n} = {pct:.3f}%")
        partes.append(f"Média % outliers numéricas: {media_pct:.3f}%")
        return "\n".join(partes)

    def kmeans_clusterizar(self, colunas: str = "", clusters: int = 3) -> str:
        try:
            from sklearn.preprocessing import StandardScaler
            from sklearn.cluster import KMeans
            from sklearn.decomposition import PCA
        except Exception:
            return "k-means requer scikit-learn. Adicione 'scikit-learn' ao requirements."
        if clusters < 2:
            clusters = 2
        cols = [c.strip() for c in colunas.split(",") if c.strip()] or self.df.select_dtypes(include="number").columns.tolist()
        if not cols:
            return "Sem colunas numéricas para clusterização."
        X = self.df[cols].dropna()
        if X.empty:
            return "Sem linhas após remoção de NAs."
        scaler = StandardScaler(); Xs = scaler.fit_transform(X)
        km = KMeans(n_clusters=clusters, n_init=10, random_state=42)
        labels = km.fit_predict(Xs)
        sizes = pd.Series(labels).value_counts().sort_index().to_dict()
        pca = PCA(n_components=2, random_state=42); XY = pca.fit_transform(Xs)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(x=XY[:,0], y=XY[:,1], hue=labels, legend=True, ax=ax)
        ax.set_title(f"K-means (k={clusters}) — PCA 2D"); ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        resumo = f"k-means (k={clusters}), inércia={km.inertia_:.2f}, tamanhos={sizes}."
        self._lembrar("clusters", resumo)
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
        self._lembrar("tempo", msg)
        return msg

    def tendencias_temporais(self, coluna: str, freq: str = "D") -> str:
        if coluna not in self.df.columns:
            return f"Erro: '{coluna}' não existe."
        if not pd.api.types.is_numeric_dtype(self.df[coluna]):
            return f"'{coluna}' deve ser numérica."
        ts_col = "Time_dt" if "Time_dt" in self.df.columns else None
        if not ts_col and "Time" in self.df.columns:
            self.converter_time_para_datetime(origem="1970-01-01 00:00:00", unidade="s", nova_coluna="Time_dt", criar_features=False)
            ts_col = "Time_dt"
        if not ts_col:
            return "Não há coluna temporal ('Time' ou 'Time_dt')."
        df_ts = self.df[[ts_col, coluna]].dropna().sort_values(ts_col).set_index(ts_col)
        series = df_ts[coluna].resample(freq).mean()
        if series.empty:
            return "Série vazia após reamostragem."
        # slope simples (regressão linear 1D)
        x = (series.index.view('i8') // 10**9)  # segundos
        y = series.values
        slope = ((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean())**2).sum() if ((x - x.mean())**2).sum() != 0 else 0
        direcao = "alta" if slope > 0 else "queda" if slope < 0 else "estável"
        self._lembrar("tendência", f"{coluna} em {freq}: tendência de {direcao} (inclinação {slope:.6f}).")

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(series.index, series.values, label=f"mean({coluna})")
        ax.set_title(f"Tendência temporal: {coluna} por {freq}")
        ax.set_xlabel("Tempo"); ax.set_ylabel(coluna); ax.grid(True, linestyle="--", alpha=0.3); ax.legend()
        st.pyplot(fig)
        return "Tendência temporal exibida."

    def mostrar_conclusoes(self) -> str:
        if not self.memoria_analises:
            return "Nenhuma análise registrada."
        linhas = []
        for item in self.memoria_analises:
            if item.startswith("[") and "] " in item:
                chave, texto = item.split("] ", 1)
                linhas.append((chave.strip("[]"), texto))
            else:
                linhas.append(("geral", item))
        linhas.sort(key=lambda x: x[0])
        blocos, atual, buffer = [], None, []
        for chave, texto in linhas:
            if chave != atual:
                if buffer:
                    blocos.append(f"**{atual.capitalize()}**\n- " + "\n- ".join(buffer))
                    buffer = []
                atual = chave
            buffer.append(texto)
        if buffer:
            blocos.append(f"**{atual.capitalize()}**\n- " + "\n- ".join(buffer))
        return "\n\n".join(blocos)

    # Pré-processador: ajuda com pedidos amplos
    def _preprocessar_pergunta(self, pergunta: str) -> str:
        # Simplificado para evitar o erro de chamadas múltiplas.
        t = pergunta.strip()
        if t in self.df.columns:
            self.ultima_coluna = t
            # Pede apenas UMA ação para evitar o erro.
            return f"Gere um histograma para a coluna '{t}'."
        
        low = t.lower()
        if "mostrar conclus" in low or "quais as conclus" in low:
              return "Use 'mostrar_conclusoes' para listar as conclusões da memória."

        return pergunta

# ========================= UI Streamlit =========================
st.set_page_config(page_title="Agente EDA (Streamlit)", layout="wide")
st.title("Agente EDA — LangChain + Gemini")
st.caption("Envie um CSV e faça perguntas. O agente gera gráficos quando necessário. Conclusões aparecem só quando você pedir.")

# --- Lógica de Persistência do Arquivo ---
DATA_DIR = "data"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Inicializar o estado da sessão
if "agente" not in st.session_state:
    st.session_state.agente = None
    st.session_state.messages = []
    st.session_state.csv_path = None
    st.session_state.chat_history_store = {}

# Tenta encontrar um CSV persistente ao iniciar a sessão
if not st.session_state.csv_path:
    try:
        csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
        if csv_files:
            # Carrega o primeiro CSV que encontrar no diretório
            st.session_state.csv_path = os.path.join(DATA_DIR, csv_files[0])
    except Exception as e:
        st.warning(f"Não foi possível ler o diretório de dados: {e}")


with st.sidebar:
    st.subheader("Upload do CSV")
    uploaded = st.file_uploader("Selecione um arquivo .csv", type=["csv"], key="file_uploader")
    
    if st.session_state.csv_path and os.path.exists(st.session_state.csv_path):
        st.success(f"Arquivo em uso: {os.path.basename(st.session_state.csv_path)}")
        if st.button("Remover arquivo e começar de novo"):
            try:
                os.remove(st.session_state.csv_path)
                # Limpa o estado da sessão completamente
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
            except Exception as e:
                st.error(f"Erro ao remover o arquivo: {e}")
    st.divider()

# 1. Lidar com um novo upload
if uploaded is not None:
    persistent_path = os.path.join(DATA_DIR, uploaded.name)
    with open(persistent_path, "wb") as f:
        f.write(uploaded.getvalue())
    
    # Se o arquivo mudou, reinicia o estado
    if st.session_state.csv_path != persistent_path:
        st.session_state.csv_path = persistent_path
        st.session_state.agente = None 
        st.session_state.messages = [] 
        st.rerun() 

# 2. Inicializar o agente se ele ainda não existir na sessão
if st.session_state.agente is None:
    if st.session_state.csv_path and os.path.exists(st.session_state.csv_path):
        try:
            st.session_state.agente = AgenteDeAnalise(
                caminho_arquivo_csv=st.session_state.csv_path,
                chat_history_store=st.session_state.chat_history_store
            )
            if not st.session_state.messages:
                st.success(f"CSV '{os.path.basename(st.session_state.csv_path)}' carregado. Pronto para conversar!")
                st.session_state.messages.append({"role": "assistant", "content": "Olá! Sou seu agente de análise. O que você gostaria de explorar no dataset?"})
        except Exception as e:
            st.error(f"Erro ao carregar o CSV '{os.path.basename(st.session_state.csv_path)}': {e}")
            st.session_state.agente = None

# 3. Exibir a mensagem inicial se nenhum agente foi carregado
if st.session_state.agente is None:
    st.info("📄 Envie um CSV para começar.")
else:
    # 4. Se o agente existe, exibe a interface de chat
    agente = st.session_state.agente
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Faça sua pergunta sobre o dataset..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analisando..."):
                try:
                    proc = agente._preprocessar_pergunta(prompt)
                    resposta = agente.agent.invoke(
                        {"input": proc},
                        config={"configurable": {"session_id": "ui_streamlit"},
                                "tags": ["ui", "streamlit"], "metadata": {"origin": "ui"}},
                    )
                    response_content = resposta.get("output", "Não consegui processar sua pergunta.")
                    st.write(response_content)
                    st.session_state.messages.append({"role": "assistant", "content": response_content})
                except Exception as e:
                    st.error(str(e))
                    st.session_state.messages.append({"role": "assistant", "content": f"Ocorreu um erro: {e}"})

