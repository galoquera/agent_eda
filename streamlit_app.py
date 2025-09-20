# -*- coding: utf-8 -*-
"""
Agente EDA (Streamlit) — LangChain + Gemini (+ LangSmith opcional)
- Arquitetura: Tool-Calling (mais direta e eficiente)
- Modelo: gemini-1.5-flash
- Upload de CSV genérico (com persistência de arquivo)
- Pergunta/resposta com ferramentas (gráficos no Streamlit)
- Memória interna (conclusões), exibida apenas quando o usuário pedir
"""

# --- REQUISITOS ---
# Para executar este agente, certifique-se de ter as seguintes bibliotecas instaladas:
# pip install streamlit pandas matplotlib seaborn python-dotenv langchain langchain-google-genai pydantic scikit-learn langsmith

import os
import io
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

# ---------- LangSmith (opcional) ----------
def _enable_langsmith(project: str = "EDA-Agent"):
    """Ativa o rastreamento com LangSmith se as variáveis de ambiente estiverem configuradas."""
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

        try:
            self.df = pd.read_csv(caminho_arquivo_csv)
        except FileNotFoundError:
            raise FileNotFoundError(f"Arquivo '{caminho_arquivo_csv}' não encontrado.")
        except Exception as e:
            raise ValueError(f"Erro ao ler o CSV: {e}")

        self.memoria_analises: List[str] = []
        self.ultima_coluna: str | None = None
        self.session_id = session_id

        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0,
            google_api_key=google_api_key,
        )

        tools = self._definir_ferramentas()

        # --- Prompt para Agente Tool-Calling ---
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system",
                 "Você é um Analista de Dados Sênior, especialista em Análise Exploratória de Dados (EDA).\n\n"
                 "**SUA MISSÃO:** Ajudar o usuário a extrair insights do dataset usando as ferramentas disponíveis.\n\n"
                 "**REGRAS DE OPERAÇÃO:**\n"
                 "1. **Use as Ferramentas:** Sempre que uma pergunta puder ser respondida por uma ferramenta, use-a.\n"
                 "2. **Seja Direto:** Execute a análise. Gere gráficos ou tabelas quando apropriado.\n"
                 "3. **Formato da Resposta (MUITO IMPORTANTE):**\n"
                 "   - Se uma ferramenta retornar uma tabela ou texto, sua resposta final **DEVE** começar com a saída **EXATA** e completa da ferramenta.\n"
                 "   - **NÃO** descreva a tabela (ex: 'A tabela abaixo mostra...'). Apenas exiba-a.\n"
                 "   - Após exibir a saída da ferramenta, adicione sua análise ou insights em um parágrafo separado.\n\n"
                 "**EXEMPLO DE RESPOSTA CORRETA:**\n"
                 "```text\n"
                 "              count      mean       std   min\n"
                 "Time       46.000000  17.065217  7.513809   0.0\n"
                 "V1         46.000000  -0.082728  1.425916  -2.3\n"
                 "```\n\n"
                 "Esta tabela apresenta as estatísticas descritivas das variáveis. A coluna 'Amount' possui a maior dispersão, enquanto a coluna 'Class' não apresenta variabilidade neste subconjunto.\n"
                ),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
                MessagesPlaceholder("agent_scratchpad"),
            ]
        )

        # --- Criação do Agente Tool-Calling ---
        agent = create_tool_calling_agent(self.llm, tools, prompt)
        
        self.executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            max_iterations=5, # Salvaguarda contra loops inesperados
            handle_parsing_errors="Por favor, reformule sua pergunta. Não consegui processar a solicitação.",
        )

        def get_session_history(session_id: str):
            if session_id not in chat_history_store:
                chat_history_store[session_id] = ChatMessageHistory()
            return chat_history_store[session_id]

        self.agent_with_history = RunnableWithMessageHistory(
            self.executor,
            get_session_history,
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
            coluna: str = Field(description="A coluna numérica para a qual gerar o histograma.")

        class HistAllInput(BaseModel):
            colunas: str = Field(default="", description="Lista de colunas separada por vírgula. Se vazio, usa todas as colunas numéricas.")
            kde: bool = Field(default=True, description="Define se a curva de densidade (KDE) deve ser exibida.")
            bins: int = Field(default=30, description="O número de barras (bins) no histograma.")
            cols_por_linha: int = Field(default=2, description="Número de gráficos a serem exibidos por linha.")
            max_colunas: int = Field(default=4, description="Número máximo de colunas para plotar para evitar poluição visual.")

        class FrequenciasInput(BaseModel):
            coluna: str = Field(description="A coluna para a qual calcular as frequências.")
            top_n: int = Field(default=10, description="Número de itens mais frequentes a serem exibidos.")
            bottom_n: int = Field(default=10, description="Número de itens menos frequentes a serem exibidos.")

        class ModaInput(BaseModel):
            coluna: str = Field(description="A coluna para a qual calcular a moda.")
            
        class CorrelacaoInput(BaseModel):
            method: str = Field(default="pearson", description="O método de correlação a ser usado: 'pearson', 'spearman' ou 'kendall'.")

        class DispersaoInput(BaseModel):
            x: str = Field(description="O nome da coluna para o eixo X (deve ser numérica).")
            y: str = Field(description="O nome da coluna para o eixo Y (deve ser numérica).")
            hue: str = Field(default="", description="Opcional: nome da coluna categórica para colorir os pontos.")
            amostra: int = Field(default=5000, description="Número máximo de pontos para plotar, para evitar gráficos lentos e poluídos.")
            
        class PairplotInput(BaseModel):
            colunas: str = Field(default="", description="Colunas separadas por vírgula. Se vazio, o sistema selecionará até 6 colunas numéricas com maior variância.")
            hue: str = Field(default="", description="Opcional: nome da coluna categórica para colorir os gráficos.")
            amostra: int = Field(default=3000, description="Número máximo de linhas a serem usadas na plotagem.")
            corner: bool = Field(default=True, description="Se True, exibe apenas a metade inferior da matriz para evitar redundância.")

        class CrosstabInput(BaseModel):
            linhas: str = Field(description="A coluna a ser usada como linhas na tabela cruzada (geralmente categórica).")
            colunas: str = Field(description="A coluna a ser usada como colunas na tabela cruzada (geralmente categórica).")
            normalizar: bool = Field(default=True, description="Se True, os valores são convertidos em porcentagens.")
            heatmap: bool = Field(default=True, description="Se True, exibe a tabela como um mapa de calor.")
            annot: bool = Field(default=False, description="Se True, anota os valores no mapa de calor (pode poluir visualmente).")
            top_k: int = Field(default=20, description="Limita o número de categorias em cada eixo para evitar tabelas excessivamente grandes.")

        class OutlierIQRInput(BaseModel):
            coluna: str = Field(description="A coluna numérica para detectar outliers.")
            plot: bool = Field(default=False, description="Se True, gera um boxplot para visualizar os outliers.")

        class OutlierZInput(BaseModel):
            coluna: str = Field(description="A coluna numérica para detectar outliers.")
            threshold: float = Field(default=3.0, description="O limite de Z-score. Pontos com Z-score absoluto maior que este valor são considerados outliers.")
            plot: bool = Field(default=False, description="Se True, gera um histograma da distribuição dos Z-scores.")
            
        class OutlierIFInput(BaseModel):
            colunas: str = Field(default="", description="Colunas separadas por vírgula. Se vazio, usa todas as colunas numéricas.")
            contamination: float = Field(default=0.01, description="A proporção esperada de outliers no dataset (ex.: 0.01 para 1%).")

        class ResumoOutInput(BaseModel):
            method: str = Field(default="iqr", description="O método a ser usado: 'iqr' ou 'zscore'.")
            top_k: int = Field(default=10, description="O número de colunas com mais outliers a serem exibidas.")

        class KMeansInput(BaseModel):
            colunas: str = Field(default="", description="Lista de colunas numéricas separada por vírgula para usar na clusterização. Se vazio, usa todas as numéricas.")
            clusters: int = Field(default=3, description="O número de clusters (k) a serem criados (mínimo de 2).")

        class TimeConvertInput(BaseModel):
            origem: str = Field(default="", description="Data e hora de início no formato 'YYYY-MM-DD HH:MM:SS'. Se vazio, a coluna 'Time' é tratada como segundos relativos.")
            unidade: str = Field(default="s", description="A unidade da coluna 'Time': 's' (segundos), 'ms' (milissegundos), 'm' (minutos), 'h' (horas).")
            nova_coluna: str = Field(default="", description="Opcional: nome para a nova coluna datetime criada.")
            criar_features: bool = Field(default=True, description="Se True, cria automaticamente colunas adicionais como 'Time_hour' e 'Time_day'.")

        class TendenciasInput(BaseModel):
            coluna: str = Field(description="A coluna numérica a ser analisada (ex.: 'Amount').")
            freq: str = Field(default="D", description="A frequência de reamostragem: 'H' (hora), 'D' (dia), 'W' (semana), 'M' (mês).")

        return [
            StructuredTool.from_function(
                self.listar_colunas, name="listar_colunas",
                description="Retorna uma lista com os nomes exatos de todas as colunas. Use para saber os nomes das colunas antes de usar outras ferramentas. Ex: 'quais são as colunas?'."
            ),
            StructuredTool.from_function(
                self.obter_descricao_geral, name="descricao_geral_dados",
                description="Fornece um resumo da estrutura do dataset (linhas, colunas, tipos, nulos). Essencial para uma primeira visão. Ex: 'me dê um resumo dos dados'."
            ),
            StructuredTool.from_function(
                self.obter_estatisticas_descritivas, name="estatisticas_descritivas",
                description="Calcula estatísticas descritivas (média, desvio padrão, etc.) para as colunas numéricas. Ex: 'qual a média da coluna X?', 'descreva as variáveis numéricas'."
            ),
            StructuredTool.from_function(
                self.plotar_histograma, name="plotar_histograma",
                description="Gera um histograma para uma única coluna numérica para visualizar sua distribuição. Ex: 'mostre a distribuição da coluna X'.",
                args_schema=HistogramaInput
            ),
            StructuredTool.from_function(
                self.plotar_histogramas_dataset, name="plotar_histogramas_dataset",
                description="Gera histogramas para múltiplas colunas numéricas de uma só vez. Ex: 'mostre a distribuição de todas as variáveis'.",
                args_schema=HistAllInput
            ),
            StructuredTool.from_function(
                self.frequencias_coluna, name="frequencias_coluna",
                description="Calcula as contagens de frequência dos valores em uma coluna. Ex: 'quais são os valores mais comuns na coluna X?'.",
                args_schema=FrequenciasInput
            ),
            StructuredTool.from_function(
                self.moda_coluna, name="moda_coluna",
                description="Calcula o valor mais frequente (moda) de uma coluna. Ex: 'qual é a moda da coluna X?'.",
                args_schema=ModaInput
            ),
            StructuredTool.from_function(
                self.mostrar_correlacao, name="plotar_mapa_correlacao",
                description="Cria um mapa de calor de correlação entre as colunas numéricas. Ex: 'quais variáveis estão mais correlacionadas?', 'mostre o mapa de correlação'.",
                args_schema=CorrelacaoInput
            ),
            StructuredTool.from_function(
                self.plotar_dispersao, name="plotar_dispersao",
                description="Cria um gráfico de dispersão para visualizar a relação entre duas variáveis numéricas (X e Y). Ex: 'mostre a relação entre X e Y'.",
                args_schema=DispersaoInput
            ),
            StructuredTool.from_function(
                self.matriz_dispersao, name="matriz_dispersao",
                description="Gera uma matriz de gráficos de dispersão (pairplot) para visualizar relações par a par. Ex: 'analise as relações entre as variáveis'.",
                args_schema=PairplotInput
            ),
            StructuredTool.from_function(
                self.tabela_cruzada, name="tabela_cruzada",
                description="Cria uma tabela de contingência (crosstab) para analisar a relação entre duas variáveis categóricas. Ex: 'qual a relação entre a categoria A e a B?'.",
                args_schema=CrosstabInput
            ),
            StructuredTool.from_function(
                self.detectar_outliers_iqr, name="detectar_outliers_iqr",
                description="Identifica outliers em uma coluna usando o método do Intervalo Interquartil (IQR). Ex: 'existem outliers na coluna X pelo método IQR?'.",
                args_schema=OutlierIQRInput
            ),
            StructuredTool.from_function(
                self.detectar_outliers_zscore, name="detectar_outliers_zscore",
                description="Identifica outliers em uma coluna usando o método Z-score. Ex: 'detecte outliers na coluna X usando Z-score'.",
                args_schema=OutlierZInput
            ),
            StructuredTool.from_function(
                self.detectar_outliers_isolation_forest, name="detectar_outliers_isolation_forest",
                description="Detecta anomalias em um contexto multivariado usando o Isolation Forest. Ex: 'encontre outliers considerando as colunas A e B juntas'.",
                args_schema=OutlierIFInput
            ),
            StructuredTool.from_function(
                self.resumo_outliers_dataset, name="resumo_outliers_dataset",
                description="Exibe um resumo de outliers para todas as colunas numéricas. Ex: 'quais colunas têm mais outliers?'.",
                args_schema=ResumoOutInput
            ),
            StructuredTool.from_function(
                self.kmeans_clusterizar, name="kmeans_clusterizar",
                description="Aplica K-means para agrupar os dados e visualiza os clusters. Ex: 'segmente os clientes em 3 grupos'.",
                args_schema=KMeansInput
            ),
            StructuredTool.from_function(
                self.converter_time_para_datetime, name="converter_time_para_datetime",
                description="Converte a coluna 'Time' (em segundos) para datetime e cria features temporais. Necessário para análises de tempo.",
                args_schema=TimeConvertInput
            ),
            StructuredTool.from_function(
                self.tendencias_temporais, name="tendencias_temporais",
                description="Plota a tendência de uma coluna numérica ao longo do tempo. Requer uma coluna de data/hora. Ex: 'mostre a tendência diária da coluna X'.",
                args_schema=TendenciasInput
            ),
            StructuredTool.from_function(
                self.mostrar_conclusoes, name="mostrar_conclusoes",
                description="Exibe um resumo de todos os insights gerados durante a sessão. Ex: 'quais são as conclusões?', 'resuma a análise'."
            ),
        ]

    # ---------------- Implementações ----------------
    def listar_colunas(self, dummy: str = "") -> str:
        return f"Colunas: {', '.join(self.df.columns.tolist())}"

    def obter_descricao_geral(self, dummy: str = "") -> str:
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

    def obter_estatisticas_descritivas(self, dummy: str = "") -> str:
        desc = self.df.describe().T
        var_cols = desc.sort_values("std", ascending=False).head(3).index.tolist() if "std" in desc else []
        if var_cols:
            self._lembrar("describe", f"Maior dispersão (std): {', '.join(var_cols)}.")
        
        desc_str = desc.to_string()
        return f"```text\n{desc_str}\n```"

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
                                   cols_por_linha: int = 2, max_colunas: int = 4) -> str:
        if colunas.strip():
            cols = [c.strip() for c in colunas.split(",") if c.strip() and c in self.df.columns]
        else:
            cols = self.df.select_dtypes(include="number").columns.tolist()
        if not cols:
            return "Não há colunas válidas para histograma."
        
        total_cols_disponiveis = len(cols)
        cols = cols[:max_colunas]
        n = len(cols)
        linhas = (n + cols_por_linha - 1) // cols_por_linha
        fig, axes = plt.subplots(linhas, cols_por_linha, figsize=(cols_por_linha*5, linhas*3.8))
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

        fig.suptitle("Distribuição das variáveis (amostra inicial)", y=1.02)
        plt.tight_layout()
        st.pyplot(fig)

        skews = self.df[cols].skew(numeric_only=True).sort_values(ascending=False)
        mais_assim = ", ".join([f"{c}: {v:.2f}" for c, v in skews.head(3).items()])
        menos_assim = ", ".join([f"{c}: {v:.2f}" for c, v in skews.tail(3).items()])
        self._lembrar("distribuições", f"Maior assimetria positiva: {mais_assim}. Negativa: {menos_assim}.")
        self.ultima_coluna = cols[0]
        
        if total_cols_disponiveis > max_colunas:
            return (f"Uma amostra inicial de {len(cols)} histogramas foi exibida para dar uma visão geral. "
                    f"O dataset possui {total_cols_disponiveis} colunas numéricas no total. "
                    f"Para analisar outras colunas, por favor, especifique os nomes (ex: 'histograma para V10, V11').")
        else:
            return f"Histogramas gerados para todas as {len(cols)} colunas numéricas: {', '.join(cols)}."

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
        
        top_df = cont_full.head(top_n).reset_index()
        top_df.columns = [coluna, 'Contagem']
        
        bottom_df = cont_full.tail(bottom_n).reset_index()
        bottom_df.columns = [coluna, 'Contagem']
        
        top_str = top_df.to_string(index=False)
        bottom_str = bottom_df.to_string(index=False)
        
        return (f"**Top {top_n} Mais Frequentes:**\n"
                f"```text\n{top_str}\n```\n\n"
                f"**Bottom {bottom_n} Menos Frequentes:**\n"
                f"```text\n{bottom_str}\n```")

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
        
        self._lembrar("crosstab", f"Crosstab gerada: {linhas} x {colunas}.")
        self.ultima_coluna = colunas

        if heatmap:
            fig, ax = plt.subplots(figsize=(max(6, 0.4 * len(tabela.columns)), max(4, 0.4 * len(tabela.index))))
            sns.heatmap(tabela, cmap="Blues", ax=ax, annot=annot, fmt=".2f" if normalizar else "d")
            ax.set_title(f"Crosstab: {linhas} x {colunas}" + (" (normalizada)" if normalizar else ""))
            ax.set_xlabel(colunas); ax.set_ylabel(linhas)
            st.pyplot(fig)
            return f"O mapa de calor da tabela cruzada entre '{linhas}' e '{colunas}' foi exibido."
        else:
            # Retorna a tabela como texto formatado se nenhum gráfico for gerado.
            return f"Tabela Cruzada: {linhas} vs {colunas}\n```text\n{tabela.to_string()}\n```"

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
        msg = f"Outliers (IQR) em '{coluna}': {n_out}/{n} = {pct:.3f}%."
        if plot:
            fig, ax = plt.subplots(figsize=(8, 1.8))
            sns.boxplot(x=s, ax=ax); ax.set_title(f"Boxplot {coluna} (IQR)"); st.pyplot(fig)
            msg += " O boxplot foi exibido para visualização."
        self._lembrar("outliers_col", f"'{coluna}': {n_out}/{n} ({pct:.2f}%) fora pelo IQR.")
        self.ultima_coluna = coluna
        return msg

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
        msg = f"Outliers (Z>|{threshold}|) em '{coluna}': {n_out}/{n} = {pct:.3f}%."
        if plot:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(z, kde=True, stat="density", linewidth=0, ax=ax)
            ax.set_title(f"Distribuição de Z-scores - {coluna}")
            st.pyplot(fig)
            msg += " O histograma dos Z-scores foi exibido."
        self._lembrar("outliers_col", f"Z-score '{coluna}': {n_out}/{n} ({pct:.2f}%) com |z|>{threshold}.")
        self.ultima_coluna = coluna
        return msg
        
    def detectar_outliers_isolation_forest(self, colunas: str = "", contamination: float = 0.01) -> str:
        try:
            from sklearn.preprocessing import StandardScaler
            from sklearn.ensemble import IsolationForest
        except ImportError:
            return "Isolation Forest requer scikit-learn. Adicione 'scikit-learn' ao seu requirements.txt."

        if colunas:
            cols = [c.strip() for c in colunas.split(",") if c.strip() and c in self.df.columns]
        else:
            cols = self.df.select_dtypes(include="number").columns.tolist()

        if not cols:
            return "Não há colunas numéricas válidas para a análise."
        X = self.df[cols].dropna()
        if X.empty:
            return "Após remover valores nulos, não sobraram dados para análise."

        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler(); Xs = scaler.fit_transform(X)
        used_contamination = max(1e-4, min(contamination, 0.5))
        from sklearn.ensemble import IsolationForest
        clf = IsolationForest(contamination=used_contamination, random_state=42)
        labels = clf.fit_predict(Xs)
        n_out = int((labels == -1).sum()); n = int(len(labels)); pct = (n_out / n * 100) if n else 0.0
        msg = (f"Isolation Forest encontrou {n_out}/{n} ({pct:.3f}%) outliers nas colunas {cols} "
               f"(com contaminação esperada de {used_contamination}).")
        self._lembrar("outliers_ds", msg)
        return msg

    def resumo_outliers_dataset(self, method: str = "iqr", top_k: int = 10) -> str:
        df_num = self.df.select_dtypes(include="number")
        if df_num.empty:
            return "Nenhuma coluna numérica encontrada para analisar outliers."
        linhas = []
        for col in df_num.columns:
            s = df_num[col].dropna(); n = int(s.shape[0])
            if n == 0:
                linhas.append((col, 0.0, 0, 0)); continue
            
            cnt = 0
            if method.lower() == "zscore":
                mu, sigma = s.mean(), s.std(ddof=0)
                if sigma > 0 and not pd.isna(sigma):
                    z = (s - mu) / sigma
                    cnt = int((z.abs() > 3.0).sum())
            else: # Padrão para IQR
                q1, q3 = s.quantile(0.25), s.quantile(0.75); iqr = q3 - q1
                if iqr > 0:
                    low, high = q1 - 1.5*iqr, q3 + 1.5*iqr
                    cnt = int(((s < low) | (s > high)).sum())
            
            pct = (cnt / n * 100) if n > 0 else 0
            linhas.append((col, pct, cnt, n))
            
        linhas.sort(key=lambda x: x[1], reverse=True)
        top = linhas[:max(1, top_k)]
        media_pct = sum(p for _, p, _, _ in linhas) / len(linhas) if linhas else 0
        
        self._lembrar("outliers_ds",
                      "Top outliers ({}): {}. Média geral: {:.2f}%.".format(
                          method.upper(),
                          ", ".join([f"{c} ({p:.2f}%)" for c, p, _, _ in top[:3]]),
                          media_pct))
        
        partes = [f"Resumo de outliers por {method.upper()} (top {len(top)} colunas):"]
        for col, pct, cnt, n in top:
            partes.append(f"- **{col}**: {cnt} de {n} pontos ({pct:.3f}%)")
        partes.append(f"\n*Média geral de outliers nas colunas numéricas: {media_pct:.3f}%*")
        return "\n".join(partes)

    def kmeans_clusterizar(self, colunas: str = "", clusters: int = 3) -> str:
        try:
            from sklearn.preprocessing import StandardScaler
            from sklearn.cluster import KMeans
            from sklearn.decomposition import PCA
        except ImportError:
            return "K-means requer scikit-learn. Adicione 'scikit-learn' ao seu requirements.txt."
        
        k = max(2, clusters)
        cols = [c.strip() for c in colunas.split(",") if c.strip() and c in self.df.columns] or self.df.select_dtypes(include="number").columns.tolist()
        if not cols:
            return "Nenhuma coluna numérica encontrada para clusterização."
        
        X = self.df[cols].dropna()
        if X.shape[0] < k:
            return f"Não há dados suficientes ({X.shape[0]} linhas) para criar {k} clusters."
            
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler(); Xs = scaler.fit_transform(X)
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=k, n_init='auto', random_state=42)
        labels = km.fit_predict(Xs)
        sizes = pd.Series(labels).value_counts().sort_index().to_dict()
        
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2, random_state=42); XY = pca.fit_transform(Xs)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(x=XY[:,0], y=XY[:,1], hue=labels, palette="viridis", legend="full", ax=ax)
        ax.set_title(f"Visualização dos Clusters (K-means, k={k}) via PCA"); ax.set_xlabel("Componente Principal 1"); ax.set_ylabel("Componente Principal 2"); ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
        resumo = f"Clusterização K-means (k={k}) concluída e o gráfico de visualização foi exibido. Inércia: {km.inertia_:.2f}. Tamanhos dos clusters: {sizes}."
        self._lembrar("clusters", resumo)
        return resumo

    def converter_time_para_datetime(self, origem: str = "", unidade: str = "s",
                                     nova_coluna: str = "", criar_features: bool = True) -> str:
        col = "Time"
        if col not in self.df.columns:
            return "Erro: coluna 'Time' não encontrada no dataset."
        s = pd.to_numeric(self.df[col], errors="coerce")
        if s.isna().all():
            return "Erro: a coluna 'Time' não pôde ser convertida para um formato numérico."
        
        try:
            td = pd.to_timedelta(s, unit=unidade)
        except Exception as e:
            return f"Erro ao converter 'Time' para Timedelta: {e}"
            
        target = (nova_coluna or "Time_dt").strip()
        created = [target]
        
        if origem.strip():
            try:
                base = pd.to_datetime(origem)
                self.df[target] = base + td
                modo = f"ancorado em '{origem}'"
            except Exception as e:
                return f"Erro ao converter a data de origem: {e}"
        else:
            self.df[target] = td
            modo = "relativo (Timedelta)"
            
        if criar_features:
            dt_accessor = self.df[target].dt if pd.api.types.is_datetime64_any_dtype(self.df[target]) else None
            if dt_accessor:
                self.df["Time_hour"] = dt_accessor.hour
                self.df["Time_day_of_week"] = dt_accessor.day_name()
                created += ["Time_hour", "Time_day_of_week"]
            
        msg = f"Coluna 'Time' convertida com sucesso ({modo}). Colunas criadas: {', '.join(created)}."
        self._lembrar("tempo", msg)
        return msg

    def tendencias_temporais(self, coluna: str, freq: str = "D") -> str:
        if coluna not in self.df.columns:
            return f"Erro: a coluna '{coluna}' não existe."
        if not pd.api.types.is_numeric_dtype(self.df[coluna]):
            return f"Erro: a coluna '{coluna}' deve ser numérica para análise de tendência."
            
        ts_col = next((c for c in self.df.columns if pd.api.types.is_datetime64_any_dtype(self.df[c])), None)
        if not ts_col:
            return "Erro: Nenhuma coluna de data/hora encontrada. Use a ferramenta `converter_time_para_datetime` primeiro."
            
        df_ts = self.df[[ts_col, coluna]].dropna().set_index(ts_col)
        if df_ts.empty:
            return "Não há dados suficientes para analisar a tendência."
            
        series_sum = df_ts[coluna].resample(freq).sum()
        series_mean = df_ts[coluna].resample(freq).mean()
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        ax1.plot(series_sum.index, series_sum.values, label=f"Soma de {coluna}", color="navy")
        ax1.set_title(f"Soma de '{coluna}' Agregada por Período ('{freq}')")
        ax1.set_ylabel("Soma Total"); ax1.grid(True, linestyle="--", alpha=0.5)
        
        ax2.plot(series_mean.index, series_mean.values, label=f"Média de {coluna}", color="darkred")
        ax2.set_title(f"Média de '{coluna}' Agregada por Período ('{freq}')")
        ax2.set_xlabel("Tempo"); ax2.set_ylabel("Média"); ax2.grid(True, linestyle="--", alpha=0.5)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        return f"Gráfico de tendência temporal para a coluna '{coluna}' com frequência '{freq}' foi exibido."

    def mostrar_conclusoes(self, dummy: str = "") -> str:
        if not self.memoria_analises:
            return "Nenhuma conclusão foi registrada na memória ainda."
        
        blocos = {}
        for item in self.memoria_analises:
            try:
                chave, texto = item.split("] ", 1)
                chave = chave.strip("[]").capitalize()
                if chave not in blocos:
                    blocos[chave] = []
                blocos[chave].append(texto)
            except ValueError:
                if "Geral" not in blocos:
                    blocos["Geral"] = []
                blocos["Geral"].append(item)
        
        output = ["### Resumo das Análises\n"]
        for chave, textos in blocos.items():
            output.append(f"**{chave}**")
            for texto in textos:
                output.append(f"- {texto}")
            output.append("")  # Linha em branco para espaçamento
            
        return "\n".join(output)

# ========================= UI Streamlit =========================
st.set_page_config(page_title="Agente EDA (Streamlit)", layout="wide")
st.title("Agente EDA — LangChain + Gemini")
st.caption("Envie um CSV e faça perguntas. O agente gera gráficos e insights. Peça para ele 'resumir a análise' a qualquer momento.")

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
            st.session_state.csv_path = os.path.join(DATA_DIR, csv_files[0])
    except Exception as e:
        st.warning(f"Não foi possível ler o diretório de dados: {e}")

with st.sidebar:
    st.subheader("Configuração do Dataset")
    uploaded = st.file_uploader("Selecione um arquivo .csv", type=["csv"], key="file_uploader")
    
    if st.session_state.csv_path and os.path.exists(st.session_state.csv_path):
        st.success(f"Em uso: {os.path.basename(st.session_state.csv_path)}")
        if st.button("🗑️ Remover arquivo e reiniciar"):
            try:
                os.remove(st.session_state.csv_path)
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
    
    if st.session_state.csv_path != persistent_path:
        st.session_state.csv_path = persistent_path
        # Reinicia o estado para forçar a recarga do agente com o novo arquivo
        keys_to_reset = ["agente", "messages", "chat_history_store"]
        for key in keys_to_reset:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

# 2. Inicializar o agente se ele ainda não existir na sessão
if "agente" not in st.session_state or st.session_state.agente is None:
    if st.session_state.csv_path and os.path.exists(st.session_state.csv_path):
        with st.spinner(f"Carregando '{os.path.basename(st.session_state.csv_path)}' e inicializando o agente..."):
            try:
                if "chat_history_store" not in st.session_state:
                        st.session_state.chat_history_store = {}
                
                st.session_state.agente = AgenteDeAnalise(
                    caminho_arquivo_csv=st.session_state.csv_path,
                    chat_history_store=st.session_state.chat_history_store
                )
                if "messages" not in st.session_state or not st.session_state.messages:
                    st.session_state.messages = [{"role": "assistant", "content": "Olá! Sou seu agente de análise. O que você gostaria de explorar no dataset?"}]
            except Exception as e:
                st.error(f"Erro ao inicializar o agente: {e}")
                st.session_state.agente = None

# 3. Exibir a interface de chat
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if st.session_state.agente is None:
    if not uploaded:
        st.info("📄 Por favor, envie um arquivo CSV para começar a análise.")
else:
    if prompt := st.chat_input("Faça sua pergunta sobre o dataset..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analisando e pensando..."):
                try:
                    agente = st.session_state.agente
                    
                    resposta = agente.agent_with_history.invoke(
                        {"input": prompt},
                        config={"configurable": {"session_id": agente.session_id}},
                    )
                    response_content = resposta.get("output", "Não consegui processar sua pergunta.")
                    st.markdown(response_content)
                    st.session_state.messages.append({"role": "assistant", "content": response_content})
                except StopIteration:
                    error_message = "O agente não conseguiu chegar a uma conclusão. Tente reformular sua pergunta de uma maneira mais direta."
                    st.error(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": error_message})
                except Exception as e:
                    error_message = f"Ocorreu um erro inesperado: {str(e)}"
                    st.error(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": error_message})

