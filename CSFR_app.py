import streamlit as st
import pandas as pd
import numpy as np
import os
import io
import base64
import tempfile
from datetime import datetime
import joblib
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
import logging

# Configurar logging para acompanhar o processo
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Função para carregar imagem local e converter para base64


def carregar_logo(path):
    try:
        with open(path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except Exception as e:
        logging.error(f"Erro ao carregar logo: {e}")
        return None

# Encontrar o caminho da logo


def encontrar_logo():
    # Lista de possíveis caminhos para a logo
    logo_paths = [
        "petrobras_logo_horizontal.png",  # Na pasta atual
        "venv/petrobras_logo_horizontal.png",  # No diretório venv
        # No mesmo diretório do script
        os.path.join(os.path.dirname(__file__),
                     "petrobras_logo_horizontal.png"),
        "assets/petrobras_logo_horizontal.png",  # Em uma pasta assets
        os.path.join(os.path.dirname(__file__), "assets",
                     "petrobras_logo_horizontal.png")  # assets relativo ao script
    ]

    # Tenta cada caminho até encontrar um válido
    for path in logo_paths:
        if os.path.exists(path):
            return path

    # Retorna None se nenhum caminho for válido
    logging.warning(
        "Logo da Petrobras não encontrada. Verifique os caminhos disponíveis.")
    return None


# Carregando imagens
logo_path = encontrar_logo()
logo_base64 = carregar_logo(logo_path) if logo_path else None

# Configurar a página do Streamlit
try:
    # Tenta usar o ícone da Petrobras, mas tem um fallback caso o arquivo não seja encontrado
    try:
        # Primeiro tentamos o caminho específico fornecido
        st.set_page_config(
            page_title="SPIC - Sistema de Previsão de Índices Contratuais",
            page_icon="C:\\Users\\FUDO\\OneDrive - PETROBRAS\\Documentos\\Python\\Problema do Leandro\\petrobras_icone.png",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    except:
        # Se não encontrar o ícone no caminho específico, procura na pasta atual
        import os
        # Lista de possíveis caminhos para o ícone
        icon_paths = [
            "petrobras_icone.png",  # Na pasta atual
            # No mesmo diretório do script
            os.path.join(os.path.dirname(__file__), "petrobras_icone.png"),
            "assets/petrobras_icone.png",  # Em uma pasta assets
            os.path.join(os.path.dirname(__file__), "assets",
                         "petrobras_icone.png")  # assets relatico ao script
        ]

        # Tenta cada caminho até encontrar um válido
        for path in icon_paths:
            if os.path.exists(path):
                st.set_page_config(
                    page_title="SPIC - Sistema de Previsão de Índices Contratuais",
                    page_icon=path,
                    layout="wide",
                    initial_sidebar_state="expanded"
                )
                break
        else:
            # Se nenhum ícone for encontrado, usa um emoji como fallback
            st.set_page_config(
                page_title="SPIC - Sistema de Previsão de Índices Contratuais",
                page_icon="🔍",
                layout="wide",
                initial_sidebar_state="expanded"
            )
except Exception as e:
    # Fallback final em caso de qualquer erro
    st.set_page_config(
        page_title="SPIC - Sistema de Previsão de Índices Contratuais",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    print(f"Erro ao configurar a página: {e}")

# CSS completo (barra fixa com nome móvel automaticamente)
st.markdown("""
<style>
    header[data-testid="stHeader"] { z-index: -1; }
    .fixed-header {
        background-color: #ffffff;
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        z-index: 1001;
        display: flex;
        align-items: center;
        padding: 15px 20px;
        padding-left: 400px; /* posição padrão quando sidebar aberta */
        color: #008542;
        border-bottom: 3px solid #008542;
        transition: padding-left 0.3s ease;
    }
    .fixed-header h2 {
        margin: 0;
        padding: 0;
        font-size: 20px;
        color: #008542;
    }
    .fixed-header img.logo {
        position: absolute;
        height: 40px;
        right: 20px;
    }
    .content {
        margin-top: 100px;
    }
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 3px solid #ffffff;
    }
    /* Detecta sidebar fechada */
    section[data-testid="stSidebar"][aria-expanded="false"] ~ div .fixed-header {
        padding-left: 80px !important;
    }
    [data-testid="stSidebar"][aria-expanded="false"] + section .fixed-header {
        padding-left: 80px !important;
    }
    
    /* Estilos adicionais para o aplicativo */
    .main-header {
        font-size: 2.5rem;
        color: #1E6091;
        text-align: center;
        margin-bottom: 1rem;
        margin-top: 100px; /* Espaço para o cabeçalho fixo */
    }
    .sub-header {
        font-size: 1.8rem;
        color: #1E6091;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #2E8BC0;
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
    }
    .info-box {
        background-color: #E6F3FF;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #DFFFDF;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .warning-box {
        background-color: #FFFFCF;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .error-box {
        background-color: #FFDFDF;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Cabeçalho fixo personalizado com logo em base64
if logo_base64:
    st.markdown(f"""
    <div class="fixed-header">
        <h2>Sistema de Previsão de Índices Contratuais</h2>
        <img class="logo" src="data:image/png;base64,{logo_base64}" alt="Logo Petrobras">
    </div>
    <div class="content"></div>
    """, unsafe_allow_html=True)
else:
    # Fallback se a logo não for encontrada
    st.markdown("""
    <div class="fixed-header">
        <h2>Sistema de Previsão de Índices Contratuais</h2>
    </div>
    <div class="content"></div>
    """, unsafe_allow_html=True)

# Função para baixar nltk stopwords
@st.cache_resource
def download_nltk_resources():
    try:
        nltk.download('stopwords', quiet=True)
        return True
    except Exception as e:
        st.error(f"Erro ao baixar stopwords do NLTK: {e}")
        return False


# Baixar recursos do NLTK na inicialização
download_nltk_resources()

# Função para criar um link de download
def get_download_link(df, filename, text):
    """
    Gera um link para download do DataFrame como arquivo Excel
    """
    # Salvar o DataFrame como um arquivo Excel em memória
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False)

    # Codificar o arquivo em base64
    b64 = base64.b64encode(output.getvalue()).decode()
    href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}">{text}</a>'
    return href

# Função para processar os dados


def process_contract_data(df, use_existing_model=False, model_dir=None):
    """
    Função principal para processar dados de contratos.

    Args:
        df (DataFrame): DataFrame pandas com os dados do contrato
        use_existing_model (bool): Se True, usa modelos existentes
        model_dir (str): Diretório onde os modelos estão salvos

    Returns:
        tuple: (DataFrame preenchido, dict com células preenchidas, estatísticas)
    """
    # Definir colunas relevantes
    features = ['Descrição Linha de Serviço',
                'Classificação', 'Fornecedor', 'Moeda', 'Item']
    targets = ['Indice1', 'Peso1', 'FatorK1', 'Indice2', 'Peso2', 'FatorK2',
               'Indice3', 'Peso3', 'FatorK3', 'Indice4', 'Peso4', 'FatorK4',
               'Indice5', 'Peso5', 'FatorK5', 'Indice6', 'Peso6', 'FatorK6']

    # Inicializar estatísticas
    stats = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "rows_for_training": 0,
        "rows_for_prediction": 0,
        "cells_filled": 0,
        "rows_modified": 0
    }

    # Verificar se as colunas necessárias existem
    missing_cols = [col for col in features + targets if col not in df.columns]
    if missing_cols:
        error_msg = f"Colunas não encontradas no DataFrame: {missing_cols}"
        st.error(error_msg)
        raise ValueError(error_msg)

    # Tratar valores nulos nas features
    for col in features:
        if col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].fillna('')
            else:
                df[col] = df[col].fillna(0)

    # Pré-processamento de texto
    try:
        stop_words = set(stopwords.words('portuguese'))
        tfidf = TfidfVectorizer(stop_words=list(stop_words), max_features=50)

        if 'Descrição Linha de Serviço' in df.columns:
            desc_tfidf = tfidf.fit_transform(
                df['Descrição Linha de Serviço']).toarray()
            desc_df = pd.DataFrame(desc_tfidf, columns=[
                                   f"tfidf_{i}" for i in range(desc_tfidf.shape[1])])
        else:
            st.error(
                "Coluna 'Descrição Linha de Serviço' não encontrada no DataFrame")
            desc_df = pd.DataFrame()
    except Exception as e:
        st.error(f"Erro ao processar TF-IDF: {e}")
        desc_df = pd.DataFrame()

    # Codificação One-Hot para variáveis categóricas
    try:
        cat_cols = [col for col in ['Classificação',
                                    'Fornecedor', 'Moeda'] if col in df.columns]
        if cat_cols:
            # Remover valores NaN ou vazios antes da codificação
            df_cat = df[cat_cols].fillna('MISSING')

            encoder = OneHotEncoder(
                sparse_output=False, handle_unknown='ignore')
            cat_encoded = encoder.fit_transform(df_cat)
            cat_df = pd.DataFrame(cat_encoded,
                                  columns=encoder.get_feature_names_out(cat_cols))
        else:
            st.warning("Nenhuma coluna categórica encontrada para codificação")
            cat_df = pd.DataFrame()
    except Exception as e:
        st.error(f"Erro na codificação One-Hot: {e}")
        cat_df = pd.DataFrame(index=range(len(df)))

    # Tratar 'Item' como numérico
    try:
        if 'Item' in df.columns:
            item_df = pd.DataFrame(df['Item'].fillna(
                0).astype(float), columns=['Item'])
        else:
            st.warning("Coluna 'Item' não encontrada")
            item_df = pd.DataFrame(index=range(len(df)))
    except Exception as e:
        st.error(f"Erro ao converter 'Item' para numérico: {e}")
        item_df = pd.DataFrame(index=range(len(df)))

    # Combinar todas as features em um único DataFrame
    X = pd.concat([desc_df, cat_df, item_df], axis=1)

    # Preparar os alvos (substituir vazios por NaN)
    y = df[targets].replace('', np.nan)

    # Identificar linhas com pelo menos um valor não-nulo nos targets (para treino)
    has_target_values = ~y.isna().all(axis=1)
    complete_rows = y[has_target_values].index
    incomplete_rows = y[~has_target_values].index

    stats["rows_for_training"] = len(complete_rows)
    stats["rows_for_prediction"] = len(incomplete_rows)

    # Separar índices e pesos para modelagem separada
    indices_cols = [col for col in targets if 'Indice' in col]
    pesos_cols = [col for col in targets if 'Peso' in col]
    fatork_cols = [col for col in targets if 'FatorK' in col]

    # Verificar se há dados suficientes para treino
    min_train_samples = 10
    if len(complete_rows) < min_train_samples and not use_existing_model:
        st.warning(
            f"⚠️ Dados insuficientes para treinamento: {len(complete_rows)} < {min_train_samples}. Os resultados podem não ser confiáveis.")

    # Separar dados de treino e teste
    X_train = X.loc[complete_rows]
    y_train = y.loc[complete_rows].fillna({'FatorK1': 1.0, 'FatorK2': 1.0, 'FatorK3': 1.0,
                                          'FatorK4': 1.0, 'FatorK5': 1.0, 'FatorK6': 1.0})
    X_pred = X.loc[incomplete_rows] if len(
        incomplete_rows) > 0 else pd.DataFrame()

    # Definir caminhos para os modelos
    clf_model_path = os.path.join(
        model_dir, "indices_classifier.joblib") if model_dir else None
    reg_model_path = os.path.join(
        model_dir, "pesos_regressor.joblib") if model_dir else None

    # Inicializar modelos
    clf, reg = None, None

    # Carregamento ou treinamento dos modelos
    with st.spinner("Carregando/treinando modelos..."):
        # Verificar se devemos usar modelos existentes
        if use_existing_model and model_dir:
            # Tentar carregar modelos existentes
            try:
                if os.path.exists(clf_model_path):
                    clf = joblib.load(clf_model_path)
                    st.success(
                        f"✅ Modelo de classificação carregado com sucesso")
                else:
                    st.warning(
                        f"⚠️ Modelo de classificação não encontrado em: {clf_model_path}")
                    use_existing_model = False

                if os.path.exists(reg_model_path):
                    reg = joblib.load(reg_model_path)
                    st.success(f"✅ Modelo de regressão carregado com sucesso")
                else:
                    st.warning(
                        f"⚠️ Modelo de regressão não encontrado em: {reg_model_path}")
                    use_existing_model = False
            except Exception as e:
                st.error(f"❌ Erro ao carregar modelos: {e}")
                use_existing_model = False

        # Se não conseguimos carregar os modelos ou não foi solicitado, treinar novos
        if not use_existing_model and len(complete_rows) >= min_train_samples:
            try:
                # Modelo para prever índices (classificação)
                indices_data = y_train[indices_cols].fillna('Nenhum')
                has_indices = (indices_data != 'Nenhum').any(axis=1)

                if has_indices.any():
                    # Mostrar progresso
                    progress_bar = st.progress(0)
                    st.info("🔄 Treinando modelo para prever índices...")

                    clf = MultiOutputClassifier(RandomForestClassifier(
                        n_estimators=100, random_state=42))
                    clf.fit(X_train[has_indices], indices_data[has_indices])

                    progress_bar.progress(50)

                    # Salvar o modelo se houver um diretório
                    if model_dir:
                        joblib.dump(clf, clf_model_path)
                        st.success(
                            f"✅ Modelo de classificação salvo em: {clf_model_path}")

                    # Modelo para prever pesos (regressão)
                    pesos_data = y_train[pesos_cols].fillna(0)
                    has_pesos = (pesos_data > 0).any(axis=1)

                    st.info("🔄 Treinando modelo para prever pesos...")

                    if has_pesos.any():
                        reg = RandomForestRegressor(
                            n_estimators=100, random_state=42)
                        reg.fit(X_train[has_pesos], pesos_data[has_pesos])

                        # Salvar o modelo se houver um diretório
                        if model_dir:
                            joblib.dump(reg, reg_model_path)
                            st.success(
                                f"✅ Modelo de regressão salvo em: {reg_model_path}")

                    progress_bar.progress(100)

                else:
                    st.warning(
                        "⚠️ Nenhum dado válido para treinar o modelo de índices")

            except Exception as e:
                st.error(f"❌ Erro no treinamento dos modelos: {e}")
                clf, reg = None, None
        elif not use_existing_model:
            st.warning("⚠️ Dados insuficientes para treinar modelos")
            clf, reg = None, None

    # Prever índices e pesos para linhas incompletas
    pred_df = pd.DataFrame()

    if len(incomplete_rows) > 0 and X_pred.shape[0] > 0:
        with st.spinner("Fazendo previsões..."):
            # Inicializar DataFrame de previsões com valores vazios
            pred_df = pd.DataFrame(index=incomplete_rows, columns=targets)

            # Prever índices se o modelo foi treinado
            if clf is not None:
                try:
                    indices_pred = clf.predict(X_pred)
                    for i, col in enumerate(indices_cols):
                        pred_df[col] = indices_pred[:, i]
                    st.success("✅ Previsão de índices concluída")
                except Exception as e:
                    st.error(f"❌ Erro na previsão de índices: {e}")
                    # Usar 'Nenhum' como fallback para índices
                    for col in indices_cols:
                        pred_df[col] = 'Nenhum'
            else:
                # Usar 'Nenhum' como fallback para índices se não há modelo
                for col in indices_cols:
                    pred_df[col] = 'Nenhum'

            # Prever pesos se o modelo foi treinado
            if reg is not None:
                try:
                    pesos_pred = reg.predict(X_pred)

                    # Normalizar pesos para soma <=
                    def normalize_weights(weights):
                        # Substituir NaN com 0 para evitar problemas
                        weights = np.nan_to_num(weights)
                        total = np.sum(weights)
                        if total > 1:
                            # Normalizar apenas valores não-nulos
                            return weights / total if total > 0 else weights
                        return weights

                    pesos_pred_normalized = np.apply_along_axis(
                        normalize_weights, 1, pesos_pred)

                    for i, col in enumerate(pesos_cols):
                        pred_df[col] = pesos_pred_normalized[:, i]
                    st.success("✅ Previsão de pesos concluída e normalizada")
                except Exception as e:
                    st.error(f"❌ Erro na previsão de pesos: {e}")
                    # Usar 0 como fallback para pesos
                    for col in pesos_cols:
                        pred_df[col] = 0
            else:
                # Usar 0 como fallback para pesos se não há modelo
                for col in pesos_cols:
                    pred_df[col] = 0

            # Garantir consistência entre índices e pesos
            for i in range(1, 7):  # Para cada par índice/peso (1 a 6)
                indice_col = f'Indice{i}'
                peso_col = f'Peso{i}'

                if indice_col in pred_df.columns and peso_col in pred_df.columns:
                    # Caso 1: Peso > 0 mas sem índice - atribuir índice padrão "IGP"
                    mask = (pred_df[peso_col] > 0) & (
                        (pred_df[indice_col] == '') | (pred_df[indice_col] == 'Nenhum'))
                    pred_df.loc[mask, indice_col] = "IGP"  # Índice padrão

                    # Caso 2: Índice vazio ou 'Nenhum', zerar o peso correspondente
                    mask = ((pred_df[indice_col] == '') | (
                        pred_df[indice_col] == 'Nenhum'))
                    pred_df.loc[mask, peso_col] = 0

            # Verificar novamente se a soma dos pesos é <= 1 após as correções
            for idx in pred_df.index:
                peso_values = [
                    pred_df.loc[idx, f'Peso{i}'] for i in range(1, 7)]
                peso_values = [p for p in peso_values if not pd.isna(p)]
                total = sum(peso_values)

                if total > 1:
                    for i in range(1, 7):
                        peso_col = f'Peso{i}'
                        if peso_col in pred_df.columns and not pd.isna(pred_df.loc[idx, peso_col]) and pred_df.loc[idx, peso_col] > 0:
                            pred_df.loc[idx, peso_col] = pred_df.loc[idx,
                                                                     peso_col] / total

            # Definir FatorK como 1.0 para todas as linhas
            for col in fatork_cols:
                pred_df[col] = 1.0

    # Copiar o DataFrame original
    df_filled = df.copy()

    # Flag para verificar se houve previsões aplicadas
    predictions_applied = False

    # Registrar quais células foram preenchidas (para colorir no Excel depois)
    # Dicionário para armazenar posições (linha, coluna) das células preenchidas
    filled_cells = {}

    # Atualizar apenas as linhas incompletas com as previsões (se houver)
    if not pred_df.empty:
        with st.spinner("Aplicando previsões aos dados originais..."):
            for idx in pred_df.index:
                # Primeiro verificar se precisamos aplicar qualquer previsão nesta linha
                any_missing = False
                for col in targets:
                    is_empty = pd.isna(
                        df_filled.loc[idx, col]) or df_filled.loc[idx, col] == ''
                    if is_empty:
                        any_missing = True
                        break

                if any_missing:
                    # Primeiro preenchimento: encontrar índices válidos e seus pesos correspondentes
                    valid_indices_and_weights = []

                    # Verificar valores existentes (não nulos e não vazios) nos dados originais
                    for i in range(1, 7):
                        indice_col = f'Indice{i}'
                        peso_col = f'Peso{i}'
                        indice_value = df_filled.loc[idx, indice_col]
                        peso_value = df_filled.loc[idx, peso_col]

                        indice_filled = not (
                            pd.isna(indice_value) or indice_value == '' or indice_value == 'Nenhum')
                        peso_filled = not (
                            pd.isna(peso_value) or peso_value == 0)

                        # Adicionar à lista se já estiver preenchido
                        if indice_filled or peso_filled:
                            valid_indices_and_weights.append(
                                (indice_col, peso_col))

                    # Agora, preencher os valores ausentes
                    for i in range(1, 7):
                        indice_col = f'Indice{i}'
                        peso_col = f'Peso{i}'
                        fatork_col = f'FatorK{i}'

                        # Verificar se o par de índice e peso é válido para atualizar
                        if (indice_col, peso_col) not in valid_indices_and_weights:
                            # Verificar se existem valores nas previsões
                            indice_pred = pred_df.loc[idx,
                                                      indice_col] if indice_col in pred_df.columns else None
                            peso_pred = pred_df.loc[idx,
                                                    peso_col] if peso_col in pred_df.columns else None

                            # Só preencher se tiver previsão com valor não nulo/vazio
                            valid_indice = indice_pred is not None and indice_pred != '' and indice_pred != 'Nenhum'
                            valid_peso = peso_pred is not None and peso_pred > 0

                            # Aplicar previsões apenas para pares válidos (índice e peso)
                            if valid_indice and valid_peso:
                                # Preencher índice
                                is_empty_indice = pd.isna(
                                    df_filled.loc[idx, indice_col]) or df_filled.loc[idx, indice_col] == ''
                                if is_empty_indice:
                                    if idx not in filled_cells:
                                        filled_cells[idx] = []
                                    filled_cells[idx].append(indice_col)
                                    df_filled.loc[idx,
                                                  indice_col] = indice_pred
                                    predictions_applied = True
                                    stats["cells_filled"] += 1

                                # Preencher peso
                                is_empty_peso = pd.isna(
                                    df_filled.loc[idx, peso_col]) or df_filled.loc[idx, peso_col] == 0
                                if is_empty_peso:
                                    if idx not in filled_cells:
                                        filled_cells[idx] = []
                                    filled_cells[idx].append(peso_col)
                                    df_filled.loc[idx, peso_col] = peso_pred
                                    predictions_applied = True
                                    stats["cells_filled"] += 1

                                # Preencher fatorK se necessário
                                is_empty_fatork = pd.isna(
                                    df_filled.loc[idx, fatork_col])
                                if is_empty_fatork:
                                    if idx not in filled_cells:
                                        filled_cells[idx] = []
                                    filled_cells[idx].append(fatork_col)
                                    df_filled.loc[idx, fatork_col] = 1.0
                                    predictions_applied = True
                                    stats["cells_filled"] += 1

            # Garantir que os valores sejam consistentes (ex.: substituir 'Nenhum' por vazio nos índices)
            for col in indices_cols:
                if col in df_filled.columns:
                    mask = df_filled[col] == 'Nenhum'
                    if mask.any():
                        df_filled.loc[mask, col] = ''

            stats["rows_modified"] = len(filled_cells)

    return df_filled, filled_cells, stats

# Função para criar um diretório temporário


def create_temp_dir():
    """Cria um diretório temporário para armazenar modelos e arquivos"""
    temp_dir = tempfile.mkdtemp()
    model_dir = os.path.join(temp_dir, "models")
    os.makedirs(model_dir, exist_ok=True)
    return temp_dir, model_dir

# Função para gerar visualizações


def generate_visualizations(df, filled_cells=None):
    """
    Gera visualizações para análise dos dados

    Args:
        df (DataFrame): DataFrame processado
        filled_cells (dict): Dicionário com células preenchidas pelo modelo
    """
    st.markdown('<h2 class="sub-header">Visualizações</h2>',
                unsafe_allow_html=True)

    # Criar abas para diferentes visualizações
    tab1, tab2, tab3 = st.tabs(
        ["Distribuição de Índices", "Distribuição de Pesos", "Mapa de Preenchimento"])

    with tab1:
        # Contar ocorrências de cada índice e visualizar
        indices_cols = [col for col in df.columns if 'Indice' in col]
        if indices_cols:
            # Juntar todos os índices em uma série
            all_indices = pd.Series()
            for col in indices_cols:
                series = df[col].dropna()
                series = series[series != '']  # Remover valores vazios
                all_indices = pd.concat([all_indices, series])

            if not all_indices.empty:
                # Contar ocorrências
                indices_count = all_indices.value_counts().reset_index()
                indices_count.columns = ['Índice', 'Contagem']

                # Filtrar apenas os 20 índices mais comuns
                top_indices = indices_count.head(20)

                # Criar gráfico
                fig, ax = plt.subplots(figsize=(12, 6))
                sns.barplot(data=top_indices, x='Índice', y='Contagem', ax=ax)
                ax.set_title('Distribuição dos 20 Índices Mais Comuns')
                ax.set_xticklabels(ax.get_xticklabels(),
                                   rotation=45, ha='right')
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.info(
                    "Não há dados suficientes para visualizar a distribuição de índices.")

    with tab2:
        # Analisar a distribuição dos pesos
        pesos_cols = [col for col in df.columns if 'Peso' in col]
        if pesos_cols:
            # Calcular estatísticas para cada coluna de peso
            pesos_stats = df[pesos_cols].describe().T

            # Criar gráfico de distribuição dos pesos
            fig, ax = plt.subplots(figsize=(12, 6))

            # Usar boxplot para visualizar a distribuição
            sns.boxplot(data=df[pesos_cols], ax=ax)
            ax.set_title('Distribuição dos Pesos por Posição')
            ax.set_ylabel('Valor do Peso')
            ax.set_xlabel('Posição do Peso')
            plt.tight_layout()
            st.pyplot(fig)

            # Mostrar estatísticas
            st.write("Estatísticas dos Pesos:")
            st.dataframe(pesos_stats)
        else:
            st.info("Não há dados de pesos para visualizar.")

    with tab3:
        # Visualizar quais células foram preenchidas
        if filled_cells and filled_cells:
            # Criar um DataFrame para visualizar o padrão de preenchimento
            fill_pattern = pd.DataFrame(
                0, index=range(len(df)), columns=df.columns)

            for idx, cols in filled_cells.items():
                for col in cols:
                    row_idx = df.index.get_loc(idx)
                    fill_pattern.loc[row_idx, col] = 1

            # Calcular a proporção de células preenchidas para cada coluna
            fill_summary = fill_pattern.sum() / len(df)

            # Filtrar apenas colunas relevantes (Indice, Peso, FatorK)
            relevant_cols = [col for col in fill_summary.index if (
                'Indice' in col) or ('Peso' in col) or ('FatorK' in col)]
            fill_relevant = fill_summary[relevant_cols]

            # Criar gráfico
            fig, ax = plt.subplots(figsize=(14, 6))
            sns.barplot(x=fill_relevant.index, y=fill_relevant.values, ax=ax)
            ax.set_title('Proporção de Células Preenchidas por Coluna')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            ax.set_ylabel('Proporção de Células Preenchidas')
            plt.tight_layout()
            st.pyplot(fig)

            # Mostrar também uma tabela com o número de células preenchidas
            cells_filled = fill_pattern.sum().loc[relevant_cols]
            cells_filled_df = pd.DataFrame({
                'Coluna': cells_filled.index,
                'Células Preenchidas': cells_filled.values
            }).sort_values('Células Preenchidas', ascending=False)

            st.write("Número de Células Preenchidas por Coluna:")
            st.dataframe(cells_filled_df)
        else:
            st.info("Nenhuma célula foi preenchida pelo modelo neste processamento.")

# Função principal da aplicação Streamlit


def main():
    # Adicionar informação na barra lateral
    st.sidebar.markdown(
        '<h2 class="section-header">Configurações</h2>', unsafe_allow_html=True)

    # Opção para carregar arquivo
    st.sidebar.markdown('<h3>Carregar Arquivo</h3>', unsafe_allow_html=True)
    uploaded_file = st.sidebar.file_uploader(
        "Selecione um arquivo Excel para processamento", type=["xlsx", "xls"])

    # Opção para usar modelos existentes
    st.sidebar.markdown('<h3>Modelos</h3>', unsafe_allow_html=True)
    use_existing_model = st.sidebar.checkbox(
        "Usar modelos pré-treinados (se disponíveis)", value=True)

    # Opção para carregar modelos personalizados
    custom_model_file = st.sidebar.file_uploader(
        "Carregar modelo de classificação personalizado", type=["joblib"])

    # Criar diretórios temporários para modelos e arquivos
    temp_dir, model_dir = create_temp_dir()

    if custom_model_file is not None:
        # Salvar o modelo carregado no diretório temporário
        with open(os.path.join(model_dir, "indices_classifier.joblib"), 'wb') as f:
            f.write(custom_model_file.getbuffer())
        st.sidebar.success(
            "✅ Modelo de classificação personalizado carregado!")

    custom_reg_file = st.sidebar.file_uploader(
        "Carregar modelo de regressão personalizado", type=["joblib"])
    if custom_reg_file is not None:
        # Salvar o modelo carregado no diretório temporário
        with open(os.path.join(model_dir, "pesos_regressor.joblib"), 'wb') as f:
            f.write(custom_reg_file.getbuffer())
        st.sidebar.success("✅ Modelo de regressão personalizado carregado!")

    # Mostrar instruções quando nenhum arquivo é carregado
    if uploaded_file is None:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("""
        ## Bem-vindo ao Sistema de Previsão de Índices Contratuais (SPIC)
        
        Este aplicativo ajuda a preencher fórmulas de reajuste em contratos, prevendo índices econômicos e seus respectivos pesos.
        
        ### Como usar:
        
        1. **Carregue um arquivo Excel** usando o seletor na barra lateral
        2. Escolha se deseja usar modelos pré-treinados ou treinar novos modelos
        3. Visualize os resultados e estatísticas
        4. Faça download do arquivo processado
        
        ### Requisitos do arquivo:
        
        O arquivo Excel deve conter as seguintes colunas:
        - Descrição Linha de Serviço
        - Classificação
        - Fornecedor
        - Moeda
        - Item
        - Colunas para índices: Indice1, Indice2, etc.
        - Colunas para pesos: Peso1, Peso2, etc.
        - Colunas para fatores: FatorK1, FatorK2, etc.
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        return

    # Processar o arquivo carregado
    try:
        # Carregar o DataFrame
        df = pd.read_excel(uploaded_file)

        # Mostrar informações básicas sobre o arquivo
        st.markdown(
            '<h2 class="sub-header">Informações do Arquivo</h2>', unsafe_allow_html=True)
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.write(f"**Nome do arquivo:** {uploaded_file.name}")
        st.write(f"**Tamanho do arquivo:** {uploaded_file.size / 1024:.2f} KB")
        st.write(f"**Total de linhas:** {df.shape[0]}")
        st.write(f"**Total de colunas:** {df.shape[1]}")
        st.markdown('</div>', unsafe_allow_html=True)

        # Botão para iniciar o processamento
        if st.button("Iniciar Processamento"):
            # Mostrar uma barra de progresso
            progress_bar = st.progress(0)

            # Processar os dados
            with st.spinner("Processando dados..."):
                df_filled, filled_cells, stats = process_contract_data(
                    df,
                    use_existing_model=use_existing_model,
                    model_dir=model_dir
                )
                progress_bar.progress(100)

            # Mostrar estatísticas do processamento
            st.markdown(
                '<h2 class="sub-header">Estatísticas do Processamento</h2>', unsafe_allow_html=True)
            st.markdown('<div class="success-box">', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total de Linhas", stats["total_rows"])
                st.metric("Linhas para Treinamento",
                          stats["rows_for_training"])
                st.metric("Linhas Modificadas", stats["rows_modified"])

            with col2:
                st.metric("Total de Colunas", stats["total_columns"])
                st.metric("Linhas para Previsão", stats["rows_for_prediction"])
                st.metric("Células Preenchidas", stats["cells_filled"])
            st.markdown('</div>', unsafe_allow_html=True)

            # Visualizar uma amostra dos dados preenchidos
            st.markdown(
                '<h2 class="sub-header">Amostra dos Dados Processados</h2>', unsafe_allow_html=True)

            # Criar abas para diferentes visualizações dos dados
            tab1, tab2, tab3 = st.tabs(
                ["Dados Processados", "Células Preenchidas", "Dados Originais"])

            with tab1:
                # Mostrar os primeiros 50 registros
                st.dataframe(df_filled.head(50))

            with tab2:
                # Mostrar apenas as linhas que foram modificadas
                if filled_cells:
                    modified_rows = pd.DataFrame(index=filled_cells.keys())
                    modified_data = df_filled.loc[modified_rows.index]
                    st.dataframe(modified_data)
                else:
                    st.info("Nenhuma célula foi preenchida pelo modelo.")

            with tab3:
                # Mostrar os dados originais
                st.dataframe(df.head(50))

            # Gerar visualizações
            generate_visualizations(df_filled, filled_cells)

            # Opções de download
            st.markdown(
                '<h2 class="sub-header">Download dos Resultados</h2>', unsafe_allow_html=True)

            # Gerar o nome do arquivo de saída
            file_name = uploaded_file.name.split('.')[0]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file_name = f"{file_name}_processado_{timestamp}.xlsx"

            # Criar link para download do arquivo processado
            download_link = get_download_link(
                df_filled, output_file_name, "📥 Baixar arquivo processado")
            st.markdown(download_link, unsafe_allow_html=True)

            # Salvar modelos para download
            if os.path.exists(os.path.join(model_dir, "indices_classifier.joblib")) and \
               os.path.exists(os.path.join(model_dir, "pesos_regressor.joblib")):

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("### Download dos Modelos Treinados")
                st.markdown(
                    "Estes modelos podem ser utilizados em processamentos futuros.")

                # Criar links para download dos modelos
                with open(os.path.join(model_dir, "indices_classifier.joblib"), "rb") as f:
                    clf_bytes = f.read()
                    clf_b64 = base64.b64encode(clf_bytes).decode()
                    clf_href = f'<a href="data:application/octet-stream;base64,{clf_b64}" download="indices_classifier.joblib">📥 Baixar modelo de classificação de índices</a>'
                    st.markdown(clf_href, unsafe_allow_html=True)

                with open(os.path.join(model_dir, "pesos_regressor.joblib"), "rb") as f:
                    reg_bytes = f.read()
                    reg_b64 = base64.b64encode(reg_bytes).decode()
                    reg_href = f'<a href="data:application/octet-stream;base64,{reg_b64}" download="pesos_regressor.joblib">📥 Baixar modelo de regressão de pesos</a>'
                    st.markdown(reg_href, unsafe_allow_html=True)

        else:
            # Mostrar instruções de uso quando o arquivo é carregado mas o processamento não iniciou
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            st.markdown("""
            ## Arquivo carregado com sucesso!
            
            Clique no botão "Iniciar Processamento" para analisar e preencher os índices e pesos faltantes.
            
            * Se você selecionou a opção "Usar modelos pré-treinados", o sistema tentará carregar modelos existentes.
            * Caso contrário, novos modelos serão treinados utilizando os dados do arquivo atual.
            """)
            st.markdown('</div>', unsafe_allow_html=True)

            # Mostrar uma prévia dos dados
            st.markdown('<h2 class="sub-header">Prévia dos Dados</h2>',
                        unsafe_allow_html=True)
            st.dataframe(df.head(10))

            # Verificar colunas obrigatórias
            required_cols = ['Descrição Linha de Serviço', 'Classificação', 'Fornecedor', 'Moeda', 'Item',
                             'Indice1', 'Peso1', 'FatorK1']
            missing_cols = [
                col for col in required_cols if col not in df.columns]

            if missing_cols:
                st.markdown('<div class="warning-box">',
                            unsafe_allow_html=True)
                st.warning(
                    f"⚠️ Atenção! O arquivo não possui todas as colunas obrigatórias. Faltam: {', '.join(missing_cols)}")
                st.markdown('</div>', unsafe_allow_html=True)

    except Exception as e:
        st.markdown('<div class="error-box">', unsafe_allow_html=True)
        st.error(f"❌ Erro ao processar o arquivo: {e}")
        st.markdown('</div>', unsafe_allow_html=True)


# Executar a aplicação
if __name__ == "__main__":
    main()
