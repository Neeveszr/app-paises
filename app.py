"""Aplicativo de classificação de países por prioridade de ajuda.

Inteligência Artificial · Unidade III · UNIFSA · Engenharia de Software · 2026.2
"""
import altair as alt
import joblib
import pandas as pd
import streamlit as st

# Informações do model card que não estão dentro do arquivo do modelo
FONTE_DADOS = "Country-data (HELP International), disponível no Kaggle"
DATA_TREINAMENTO = "11/11/2026"

# 1. Configuração da página
st.set_page_config(page_title="Classificação de países", layout="centered")


# 2. Carregamento do modelo (executado uma única vez e mantido em cache)
@st.cache_resource
def carregar_modelo():
    return joblib.load("modelo_paises.pkl")


pacote = carregar_modelo()
modelo = pacote["modelo"]      # pipeline: StandardScaler -> PCA -> KMeans
colunas = pacote["colunas"]    # ordem das colunas usada no treinamento
nomes = pacote["nomes"]        # nome de cada grupo

escalonador = modelo.named_steps["standardscaler"]
pca = modelo.named_steps["pca"]
kmeans = modelo.named_steps["kmeans"]

# 3. Descrição de cada campo: rótulo exibido e valor inicial (mediana da base)
CAMPOS = {
    "child_mort": ("Mortalidade infantil (mortes por 1.000 nascidos vivos)", 19.3),
    "exports": ("Exportações (% do PIB per capita)", 35.0),
    "health": ("Gasto com saúde (% do PIB per capita)", 6.3),
    "imports": ("Importações (% do PIB per capita)", 43.3),
    "income": ("Renda líquida por pessoa (US$)", 9960.0),
    "inflation": ("Inflação anual (%)", 5.4),
    "life_expec": ("Expectativa de vida (anos)", 73.1),
    "total_fer": ("Filhos por mulher", 2.4),
    "gdpp": ("PIB per capita (US$)", 4660.0),
}

# 4. Interface
st.title("Classificação de países")
st.write("Informe os indicadores do país para obter o nível de prioridade estimado pelo modelo.")

valores = {}
esquerda, direita = st.columns(2)
for i, coluna in enumerate(colunas):
    rotulo, inicial = CAMPOS[coluna]
    with esquerda if i % 2 == 0 else direita:
        valores[coluna] = st.number_input(rotulo, value=inicial, key=coluna)

# 5. Classificação
if st.button("Classificar", type="primary"):
    entrada = pd.DataFrame([valores])[colunas]   # uma linha, colunas na ordem do treinamento
    grupo = int(modelo.predict(entrada)[0])      # o pipeline padroniza, aplica o PCA e agrupa
    st.success(f"Grupo {grupo}: {nomes[grupo]}")

    # 5.1 Comparação com o perfil típico do grupo: o centro do KMeans, convertido de volta
    # para a escala original (desfaz o PCA e a padronização). É uma aproximação da média.
    centros = escalonador.inverse_transform(pca.inverse_transform(kmeans.cluster_centers_))
    media_grupo = centros[grupo]
    informado = entrada.iloc[0].to_numpy()

    st.subheader("Comparação com o perfil do grupo")
    st.dataframe(
        pd.DataFrame(
            {
                "País informado": informado,
                f"Perfil do grupo ({nomes[grupo]})": media_grupo,
                "Diferença (%)": (informado / media_grupo - 1) * 100,
            },
            index=[CAMPOS[c][0] for c in colunas],
        ).round(2)
    )
    st.caption("O perfil do grupo é aproximado: vem do centro do grupo no KMeans.")

    # 5.2 Posição no plano PC1 x PC2 (só os dois primeiros dos cinco componentes)
    pc = modelo[:-1].transform(entrada)[0, :2]
    pontos = pd.DataFrame(kmeans.cluster_centers_[:, :2], columns=["PC1", "PC2"])
    pontos["Ponto"] = [f"Centro: {nomes[i]}" for i in range(len(pontos))]
    pontos.loc[len(pontos)] = [pc[0], pc[1], "País informado"]

    variancia = pca.explained_variance_ratio_
    st.subheader("Posição no plano PC1 × PC2")
    st.altair_chart(
        alt.Chart(pontos)
        .mark_point(size=250, filled=True, opacity=0.8)
        .encode(
            x=alt.X("PC1", title=f"PC1 ({variancia[0]:.1%} da variância)"),
            y=alt.Y("PC2", title=f"PC2 ({variancia[1]:.1%} da variância)"),
            color=alt.Color("Ponto", title=None),
            tooltip=["Ponto", "PC1", "PC2"],
        ),
        width="stretch",
    )

# 6. Sobre o modelo (versão inicial do model card)
st.divider()
st.header("Sobre o modelo")
st.markdown(
    f"""
- **Origem dos dados:** {FONTE_DADOS}
- **Países no treinamento:** {int(escalonador.n_samples_seen_)}
- **Data de treinamento:** {DATA_TREINAMENTO}
- **Componentes do PCA:** {pca.n_components_} (variância explicada: {pca.explained_variance_ratio_.sum():.1%})
- **Número de grupos:** {kmeans.n_clusters} ({", ".join(nomes.values())})
"""
)
