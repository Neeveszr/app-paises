<div align="center">

# 🌍 Classificação de Países por Prioridade de Ajuda

Aplicativo web que agrupa países a partir de indicadores socioeconômicos e de saúde, usando aprendizado de máquina **não supervisionado**.

![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.67-FF4B4B?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9-F7931E?logo=scikitlearn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-3.1-150458?logo=pandas&logoColor=white)

</div>

---

## 📌 Sobre o projeto

O usuário informa nove indicadores de um país e o modelo devolve o **grupo** em que ele se encaixa, indicando o nível de prioridade de ajuda estimado.

Projeto desenvolvido para a disciplina de **Inteligência Artificial · Unidade III**, no curso de Engenharia de Software da **UNIFSA** (2026.2).

## 🧠 Como funciona

O modelo é um *pipeline* do scikit-learn com três etapas:

```
Indicadores  →  StandardScaler  →  PCA  →  KMeans  →  Grupo
```

| Etapa | Função |
|---|---|
| **StandardScaler** | Coloca todos os indicadores na mesma escala |
| **PCA** | Reduz a dimensionalidade, mantendo a maior parte da variância |
| **KMeans** | Agrupa os países em clusters por similaridade |

O arquivo `modelo_paises.pkl` guarda o pipeline treinado, a ordem das colunas usada no treinamento e o nome de cada grupo.

## 📊 Indicadores de entrada

| Campo | Descrição |
|---|---|
| `child_mort` | Mortalidade infantil (mortes por 1.000 nascidos vivos) |
| `exports` | Exportações (% do PIB per capita) |
| `health` | Gasto com saúde (% do PIB per capita) |
| `imports` | Importações (% do PIB per capita) |
| `income` | Renda líquida por pessoa (US$) |
| `inflation` | Inflação anual (%) |
| `life_expec` | Expectativa de vida (anos) |
| `total_fer` | Filhos por mulher |
| `gdpp` | PIB per capita (US$) |

Os valores iniciais dos campos na interface são as medianas da base de treino.

## 🚀 Como rodar localmente

**1. Clone o repositório**

```bash
git clone https://github.com/Neeveszr/app-paises.git
cd app-paises
```

**2. Crie e ative um ambiente virtual**

```powershell
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1
```

```bash
# Mac / Linux
python -m venv .venv
source .venv/bin/activate
```

**3. Instale as dependências**

```bash
pip install -r requirements.txt
```

**4. Execute o app**

```bash
streamlit run app.py
```

O app abre em `http://localhost:8501`.

## 📁 Estrutura do repositório

```
app-paises/
├── app.py               # Interface Streamlit
├── modelo_paises.pkl    # Pipeline treinado + colunas + nomes dos grupos
├── requirements.txt     # Dependências com versões fixas
└── README.md
```

## ⚠️ Observações

- O modelo foi treinado com **scikit-learn 1.9.1**. Use a mesma versão para evitar avisos ou erros ao carregar o `.pkl`.
- Os grupos vêm de um algoritmo de agrupamento: o resultado é uma estimativa por similaridade entre países, não um diagnóstico definitivo.
- Arquivos `.pkl` executam código ao serem carregados. Carregue apenas arquivos de fonte confiável.

## 🛠️ Tecnologias

- [Python](https://www.python.org/)
- [Streamlit](https://streamlit.io/)
- [scikit-learn](https://scikit-learn.org/)
- [pandas](https://pandas.pydata.org/) e [NumPy](https://numpy.org/)
- [joblib](https://joblib.readthedocs.io/)

## 👤 Autor

**Neeveszr** · [GitHub](https://github.com/Neeveszr)
