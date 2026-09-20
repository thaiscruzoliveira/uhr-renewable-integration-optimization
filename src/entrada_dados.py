"""
entrada_dados.py

MÓDULO 1 - LEITURA E TRATAMENTO FINAL DOS DADOS DE ENTRADA

Sistema:
    UHEs:
        - Três Marias
        - Sobradinho
        - Itaparica / Luiz Gonzaga
        - Complexo Paulo Afonso
        - Xingó

    UTEs:
        - Prosperidade I
        - Suape II

    Renováveis:
        - Equivalente eólico do Nordeste
        - Equivalente solar do Nordeste

    Armazenamento:
        - UHR hipotética de circuito fechado

Período:
    05/09/2026 00:00 a 11/09/2026 23:00
    168 períodos horários

Arquivos esperados em "Arquivos de Entrada":
    - CURVA_CARGA_2026.csv
    - CVU_USINA_TERMICA_2026.csv
    - CAPACIDADE_GERACAO.csv
    - FATOR_CAPACIDADE-2_2026_09.csv
    - DADOS_HIDROLOGICOS_HO_2026_09.csv
    - DADOS_UHE_UHR.xlsx

Saída:
    Arquivos de Saida/entradas_processadas.xlsx
"""

from pathlib import Path
import unicodedata
import warnings

import numpy as np
import pandas as pd


# ============================================================
# 1. CONFIGURAÇÕES GERAIS
# ============================================================

PASTA_ENTRADA = Path("Arquivos de Entrada")
PASTA_SAIDA = Path("Arquivos de Saida")

PASTA_SAIDA.mkdir(
    parents=True,
    exist_ok=True
)

ARQ_CARGA = PASTA_ENTRADA / "CURVA_CARGA_2026.csv"

ARQ_CVU = (
    PASTA_ENTRADA /
    "CVU_USINA_TERMICA_2026.csv"
)

ARQ_CAPACIDADE = (
    PASTA_ENTRADA /
    "CAPACIDADE_GERACAO.csv"
)

ARQ_RENOVAVEIS = (
    PASTA_ENTRADA /
    "FATOR_CAPACIDADE-2_2026_09.csv"
)

ARQ_HIDRO = (
    PASTA_ENTRADA /
    "DADOS_HIDROLOGICOS_HO_2026_09.csv"
)

ARQ_UHE_UHR = (
    PASTA_ENTRADA /
    "DADOS_UHE_UHR.xlsx"
)

ARQ_SAIDA = (
    PASTA_SAIDA /
    "entradas_processadas.xlsx"
)


# ============================================================
# 2. PERÍODO DO ESTUDO
# ============================================================

DATA_INICIO = pd.Timestamp(
    "2026-09-05 00:00:00"
)

DATA_FIM = pd.Timestamp(
    "2026-09-11 23:00:00"
)

NUM_HORAS = 168

# Hipóteses finais do estudo
POTENCIA_UHR_ADOTADA_MW = 800.0
QUANTIL_DEFLUENCIA_MINIMA = 0.10  # piso diário = percentil 10% da defluência observada
QUANTIL_TURBINAMENTO_MINIMO = 0.10  # piso diário = percentil 10% da vazão turbinada observada


# ============================================================
# 3. USINAS DO MODELO
# ============================================================

UHE_MODELO = [
    "TRES MARIAS",
    "SOBRADINHO",
    "ITAPARICA",
    "PAULO AFONSO",
    "XINGO",
]

UTE_MODELO = [
    "PROSPERIDADE I",
    "SUAPE II",
]


# ============================================================
# 4. MAPEAMENTOS DE NOMES
# ============================================================

# ------------------------------------------------------------
# CAPACIDADE_GERACAO - UHE
# ------------------------------------------------------------

MAPA_CAPACIDADE_UHE = {

    "TRES MARIAS": [
        "TRÊS MARIAS"
    ],

    "SOBRADINHO": [
        "SOBRADINHO"
    ],

    "ITAPARICA": [
        "LUIZ GONZAGA"
    ],

    "PAULO AFONSO": [
        "PAULO AFONSO I",
        "PAULO AFONSO II",
        "PAULO AFONSO III",
        "PAULO AFONSO IV",
    ],

    "XINGO": [
        "XINGÓ"
    ],
}


# ------------------------------------------------------------
# DADOS_HIDROLOGICOS
# ------------------------------------------------------------

MAPA_HIDRO = {

    "TRES MARIAS": [
        "TRÊS MARIAS"
    ],

    "SOBRADINHO": [
        "SOBRADINHO"
    ],

    "ITAPARICA": [
        "LUIZ GONZAGA"
    ],

    "PAULO AFONSO": [
        "P. AFONSO 4",
    ],

    "XINGO": [
        "XINGO"
    ],
}


# ------------------------------------------------------------
# CAPACIDADE_GERACAO - UTE
# ------------------------------------------------------------

MAPA_CAPACIDADE_UTE = {

    "PROSPERIDADE I": [
        "PROSPERIDADE I"
    ],

    "SUAPE II": [
        "SUAPE II"
    ],
}


# ------------------------------------------------------------
# CVU - UTE
# ------------------------------------------------------------

MAPA_CVU_UTE = {

    "PROSPERIDADE I": [
        "PROSPERIDA"
    ],

    "SUAPE II": [
        "SUAPE II"
    ],
}


# ============================================================
# 5. FUNÇÕES AUXILIARES
# ============================================================

def normalizar_texto(texto):
    """
    Remove acentos, espaços extras e converte para maiúsculas.
    """

    if pd.isna(texto):
        return ""

    texto = str(texto).strip().upper()

    texto = unicodedata.normalize(
        "NFKD",
        texto
    )

    texto = "".join(
        c
        for c in texto
        if not unicodedata.combining(c)
    )

    return texto


def ler_csv(caminho):
    """
    Lê CSV tentando detectar automaticamente separador
    e codificação.
    """

    if not caminho.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado:\n{caminho}"
        )

    try:
        return pd.read_csv(
            caminho,
            sep=None,
            engine="python",
            encoding="utf-8"
        )

    except UnicodeDecodeError:
        return pd.read_csv(
            caminho,
            sep=None,
            engine="python",
            encoding="latin1"
        )


def verificar_colunas(
    df,
    colunas,
    nome_base
):
    """
    Verifica a existência das colunas necessárias.
    """

    faltantes = [
        col
        for col in colunas
        if col not in df.columns
    ]

    if faltantes:
        raise ValueError(
            f"\nBase {nome_base}: "
            f"colunas ausentes:\n{faltantes}"
        )


def converter_numerico(serie):
    """
    Converte uma série para numérico.

    Também aceita números escritos com vírgula decimal,
    caso o Excel tenha sido importado como texto.
    """

    if pd.api.types.is_numeric_dtype(serie):
        return pd.to_numeric(
            serie,
            errors="coerce"
        )

    return pd.to_numeric(
        serie.astype(str)
        .str.strip()
        .str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False),
        errors="coerce"
    )


def criar_indice_horario():
    """
    Cria o índice oficial de 168 horas do estudo.
    """

    datas = pd.date_range(
        start=DATA_INICIO,
        end=DATA_FIM,
        freq="h"
    )

    if len(datas) != NUM_HORAS:
        raise ValueError(
            "O período selecionado não possui "
            "168 horas."
        )

    return pd.DataFrame({
        "hora": np.arange(
            1,
            NUM_HORAS + 1
        ),
        "data_hora": datas
    })


# ============================================================
# 6. LEITURA DA PLANILHA DADOS_UHE_UHR
# ============================================================

def processar_dados_uhe_uhr():
    """
    Lê a planilha DADOS_UHE_UHR.xlsx.

    Estrutura esperada:

    Elemento
    Nome
    Q tur max (m³/s)
    Produtibilidade (MW/(m³/s))
    Vmin (hm³)
    Vmax (hm³)
    Pmax geração (MW)
    Pmax bombeamento (MW)
    Eficiência bombeamento
    Eficiência turbinamento
    Emax (MWh)
    Einicial (MWh)
    Efinal (MWh)
    Custo (R$/MWh)
    """

    print(
        "Processando cadastro das UHEs e UHR..."
    )

    if not ARQ_UHE_UHR.exists():
        raise FileNotFoundError(
            f"Planilha não encontrada:\n"
            f"{ARQ_UHE_UHR}"
        )

    df = pd.read_excel(
        ARQ_UHE_UHR,
        sheet_name=0
    )

    # Remove colunas completamente vazias
    df = df.dropna(
        axis=1,
        how="all"
    )

    # Remove linhas completamente vazias
    df = df.dropna(
        axis=0,
        how="all"
    ).reset_index(
        drop=True
    )

    colunas_necessarias = [
        "Elemento",
        "Nome",
        "Q tur max (m³/s)",
        "Produtibilidade (MW/(m³/s))",
        "Vmin (hm³)",
        "Vmax (hm³)",
        "Pmax geração (MW)",
        "Pmax bombeamento (MW)",
        "Eficiência bombeamento",
        "Eficiência turbinamento",
        "Emax (MWh)",
        "Einicial (MWh)",
        "Efinal (MWh)",
        "Custo (R$/MWh)",
    ]

    verificar_colunas(
        df,
        colunas_necessarias,
        "DADOS_UHE_UHR"
    )

    # --------------------------------------------------------
    # Padronização dos nomes
    # --------------------------------------------------------

    df["Elemento"] = (
        df["Elemento"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    df["Nome"] = (
        df["Nome"]
        .astype(str)
        .str.strip()
    )

    df["nome_modelo"] = (
        df["Nome"]
        .apply(normalizar_texto)
    )

    # --------------------------------------------------------
    # Converte parâmetros numéricos
    # --------------------------------------------------------

    colunas_numericas = [
        "Q tur max (m³/s)",
        "Produtibilidade (MW/(m³/s))",
        "Vmin (hm³)",
        "Vmax (hm³)",
        "Pmax geração (MW)",
        "Pmax bombeamento (MW)",
        "Eficiência bombeamento",
        "Eficiência turbinamento",
        "Emax (MWh)",
        "Einicial (MWh)",
        "Efinal (MWh)",
        "Custo (R$/MWh)",
    ]

    for coluna in colunas_numericas:
        df[coluna] = converter_numerico(
            df[coluna]
        )

    # --------------------------------------------------------
    # Separa UHE e UHR
    # --------------------------------------------------------

    uhe = df[
        df["Elemento"] == "UHE"
    ].copy()

    uhr = df[
        df["Elemento"] == "UHR"
    ].copy()

    # --------------------------------------------------------
    # Verificações das UHEs
    # --------------------------------------------------------

    if len(uhe) != 5:
        raise ValueError(
            f"Esperadas 5 UHEs na planilha, "
            f"mas foram encontradas {len(uhe)}."
        )

    nomes_uhe = set(
        uhe["nome_modelo"]
    )

    nomes_esperados = set(
        UHE_MODELO
    )

    faltantes = (
        nomes_esperados -
        nomes_uhe
    )

    if faltantes:
        raise ValueError(
            "UHEs ausentes em DADOS_UHE_UHR.xlsx: "
            f"{faltantes}"
        )

    # --------------------------------------------------------
    # Verificações da UHR
    # --------------------------------------------------------

    if len(uhr) != 1:
        raise ValueError(
            "A planilha deve possuir exatamente "
            "uma linha do tipo UHR."
        )

    parametros_uhr = [
        "Pmax geração (MW)",
        "Pmax bombeamento (MW)",
        "Eficiência bombeamento",
        "Eficiência turbinamento",
        "Emax (MWh)",
        "Einicial (MWh)",
        "Efinal (MWh)",
        "Custo (R$/MWh)",
    ]

    for parametro in parametros_uhr:

        if pd.isna(
            uhr.iloc[0][parametro]
        ):
            raise ValueError(
                f"Parâmetro da UHR não preenchido: "
                f"{parametro}"
            )

    # --------------------------------------------------------
    # Verificações físicas simples
    # --------------------------------------------------------

    eta_bomb = (
        uhr.iloc[0]
        ["Eficiência bombeamento"]
    )

    eta_turb = (
        uhr.iloc[0]
        ["Eficiência turbinamento"]
    )

    if not (
        0 < eta_bomb <= 1
    ):
        raise ValueError(
            "Eficiência de bombeamento deve "
            "estar entre 0 e 1."
        )

    if not (
        0 < eta_turb <= 1
    ):
        raise ValueError(
            "Eficiência de turbinamento deve "
            "estar entre 0 e 1."
        )

    emax = (
        uhr.iloc[0]["Emax (MWh)"]
    )

    einicial = (
        uhr.iloc[0]["Einicial (MWh)"]
    )

    efinal = (
        uhr.iloc[0]["Efinal (MWh)"]
    )

    if einicial > emax:
        raise ValueError(
            "Einicial da UHR é maior que Emax."
        )

    if efinal > emax:
        raise ValueError(
            "Efinal da UHR é maior que Emax."
        )

    # --------------------------------------------------------
    # Renomeia para nomes mais amigáveis ao código
    # --------------------------------------------------------

    uhe = uhe.rename(
        columns={
            "Q tur max (m³/s)":
                "Qtur_max_m3s",

            "Produtibilidade (MW/(m³/s))":
                "produtibilidade_MW_m3s",

            "Vmin (hm³)":
                "Vmin_hm3",

            "Vmax (hm³)":
                "Vmax_hm3",
        }
    )

    uhr = uhr.rename(
        columns={
            "Pmax geração (MW)":
                "Pger_max_MW",

            "Pmax bombeamento (MW)":
                "Pbomb_max_MW",

            "Eficiência bombeamento":
                "eta_bomb",

            "Eficiência turbinamento":
                "eta_turb",

            "Emax (MWh)":
                "Emax_MWh",

            "Einicial (MWh)":
                "Einicial_MWh",

            "Efinal (MWh)":
                "Efinal_MWh",

            "Custo (R$/MWh)":
                "custo_ciclo_R$_MWh",
        }
    )

    # Mantém somente as colunas relevantes
    uhe = uhe[
        [
            "Elemento",
            "Nome",
            "nome_modelo",
            "Qtur_max_m3s",
            "produtibilidade_MW_m3s",
            "Vmin_hm3",
            "Vmax_hm3",
        ]
    ].copy()

    uhr = uhr[
        [
            "Elemento",
            "Nome",
            "Pger_max_MW",
            "Pbomb_max_MW",
            "eta_bomb",
            "eta_turb",
            "Emax_MWh",
            "Einicial_MWh",
            "Efinal_MWh",
            "custo_ciclo_R$_MWh",
        ]
    ].copy()

    # Potência da UHR adotada no estudo. Mantêm-se Emax, E0 e Ef
    # informados na planilha; apenas as potências de geração/bombeamento
    # são atualizadas para 800 MW.
    uhr.loc[:, "Pger_max_MW"] = POTENCIA_UHR_ADOTADA_MW
    uhr.loc[:, "Pbomb_max_MW"] = POTENCIA_UHR_ADOTADA_MW

    print(
        "  UHEs cadastradas: 5"
    )

    print(
        "  UHR cadastrada: 1"
    )

    print(
        f"  UHR: "
        f"{uhr.iloc[0]['Pger_max_MW']:.1f} MW / "
        f"{uhr.iloc[0]['Emax_MWh']:.1f} MWh"
    )

    return uhe, uhr


# ============================================================
# 7. CURVA DE CARGA
# ============================================================

def processar_carga():

    print(
        "Processando curva de carga..."
    )

    df = ler_csv(
        ARQ_CARGA
    )

    verificar_colunas(
        df,
        [
            "id_subsistema",
            "din_instante",
            "val_cargaenergiahomwmed",
        ],
        "CURVA_CARGA"
    )

    df["din_instante"] = pd.to_datetime(
        df["din_instante"]
    )

    # Nordeste
    df = df[
        df["id_subsistema"]
        .astype(str)
        .str.strip()
        .eq("NE")
    ].copy()

    # Semana
    df = df[
        (
            df["din_instante"]
            >= DATA_INICIO
        )
        &
        (
            df["din_instante"]
            <= DATA_FIM
        )
    ].copy()

    df = (
        df[
            [
                "din_instante",
                "val_cargaenergiahomwmed",
            ]
        ]
        .rename(
            columns={
                "din_instante":
                    "data_hora",

                "val_cargaenergiahomwmed":
                    "carga_NE_MW",
            }
        )
        .sort_values(
            "data_hora"
        )
        .reset_index(
            drop=True
        )
    )

    if len(df) != NUM_HORAS:
        raise ValueError(
            "Carga Nordeste: "
            f"esperadas {NUM_HORAS} horas, "
            f"encontradas {len(df)}."
        )

    df.insert(
        0,
        "hora",
        np.arange(
            1,
            NUM_HORAS + 1
        )
    )

    return df


# ============================================================
# 8. RENOVÁVEIS
# ============================================================

def processar_renovaveis():

    print(
        "Processando eólica e solar..."
    )

    df = ler_csv(
        ARQ_RENOVAVEIS
    )

    verificar_colunas(
        df,
        [
            "id_subsistema",
            "nom_tipousina",
            "din_instante",
            "val_geracaoverificada",
            "val_capacidadeinstalada",
        ],
        "FATOR_CAPACIDADE"
    )

    df["din_instante"] = pd.to_datetime(
        df["din_instante"]
    )

    # Nordeste
    df = df[
        df["id_subsistema"]
        .astype(str)
        .str.strip()
        .eq("NE")
    ].copy()

    # Semana
    df = df[
        (
            df["din_instante"]
            >= DATA_INICIO
        )
        &
        (
            df["din_instante"]
            <= DATA_FIM
        )
    ].copy()

    df["tipo_norm"] = (
        df["nom_tipousina"]
        .apply(normalizar_texto)
    )

    def agregar_tecnologia(
        dados,
        termo,
        nome
    ):

        temp = dados[
            dados["tipo_norm"]
            .str.contains(
                termo,
                na=False
            )
        ].copy()

        if temp.empty:
            raise ValueError(
                f"Nenhum registro encontrado "
                f"para {nome} no Nordeste."
            )

        temp = (
            temp
            .groupby(
                "din_instante",
                as_index=False
            )
            .agg(
                geracao_MW=(
                    "val_geracaoverificada",
                    "sum"
                ),

                capacidade_MW=(
                    "val_capacidadeinstalada",
                    "sum"
                )
            )
        )

        # FC equivalente ponderado
        temp["fc"] = np.where(
            temp["capacidade_MW"] > 0,
            temp["geracao_MW"]
            / temp["capacidade_MW"],
            0.0
        )

        # Pequena proteção contra inconsistências numéricas
        # ou registros excepcionais da base.
        temp["fc"] = temp["fc"].clip(
            lower=0.0,
            upper=1.0
        )

        temp = temp.rename(
            columns={
                "din_instante":
                    "data_hora",

                "geracao_MW":
                    f"geracao_{nome}_NE_MW",

                "capacidade_MW":
                    f"capacidade_{nome}_NE_MW",

                "fc":
                    f"fc_{nome}_NE",
            }
        )

        return temp

    eolica = agregar_tecnologia(
        df,
        "EOL",
        "eolica"
    )

    solar = agregar_tecnologia(
        df,
        "SOLAR",
        "solar"
    )

    renovaveis = pd.merge(
        eolica,
        solar,
        on="data_hora",
        how="outer"
    )

    renovaveis = (
        renovaveis
        .sort_values("data_hora")
        .reset_index(drop=True)
    )

    if len(renovaveis) != NUM_HORAS:
        raise ValueError(
            "Renováveis: "
            f"esperadas {NUM_HORAS} horas, "
            f"encontradas {len(renovaveis)}."
        )

    renovaveis.insert(
        0,
        "hora",
        np.arange(
            1,
            NUM_HORAS + 1
        )
    )

    return renovaveis


# ============================================================
# 9. CAPACIDADE DAS UHEs
# ============================================================

def processar_capacidade_uhe():

    print(
        "Processando capacidade elétrica das UHEs..."
    )

    df = ler_csv(
        ARQ_CAPACIDADE
    )

    verificar_colunas(
        df,
        [
            "nom_usina",
            "val_potenciaefetiva",
        ],
        "CAPACIDADE_GERACAO"
    )

    df["nome_norm"] = (
        df["nom_usina"]
        .apply(normalizar_texto)
    )

    resultados = []

    for nome_modelo, nomes_base \
            in MAPA_CAPACIDADE_UHE.items():

        nomes_norm = [
            normalizar_texto(x)
            for x in nomes_base
        ]

        temp = df[
            df["nome_norm"]
            .isin(nomes_norm)
        ].copy()

        if temp.empty:
            raise ValueError(
                "Capacidade não encontrada "
                f"para {nome_modelo}."
            )

        pmax = (
            pd.to_numeric(
                temp["val_potenciaefetiva"],
                errors="coerce"
            )
            .fillna(0)
            .sum()
        )

        resultados.append({
            "nome_modelo":
                nome_modelo,

            "Pmax_MW":
                pmax,

            "cadastro_ONS":
                " + ".join(
                    nomes_base
                ),
        })

    return pd.DataFrame(
        resultados
    )


# ============================================================
# 10. JUNÇÃO DOS PARÂMETROS DAS UHEs
# ============================================================

def montar_parametros_uhe(
    cadastro_uhe,
    capacidade_uhe
):
    """
    Combina:

    DADOS_UHE_UHR.xlsx:
        Qmax
        produtibilidade
        Vmin
        Vmax

    CAPACIDADE_GERACAO:
        Pmax
    """

    parametros = pd.merge(
        cadastro_uhe,
        capacidade_uhe,
        on="nome_modelo",
        how="left"
    )

    if parametros[
        "Pmax_MW"
    ].isna().any():

        faltantes = parametros.loc[
            parametros["Pmax_MW"].isna(),
            "nome_modelo"
        ].tolist()

        raise ValueError(
            "Pmax não encontrado para: "
            f"{faltantes}"
        )

    # --------------------------------------------------------
    # Checagem útil:
    # potência aproximada rho * Qmax
    # --------------------------------------------------------

    parametros[
        "Pmax_rho_Q_MW"
    ] = (
        parametros[
            "Qtur_max_m3s"
        ]
        *
        parametros[
            "produtibilidade_MW_m3s"
        ]
    )

    parametros[
        "diferenca_Pmax_pct"
    ] = (
        100
        *
        (
            parametros[
                "Pmax_rho_Q_MW"
            ]
            -
            parametros[
                "Pmax_MW"
            ]
        )
        /
        parametros[
            "Pmax_MW"
        ]
    )

    return parametros

def classificar_uhe(parametros_uhe):
    """
    Classifica as UHEs conforme a representação adotada
    no modelo de otimização.
    """

    parametros_uhe = parametros_uhe.copy()

    tipo = {
        "TRES MARIAS": "RESERVATORIO",
        "SOBRADINHO": "RESERVATORIO",
        "ITAPARICA": "RESERVATORIO",
        "PAULO AFONSO": "FIO_DAGUA",
        "XINGO": "FIO_DAGUA",
    }

    parametros_uhe["tipo_modelagem"] = (
        parametros_uhe["nome_modelo"].map(tipo)
    )

    parametros_uhe["possui_estado_volume"] = (
        parametros_uhe["tipo_modelagem"]
        == "RESERVATORIO"
    )

    return parametros_uhe


# ============================================================
# 11. CAPACIDADE DAS UTEs
# ============================================================

def processar_capacidade_ute():

    print(
        "Processando capacidade das UTEs..."
    )

    df = ler_csv(
        ARQ_CAPACIDADE
    )

    verificar_colunas(
        df,
        [
            "nom_usina",
            "val_potenciaefetiva",
        ],
        "CAPACIDADE_GERACAO"
    )

    df["nome_norm"] = (
        df["nom_usina"]
        .apply(normalizar_texto)
    )

    resultados = []

    for nome_modelo, nomes_base \
            in MAPA_CAPACIDADE_UTE.items():

        nomes_norm = [
            normalizar_texto(x)
            for x in nomes_base
        ]

        temp = df[
            df["nome_norm"]
            .isin(nomes_norm)
        ].copy()

        if temp.empty:
            raise ValueError(
                "Capacidade não encontrada "
                f"para {nome_modelo}."
            )

        pmax = (
            pd.to_numeric(
                temp["val_potenciaefetiva"],
                errors="coerce"
            )
            .fillna(0)
            .sum()
        )

        resultados.append({
            "nome":
                nome_modelo,

            "Pmax_MW":
                pmax,
        })

    return pd.DataFrame(
        resultados
    )


# ============================================================
# 12. CVU DAS UTEs
# ============================================================

def processar_cvu():

    print(
        "Processando CVUs das UTEs..."
    )

    df = ler_csv(
        ARQ_CVU
    )

    verificar_colunas(
        df,
        [
            "dat_iniciosemana",
            "dat_fimsemana",
            "nom_usina",
            "val_cvu",
        ],
        "CVU_USINA_TERMICA"
    )

    df["dat_iniciosemana"] = pd.to_datetime(
        df["dat_iniciosemana"]
    )

    df["dat_fimsemana"] = pd.to_datetime(
        df["dat_fimsemana"]
    )

    # Semana contendo o período do estudo
    df = df[
        (
            df["dat_iniciosemana"]
            <= DATA_INICIO.normalize()
        )
        &
        (
            df["dat_fimsemana"]
            >= DATA_FIM.normalize()
        )
    ].copy()

    df["nome_norm"] = (
        df["nom_usina"]
        .apply(normalizar_texto)
    )

    resultados = []

    for nome_modelo, nomes_base \
            in MAPA_CVU_UTE.items():

        nomes_norm = [
            normalizar_texto(x)
            for x in nomes_base
        ]

        temp = df[
            df["nome_norm"]
            .isin(nomes_norm)
        ].copy()

        if temp.empty:
            raise ValueError(
                f"CVU não encontrado para "
                f"{nome_modelo}."
            )

        valores = (
            pd.to_numeric(
                temp["val_cvu"],
                errors="coerce"
            )
            .dropna()
            .unique()
        )

        if len(valores) == 0:
            raise ValueError(
                f"CVU vazio para {nome_modelo}."
            )

        if len(valores) > 1:
            raise ValueError(
                f"Mais de um CVU encontrado para "
                f"{nome_modelo}: {valores}"
            )

        resultados.append({
            "nome":
                nome_modelo,

            "CVU_R$_MWh":
                float(valores[0]),

            "nome_CVU_ONS":
                " + ".join(
                    temp["nom_usina"]
                    .astype(str)
                    .unique()
                ),
        })

    return pd.DataFrame(
        resultados
    )


# ============================================================
# 13. PARÂMETROS DAS UTEs
# ============================================================

def processar_termicas():

    capacidade = (
        processar_capacidade_ute()
    )

    cvu = (
        processar_cvu()
    )

    ute = pd.merge(
        capacidade,
        cvu,
        on="nome",
        how="inner"
    )

    return ute


# ============================================================
# 14. DADOS HIDROLÓGICOS
# ============================================================

def processar_hidrologia():

    print(
        "Processando dados hidrológicos..."
    )

    df = ler_csv(
        ARQ_HIDRO
    )

    verificar_colunas(
        df,
        [
            "nom_reservatorio",
            "din_instante",
            "val_volumeutil",
            "val_vazaoafluente",
            "val_vazaodefluente",
            "val_vazaoturbinada",
            "val_vazaovertida",
        ],
        "DADOS_HIDROLOGICOS"
    )

    df["din_instante"] = pd.to_datetime(
        df["din_instante"]
    )

    df["nome_norm"] = (
        df["nom_reservatorio"]
        .apply(normalizar_texto)
    )

    # Seleciona os dias 05 a 11/09
    inicio = DATA_INICIO.normalize()

    fim_exclusivo = (
        DATA_FIM.normalize()
        + pd.Timedelta(days=1)
    )

    df = df[
        (
            df["din_instante"] >= inicio
        )
        &
        (
            df["din_instante"] < fim_exclusivo
        )
    ].copy()

    resultados = {}

    for nome_modelo, nomes_base \
            in MAPA_HIDRO.items():

        nomes_norm = [
            normalizar_texto(x)
            for x in nomes_base
        ]

        temp = df[
            df["nome_norm"]
            .isin(nomes_norm)
        ].copy()

        if temp.empty:
            raise ValueError(
                "Dados hidrológicos não encontrados "
                f"para {nome_modelo}."
            )

        temp = temp[
            [
                "din_instante",
                "nom_reservatorio",
                "val_volumeutil",
                "val_vazaoafluente",
                "val_vazaodefluente",
                "val_vazaoturbinada",
                "val_vazaovertida",
            ]
        ].copy()

        temp = temp.rename(
            columns={
                "din_instante":
                    "data_hora",

                "val_volumeutil":
                    "volume_util_ONS",

                "val_vazaoafluente":
                    "vazao_afluente_m3s",

                "val_vazaodefluente":
                    "vazao_defluente_m3s",

                "val_vazaoturbinada":
                    "vazao_turbinada_m3s",

                "val_vazaovertida":
                    "vazao_vertida_m3s",
            }
        )

        temp = (
            temp
            .sort_values(
                [
                    "data_hora",
                    "nom_reservatorio"
                ]
            )
            .reset_index(drop=True)
        )

        temp = (
            temp
            .sort_values("data_hora")
            .reset_index(drop=True)
        )

        if len(temp) != NUM_HORAS:
            raise ValueError(
                f"{nome_modelo}: esperados {NUM_HORAS} "
                f"registros hidrológicos, encontrados {len(temp)}."
            )

        temp.insert(
            0,
            "hora",
            np.arange(1, NUM_HORAS + 1)
        )

        resultados[
            nome_modelo
        ] = temp

    return resultados

# ============================================================
# 15. AFLUÊNCIAS REPRESENTATIVAS E CONTRIBUIÇÕES LATERAIS
# ============================================================
# 15B. ENTRADAS HIDRÁULICAS FINAIS DA CASCATA
# ============================================================

# Valores equivalentes calibrados para a semana de estudo.
# Zero significa que não foi identificada contribuição lateral positiva
# compatível com a representação hidráulica simplificada adotada.
LATERAIS_EQ_M3S = {
    "SOBRADINHO": 0.0,
    "ITAPARICA": 0.0,
    "PAULO AFONSO": 0.0,
    "XINGO": 262.37,
}

def calcular_afluencias_representativas(hidrologia):
    """
    Calcula a afluência externa representativa de Três Marias.

    A série horária observada é agregada em médias diárias e a média
    de cada dia é repetida nas 24 horas correspondentes.

    Três Marias é a cabeceira da cascata representada no modelo.
    Portanto, sua afluência é tratada como entrada hídrica externa.

    As afluências das demais UHEs não são utilizadas diretamente no
    otimizador, pois o acoplamento hidráulico será representado pelas
    defluências das usinas de montante e pelas contribuições laterais
    equivalentes calibradas.
    """

    nome_uhe = "TRES MARIAS"

    if nome_uhe not in hidrologia:
        raise KeyError(
            f"Dados hidrológicos de {nome_uhe} não encontrados."
        )

    df = hidrologia[nome_uhe].copy()

    colunas_obrigatorias = [
        "data_hora",
        "vazao_afluente_m3s",
    ]

    faltantes = [
        col for col in colunas_obrigatorias
        if col not in df.columns
    ]

    if faltantes:
        raise ValueError(
            f"Colunas ausentes nos dados hidrológicos de "
            f"{nome_uhe}: {faltantes}"
        )

    # ---------------------------------------------------------
    # Tratamento das datas e vazões
    # ---------------------------------------------------------

    df["data_hora"] = pd.to_datetime(
        df["data_hora"],
        errors="coerce"
    )

    df["vazao_afluente_m3s"] = pd.to_numeric(
        df["vazao_afluente_m3s"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["data_hora", "vazao_afluente_m3s"]
    ).copy()

    df = df.sort_values("data_hora").reset_index(drop=True)

    # ---------------------------------------------------------
    # Seleção da semana do modelo
    # ---------------------------------------------------------

    inicio = pd.Timestamp(DATA_INICIO)
    fim = inicio + pd.Timedelta(hours=NUM_HORAS - 1)

    df = df[
        (df["data_hora"] >= inicio)
        & (df["data_hora"] <= fim)
    ].copy()

    if df.empty:
        raise ValueError(
            "Não foram encontrados dados de afluência de Três "
            "Marias para o horizonte do modelo."
        )

    # ---------------------------------------------------------
    # Média diária da afluência observada
    # ---------------------------------------------------------

    df["data_modelo"] = df["data_hora"].dt.normalize()

    medias_diarias = (
        df.groupby("data_modelo", as_index=False)
        ["vazao_afluente_m3s"]
        .mean()
        .rename(
            columns={
                "vazao_afluente_m3s":
                "afluencia_externa_tres_marias_m3s"
            }
        )
    )

    # ---------------------------------------------------------
    # Construção dos 168 períodos horários
    # ---------------------------------------------------------

    datas_modelo = pd.date_range(
        start=inicio,
        periods=NUM_HORAS,
        freq="h"
    )

    resultado = pd.DataFrame({
        "hora": range(1, NUM_HORAS + 1),
        "data_hora": datas_modelo,
    })

    resultado["data_modelo"] = (
        resultado["data_hora"].dt.normalize()
    )

    resultado = resultado.merge(
        medias_diarias,
        on="data_modelo",
        how="left"
    )

    # ---------------------------------------------------------
    # Validação
    # ---------------------------------------------------------

    if resultado[
        "afluencia_externa_tres_marias_m3s"
    ].isna().any():

        dias_sem_dados = (
            resultado.loc[
                resultado[
                    "afluencia_externa_tres_marias_m3s"
                ].isna(),
                "data_modelo",
            ]
            .dt.strftime("%d/%m/%Y")
            .unique()
            .tolist()
        )

        raise ValueError(
            "Existem dias sem afluência representativa de "
            f"Três Marias: {dias_sem_dados}"
        )

    return resultado

def diagnosticar_tres_marias(afluencias_representativas, hidrologia):
    """
    Gera diagnóstico da afluência externa de Três Marias utilizada
    no modelo.

    A análise compara:
      - afluência externa representativa;
      - defluência observada;
      - volume útil observado.

    O diagnóstico é apenas informativo e não altera a série que será
    fornecida ao otimizador.
    """

    print("\nAnalisando afluência externa de Três Marias...")

    # ==========================================================
    # 1. AFLUÊNCIA REPRESENTATIVA
    # ==========================================================

    # afluencias_representativas agora é diretamente um DataFrame,
    # e não mais um dicionário indexado pelo nome da UHE.
    rep = afluencias_representativas.copy()

    col_afl = "afluencia_externa_tres_marias_m3s"

    if col_afl not in rep.columns:
        raise ValueError(
            f"Coluna '{col_afl}' não encontrada na série "
            "representativa de Três Marias."
        )

    rep["data_hora"] = pd.to_datetime(
        rep["data_hora"],
        errors="coerce"
    )

    rep["data_modelo"] = rep["data_hora"].dt.normalize()

    # Como a afluência representativa é constante dentro de cada dia,
    # usamos a média diária.
    afl_diaria = (
        rep.groupby("data_modelo", as_index=False)[col_afl]
        .mean()
        .rename(
            columns={
                col_afl: "afluencia_externa_media_m3s"
            }
        )
    )

    # ==========================================================
    # 2. DADOS HIDROLÓGICOS OBSERVADOS
    # ==========================================================

    if "TRES MARIAS" not in hidrologia:
        raise KeyError(
            "Dados hidrológicos de TRES MARIAS não encontrados."
        )

    hidro_tm = hidrologia["TRES MARIAS"].copy()

    hidro_tm["data_hora"] = pd.to_datetime(
        hidro_tm["data_hora"],
        errors="coerce"
    )

    hidro_tm["data_modelo"] = (
        hidro_tm["data_hora"].dt.normalize()
    )

    # Converte as colunas necessárias para numérico
    for col in [
        "vazao_defluente_m3s",
        "volume_util_ONS",
    ]:
        if col in hidro_tm.columns:
            hidro_tm[col] = pd.to_numeric(
                hidro_tm[col],
                errors="coerce"
            )

    # ==========================================================
    # 3. ESTATÍSTICAS DIÁRIAS OBSERVADAS
    # ==========================================================

    agregacoes = {}

    if "vazao_defluente_m3s" in hidro_tm.columns:
        agregacoes["vazao_defluente_m3s"] = "mean"

    if "volume_util_ONS" in hidro_tm.columns:
        agregacoes["volume_util_ONS"] = [
            "mean",
            "min",
            "max",
        ]

    if not agregacoes:
        raise ValueError(
            "Não foram encontradas colunas suficientes para "
            "o diagnóstico de Três Marias."
        )

    obs_diario = (
        hidro_tm.groupby("data_modelo")
        .agg(agregacoes)
        .reset_index()
    )

    # O groupby com várias estatísticas cria MultiIndex.
    # Vamos normalizar os nomes.
    novas_colunas = []

    for col in obs_diario.columns:
        if isinstance(col, tuple):

            nome = col[0]
            estatistica = col[1]

            if nome == "data_modelo":
                novas_colunas.append("data_modelo")

            elif nome == "vazao_defluente_m3s":
                novas_colunas.append(
                    "defluencia_observada_media_m3s"
                )

            elif nome == "volume_util_ONS":

                if estatistica == "mean":
                    novas_colunas.append(
                        "volume_util_ONS_medio"
                    )

                elif estatistica == "min":
                    novas_colunas.append(
                        "volume_util_ONS_min"
                    )

                elif estatistica == "max":
                    novas_colunas.append(
                        "volume_util_ONS_max"
                    )

                else:
                    novas_colunas.append(
                        f"{nome}_{estatistica}"
                    )

            else:
                novas_colunas.append(
                    f"{nome}_{estatistica}".strip("_")
                )

        else:
            novas_colunas.append(col)

    obs_diario.columns = novas_colunas

    # ==========================================================
    # 4. JUNÇÃO DO DIAGNÓSTICO
    # ==========================================================

    diagnostico = afl_diaria.merge(
        obs_diario,
        on="data_modelo",
        how="left"
    )

    # ==========================================================
    # 5. ESTATÍSTICAS DA AFLUÊNCIA EXTERNA
    # ==========================================================

    media_semana = diagnostico[
        "afluencia_externa_media_m3s"
    ].mean()

    minimo_semana = diagnostico[
        "afluencia_externa_media_m3s"
    ].min()

    maximo_semana = diagnostico[
        "afluencia_externa_media_m3s"
    ].max()

    desvio_padrao = diagnostico[
        "afluencia_externa_media_m3s"
    ].std()

    if media_semana != 0:
        coef_variacao = (
            desvio_padrao / abs(media_semana) * 100
        )
    else:
        coef_variacao = np.nan

    diagnostico[
        "afluencia_media_semanal_m3s"
    ] = media_semana

    diagnostico[
        "desvio_da_media_m3s"
    ] = (
        diagnostico["afluencia_externa_media_m3s"]
        - media_semana
    )

    diagnostico[
        "desvio_da_media_pct"
    ] = np.where(
        media_semana != 0,
        100
        * diagnostico["desvio_da_media_m3s"]
        / media_semana,
        np.nan,
    )

    # ==========================================================
    # 6. IMPRESSÃO DO DIAGNÓSTICO
    # ==========================================================

    print("\n  Três Marias - afluência externa representativa:")
    print(f"    Média semanal : {media_semana:.2f} m³/s")
    print(f"    Mínimo diário : {minimo_semana:.2f} m³/s")
    print(f"    Máximo diário : {maximo_semana:.2f} m³/s")
    print(f"    Desvio padrão : {desvio_padrao:.2f} m³/s")
    print(f"    Coef. variação: {coef_variacao:.2f} %")

    print("\n  Valores diários:")

    for _, linha in diagnostico.iterrows():

        data = pd.Timestamp(
            linha["data_modelo"]
        ).strftime("%d/%m/%Y")

        afl = linha[
            "afluencia_externa_media_m3s"
        ]

        print(
            f"    {data}: "
            f"{afl:.2f} m³/s"
        )

    return diagnostico

# ============================================================
# 15B. DEFLUÊNCIAS MÍNIMAS REPRESENTATIVAS
# ============================================================

def calcular_defluencias_minimas(hidrologia):
    """
    Constrói um piso operacional exógeno de defluência para cada UHE.

    Para cada dia e usina, utiliza o percentil configurável da
    defluência observada no ONS e repete esse valor nas 24 horas.
    O padrão é o percentil 10%, escolhido como piso conservador e
    data-driven, evitando impor a média observada como obrigação.
    """
    base = criar_indice_horario()
    out = base[["hora", "data_hora"]].copy()
    out["dia"] = out["data_hora"].dt.normalize()

    for usina in UHE_MODELO:
        df = hidrologia[usina][["data_hora", "vazao_defluente_m3s"]].copy()
        df["dia"] = pd.to_datetime(df["data_hora"]).dt.normalize()
        df["vazao_defluente_m3s"] = pd.to_numeric(
            df["vazao_defluente_m3s"], errors="coerce"
        )
        piso = (
            df.groupby("dia")["vazao_defluente_m3s"]
              .quantile(QUANTIL_DEFLUENCIA_MINIMA)
              .clip(lower=0.0)
        )
        slug = usina.lower().replace(" ", "_")
        out[f"defluencia_min_{slug}_m3s"] = out["dia"].map(piso)

    out = out.drop(columns="dia")
    if out.isna().any().any():
        raise ValueError("Falha ao construir defluências mínimas representativas.")
    return out



# ============================================================
# 15C. TURBINAMENTOS MÍNIMOS REPRESENTATIVOS
# ============================================================

def calcular_turbinamentos_minimos(hidrologia):
    """
    Constrói um piso operacional exógeno de vazão turbinada para cada UHE.

    Para cada dia e usina, utiliza o percentil configurável da vazão
    turbinada observada e repete esse valor nas 24 horas. O piso não
    reproduz o despacho histórico: apenas evita soluções em que a usina
    substitui sistematicamente turbinamento por vertimento.
    """
    base = criar_indice_horario()
    out = base[["hora", "data_hora"]].copy()
    out["dia"] = out["data_hora"].dt.normalize()

    for usina in UHE_MODELO:
        df = hidrologia[usina][["data_hora", "vazao_turbinada_m3s"]].copy()
        df["dia"] = pd.to_datetime(df["data_hora"]).dt.normalize()
        df["vazao_turbinada_m3s"] = pd.to_numeric(
            df["vazao_turbinada_m3s"], errors="coerce"
        )
        piso = (
            df.groupby("dia")["vazao_turbinada_m3s"]
              .quantile(QUANTIL_TURBINAMENTO_MINIMO)
              .clip(lower=0.0)
        )
        slug = usina.lower().replace(" ", "_")
        out[f"turbinamento_min_{slug}_m3s"] = out["dia"].map(piso)

    out = out.drop(columns="dia")
    if out.isna().any().any():
        raise ValueError("Falha ao construir turbinamentos mínimos representativos.")
    return out

# ============================================================
# 16. SÉRIES HORÁRIAS FINAIS PARA O OTIMIZADOR
# ============================================================

def montar_series_modelo(carga, renovaveis, afluencias_representativas, defluencias_minimas, turbinamentos_minimos):
    """Monta somente as séries exógenas necessárias ao otimizador.py."""
    print("Montando séries horárias finais do modelo...")
    base=criar_indice_horario()
    base=pd.merge(base,carga[["data_hora","carga_NE_MW"]],on="data_hora",how="left",validate="one_to_one")
    base=pd.merge(base,renovaveis[["data_hora","fc_eolica_NE","fc_solar_NE","capacidade_eolica_NE_MW","capacidade_solar_NE_MW"]],on="data_hora",how="left",validate="one_to_one")
    tm = afluencias_representativas[["hora", "afluencia_externa_tres_marias_m3s"]].copy()
    base=pd.merge(base,tm,on="hora",how="left",validate="one_to_one")
    for usina,valor in LATERAIS_EQ_M3S.items():
        slug=usina.lower().replace(" ","_")
        base[f"contrib_lateral_eq_{slug}_m3s"]=float(valor)
    cols_qmin = [c for c in defluencias_minimas.columns if c.startswith("defluencia_min_")]
    base = pd.merge(
        base, defluencias_minimas[["hora"] + cols_qmin],
        on="hora", how="left", validate="one_to_one"
    )
    cols_qtur_min = [c for c in turbinamentos_minimos.columns if c.startswith("turbinamento_min_")]
    base = pd.merge(
        base, turbinamentos_minimos[["hora"] + cols_qtur_min],
        on="hora", how="left", validate="one_to_one"
    )
    if len(base)!=NUM_HORAS: raise ValueError("Series_Modelo não possui 168 períodos.")
    if base.isna().any().any():
        raise ValueError(f"Valores ausentes em Series_Modelo: {base.columns[base.isna().any()].tolist()}")
    return base

# ============================================================
# 17. VOLUMES INICIAIS OBSERVADOS
# ============================================================

def calcular_volumes_iniciais(
    hidrologia,
    parametros_uhe
):
    """
    Converte o volume útil percentual informado pelo ONS
    para volume absoluto em hm³.

    V0 = Vmin + VU/100 * (Vmax - Vmin)

    A conversão é aplicada somente às UHEs representadas
    explicitamente com reservatório.
    """

    resultados = []

    for _, row in parametros_uhe.iterrows():

        nome = row["nome_modelo"]
        tipo = row["tipo_modelagem"]

        if nome not in hidrologia:
            raise ValueError(
                f"Hidrologia não encontrada para {nome}."
            )

        dados = hidrologia[nome]

        volume_percentual = (
            dados["volume_util_ONS"]
            .dropna()
            .iloc[0]
        )

        registro = {
            "nome_modelo": nome,
            "tipo_modelagem": tipo,
            "volume_util_inicial_pct":
                volume_percentual,
        }

        if tipo == "RESERVATORIO":

            vmin = row["Vmin_hm3"]
            vmax = row["Vmax_hm3"]

            v0 = (
                vmin
                +
                (volume_percentual / 100.0)
                * (vmax - vmin)
            )

            registro.update({
                "Vmin_hm3": vmin,
                "Vmax_hm3": vmax,
                "Vinicial_hm3": v0,
            })

        else:

            registro.update({
                "Vmin_hm3": np.nan,
                "Vmax_hm3": np.nan,
                "Vinicial_hm3": np.nan,
            })

        resultados.append(registro)

    return pd.DataFrame(resultados)

# ============================================================
# 18. RESUMO DAS ENTRADAS
# ============================================================

def criar_resumo(
    carga,
    renovaveis,
    parametros_uhe,
    ute,
    uhr
):

    linhas = []

    # --------------------------------------------------------
    # Carga
    # --------------------------------------------------------

    linhas.extend([
        {
            "Indicador":
                "Carga mínima NE",
            "Valor":
                carga["carga_NE_MW"].min(),
            "Unidade":
                "MW"
        },

        {
            "Indicador":
                "Carga média NE",
            "Valor":
                carga["carga_NE_MW"].mean(),
            "Unidade":
                "MW"
        },

        {
            "Indicador":
                "Carga máxima NE",
            "Valor":
                carga["carga_NE_MW"].max(),
            "Unidade":
                "MW"
        },

        {
            "Indicador":
                "Energia semanal NE",
            "Valor":
                carga["carga_NE_MW"].sum(),
            "Unidade":
                "MWh"
        },
    ])

    # --------------------------------------------------------
    # Renováveis
    # --------------------------------------------------------

    linhas.extend([
        {
            "Indicador":
                "FC médio eólico NE",
            "Valor":
                renovaveis[
                    "fc_eolica_NE"
                ].mean(),
            "Unidade":
                "pu"
        },

        {
            "Indicador":
                "FC médio solar NE",
            "Valor":
                renovaveis[
                    "fc_solar_NE"
                ].mean(),
            "Unidade":
                "pu"
        },
    ])

    # --------------------------------------------------------
    # UHEs
    # --------------------------------------------------------

    for _, row in parametros_uhe.iterrows():

        linhas.append({
            "Indicador":
                f"Pmax {row['nome_modelo']}",
            "Valor":
                row["Pmax_MW"],
            "Unidade":
                "MW"
        })

    # --------------------------------------------------------
    # UTEs
    # --------------------------------------------------------

    for _, row in ute.iterrows():

        linhas.append({
            "Indicador":
                f"Pmax {row['nome']}",
            "Valor":
                row["Pmax_MW"],
            "Unidade":
                "MW"
        })

        linhas.append({
            "Indicador":
                f"CVU {row['nome']}",
            "Valor":
                row["CVU_R$_MWh"],
            "Unidade":
                "R$/MWh"
        })

    # --------------------------------------------------------
    # UHR
    # --------------------------------------------------------

    row = uhr.iloc[0]

    linhas.extend([
        {
            "Indicador":
                "Pmax geração UHR",
            "Valor":
                row["Pger_max_MW"],
            "Unidade":
                "MW"
        },

        {
            "Indicador":
                "Pmax bombeamento UHR",
            "Valor":
                row["Pbomb_max_MW"],
            "Unidade":
                "MW"
        },

        {
            "Indicador":
                "Emax UHR",
            "Valor":
                row["Emax_MWh"],
            "Unidade":
                "MWh"
        },

        {
            "Indicador":
                "Eficiência round-trip UHR",
            "Valor":
                row["eta_bomb"]
                * row["eta_turb"],
            "Unidade":
                "pu"
        },
    ])

    return pd.DataFrame(
        linhas
    )


# ============================================================
# 19. VALIDAÇÕES
# ============================================================

def validar_dados(
    carga,
    renovaveis,
    parametros_uhe,
    ute,
    uhr,
    hidrologia
):

    print(
        "\nRealizando validações..."
    )

    problemas = []
    avisos = []

    # --------------------------------------------------------
    # 168 horas
    # --------------------------------------------------------

    if len(carga) != NUM_HORAS:
        problemas.append(
            "Carga não possui 168 horas."
        )

    if len(renovaveis) != NUM_HORAS:
        problemas.append(
            "Renováveis não possuem 168 horas."
        )

    # --------------------------------------------------------
    # NaN
    # --------------------------------------------------------

    if carga[
        "carga_NE_MW"
    ].isna().any():

        problemas.append(
            "Carga possui valores ausentes."
        )

    for col in [
        "fc_eolica_NE",
        "fc_solar_NE"
    ]:

        if renovaveis[
            col
        ].isna().any():

            problemas.append(
                f"{col} possui valores ausentes."
            )

    # --------------------------------------------------------
    # UHE
    # --------------------------------------------------------

    if len(parametros_uhe) != 5:
        problemas.append(
            "Número de UHEs diferente de 5."
        )

    for _, row in parametros_uhe.iterrows():

        if (
            row["Qtur_max_m3s"] <= 0
        ):
            problemas.append(
                f"Qtur inválido: "
                f"{row['nome_modelo']}"
            )

        if (
            row[
                "produtibilidade_MW_m3s"
            ] <= 0
        ):
            problemas.append(
                f"Produtibilidade inválida: "
                f"{row['nome_modelo']}"
            )

        if row["tipo_modelagem"] == "RESERVATORIO":

            if pd.isna(row["Vmin_hm3"]) \
                    or pd.isna(row["Vmax_hm3"]):

                problemas.append(
                    "Volume ausente para reservatório: "
                    f"{row['nome_modelo']}"
                )

            elif row["Vmin_hm3"] >= row["Vmax_hm3"]:

                problemas.append(
                    f"Vmin >= Vmax: "
                    f"{row['nome_modelo']}"
                )

        diferenca = abs(
            row["diferenca_Pmax_pct"]
        )

        if diferenca > 10:

            avisos.append(
                f"{row['nome_modelo']}: "
                f"rho*Qmax difere de Pmax "
                f"em {diferenca:.1f}%."
            )

    # --------------------------------------------------------
    # Afluências negativas
    # --------------------------------------------------------

    for nome, dados_h in hidrologia.items():

        q = pd.to_numeric(
            dados_h["vazao_afluente_m3s"],
            errors="coerce"
        )

        n_neg = int((q < 0).sum())

        if n_neg > 0:

            avisos.append(
                f"{nome}: {n_neg} registros de "
                "afluência observada negativa na base "
                "original. Os dados brutos foram mantidos "
                "para rastreabilidade; a série utilizada "
                "pelo modelo é tratada posteriormente."
            )

    # --------------------------------------------------------
    # UTE
    # --------------------------------------------------------

    if len(ute) != 2:
        problemas.append(
            "Número de UTEs diferente de 2."
        )

    # --------------------------------------------------------
    # UHR
    # --------------------------------------------------------

    if len(uhr) != 1:
        problemas.append(
            "Número de UHRs diferente de 1."
        )

    # --------------------------------------------------------
    # Hidrologia
    # --------------------------------------------------------

    for nome in UHE_MODELO:

        if nome not in hidrologia:
            problemas.append(
                f"Hidrologia ausente: {nome}"
            )

    # --------------------------------------------------------
    # Exibe resultados
    # --------------------------------------------------------

    if problemas:

        print(
            "\nERROS/PROBLEMAS:"
        )

        for item in problemas:
            print(
                f"  - {item}"
            )

    else:
        print(
            "  OK: nenhuma inconsistência "
            "básica encontrada."
        )

    if avisos:

        print(
            "\nAVISOS:"
        )

        for item in avisos:
            print(
                f"  - {item}"
            )

    return problemas, avisos


# ============================================================
# 20. EXPORTAÇÃO
# ============================================================

def exportar_excel(parametros_uhe, ute, uhr, volumes_iniciais, series_modelo, diagnostico_tm, resumo):
    """Exporta somente parâmetros, séries finais e diagnóstico de Três Marias."""
    print("\nExportando entradas processadas finais...")
    with pd.ExcelWriter(ARQ_SAIDA, engine="openpyxl") as writer:
        resumo.to_excel(writer,sheet_name="Resumo",index=False)
        parametros_uhe.to_excel(writer,sheet_name="Parametros_UHE",index=False)
        ute.to_excel(writer,sheet_name="Parametros_UTE",index=False)
        uhr.to_excel(writer,sheet_name="Parametros_UHR",index=False)
        volumes_iniciais.to_excel(writer,sheet_name="Volumes_Iniciais_ONS",index=False)
        series_modelo.to_excel(writer,sheet_name="Series_Modelo",index=False)
        diagnostico_tm.to_excel(writer,sheet_name="Diagnostico_TM",index=False)
        metadados=pd.DataFrame({"Parametro":[
            "Data inicial","Data final","Número de períodos","Discretização","Cascata hidráulica","Tempo de viagem da água",
            "Entrada externa Três Marias","Lateral Sobradinho","Lateral Itaparica","Lateral Paulo Afonso","Lateral Xingó","Interpretação das laterais"],
            "Valor":[str(DATA_INICIO),str(DATA_FIM),NUM_HORAS,"Horária",
            "Três Marias -> Sobradinho -> Itaparica -> Paulo Afonso -> Xingó",
            "Desprezado; defluência de montante disponível a jusante no mesmo período",
            "Média diária da vazão afluente observada, repetida nas 24 horas",
            f"{LATERAIS_EQ_M3S['SOBRADINHO']:.2f} m3/s",f"{LATERAIS_EQ_M3S['ITAPARICA']:.2f} m3/s",
            f"{LATERAIS_EQ_M3S['PAULO AFONSO']:.2f} m3/s",f"{LATERAIS_EQ_M3S['XINGO']:.2f} m3/s",
            "Laterais equivalentes não negativas calibradas para a semana; zero não significa inexistência física de contribuição lateral."]})
        metadados.to_excel(writer,sheet_name="Metadados",index=False)
    print(f"\nArquivo gerado:\n{ARQ_SAIDA}")

# ============================================================
# 21. FUNÇÃO PRINCIPAL DE CARREGAMENTO
# ============================================================

def carregar_dados(exportar=False):
    """Prepara as entradas finais utilizadas pelo otimizador.py."""
    cadastro_uhe,uhr=processar_dados_uhe_uhr()
    carga=processar_carga(); renovaveis=processar_renovaveis()
    capacidade_uhe=processar_capacidade_uhe()
    parametros_uhe=classificar_uhe(montar_parametros_uhe(cadastro_uhe,capacidade_uhe))
    ute=processar_termicas(); hidrologia=processar_hidrologia()
    volumes_iniciais=calcular_volumes_iniciais(hidrologia,parametros_uhe)
    afluencias_representativas=calcular_afluencias_representativas(hidrologia)
    diagnostico_tm=diagnosticar_tres_marias(afluencias_representativas,hidrologia)
    defluencias_minimas=calcular_defluencias_minimas(hidrologia)
    turbinamentos_minimos=calcular_turbinamentos_minimos(hidrologia)
    series_modelo=montar_series_modelo(carga,renovaveis,afluencias_representativas,defluencias_minimas,turbinamentos_minimos)
    resumo=criar_resumo(carga,renovaveis,parametros_uhe,ute,uhr)
    problemas,avisos=validar_dados(carga,renovaveis,parametros_uhe,ute,uhr,hidrologia)
    dados={"carga":carga,"renovaveis":renovaveis,"uhe":parametros_uhe,"ute":ute,"uhr":uhr,
           "series_modelo":series_modelo,"volumes_iniciais":volumes_iniciais,"diagnostico_tm":diagnostico_tm,
           "resumo":resumo,"problemas":problemas,"avisos":avisos}
    if exportar:
        exportar_excel(parametros_uhe,ute,uhr,volumes_iniciais,series_modelo,diagnostico_tm,resumo)
    return dados

# ============================================================
# 22. EXECUÇÃO DIRETA
# ============================================================

def main():

    print(
        "\n"
        "====================================================\n"
        " PREPARAÇÃO DOS DADOS DE ENTRADA\n"
        " Sistema equivalente - Bacia do São Francisco\n"
        " Semana: 05/09/2026 a 11/09/2026\n"
        "====================================================\n"
    )

    dados = carregar_dados(
        exportar=True
    )

    print(
        "\n"
        "====================================================\n"
        " RESUMO DAS ENTRADAS\n"
        "====================================================\n"
    )

    print(
        dados["resumo"]
        .to_string(
            index=False
        )
    )

    print(
        "\n"
        "====================================================\n"
        " PROCESSAMENTO CONCLUÍDO\n"
        "===================================================="
    )


if __name__ == "__main__":
    main()
