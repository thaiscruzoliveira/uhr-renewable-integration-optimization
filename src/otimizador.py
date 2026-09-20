"""
otimizador.py
============================================================
Otimização determinística multi-período (168 h) de um sistema
equivalente do Nordeste com:

- 5 UHEs:
    Três Marias, Sobradinho, Itaparica,
    Complexo Paulo Afonso e Xingó;
- 2 UTEs:
    Prosperidade I e Suape II;
- geração eólica e solar representadas por fatores de
  capacidade horários reais;
- 1 UHR de circuito fechado;
- 4 cenários:
    C1 = penetração renovável base, sem UHR
    C2 = penetração renovável base, com UHR
    C3 = penetração renovável alta, sem UHR
    C4 = penetração renovável alta, com UHR

Formulação:
    Programação Linear determinística multi-período.

IMPORTANTE SOBRE A HIDROLOGIA
------------------------------------------------------------
A cascata hidráulica é explicitamente acoplada, sem tempo de
viagem da água entre aproveitamentos:

    Três Marias -> Sobradinho -> Itaparica ->
    Paulo Afonso -> Xingó

Três Marias recebe uma afluência externa representativa.
As demais usinas recebem a defluência otimizada da usina
imediatamente a montante mais uma contribuição lateral
equivalente exógena calibrada no módulo entrada_dados.py.

Três Marias, Sobradinho e Itaparica possuem estado de volume.
Paulo Afonso e Xingó são representadas a fio d'água.

Dependências:
    numpy
    pandas
    scipy
    openpyxl

Entrada:
    Arquivos de Entrada/entradas_processadas.xlsx
ou
    entradas_processadas.xlsx

Saída:
    Arquivos de Saida/resultados_otimizacao.xlsx
============================================================
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import lil_matrix


# ============================================================
# 1. CONFIGURAÇÕES GERAIS
# ============================================================

NUM_HORAS = 168
FATOR_HM3 = 0.0036  # (m³/s) * 1 h -> hm³
FATOR_CARGA_EQUIVALENTE = 1.00  # carga real do NE

# Penetração renovável energética desejada:
PEN_RENOVAVEL_BASE = 0.455
PEN_RENOVAVEL_ALTA = 0.80

# Penalidades
CUSTO_CURTAILMENT = 100.0       # R$/MWh
CUSTO_DEFICIT = 10000.0         # R$/MWh
CUSTO_VERTIMENTO = 0.10         # R$ por (m3/s)*h; penalização baixa para evitar vertimento sem necessidade

# Condição terminal dos reservatórios:
# 1.00 -> Vfinal >= Vinicial
FATOR_VOLUME_FINAL = 0.95

# Tolerâncias
TOL = 1e-7
TOL_SIMULTANEIDADE = 1e-5

# Pastas/arquivos
PASTA_ENTRADA = Path("Arquivos de Entrada")
PASTA_SAIDA = Path("Arquivos de Saida")

ARQUIVOS_ENTRADA_CANDIDATOS = [
    PASTA_SAIDA / "entradas_processadas.xlsx",
    PASTA_ENTRADA / "entradas_processadas.xlsx",
    Path("entradas_processadas.xlsx"),
]

ARQ_SAIDA = PASTA_SAIDA / "resultados_otimizacao.xlsx"


# ============================================================
# 2. CENÁRIOS
# ============================================================

CENARIOS = {
    "C1": {
        "penetracao_renovavel": PEN_RENOVAVEL_BASE,
        "com_uhr": False,
        "descricao": "Penetração renovável base - sem UHR",
    },
    "C2": {
        "penetracao_renovavel": PEN_RENOVAVEL_BASE,
        "com_uhr": True,
        "descricao": "Penetração renovável base - com UHR",
    },
    "C3": {
        "penetracao_renovavel": PEN_RENOVAVEL_ALTA,
        "com_uhr": False,
        "descricao": "Penetração renovável alta - sem UHR",
    },
    "C4": {
        "penetracao_renovavel": PEN_RENOVAVEL_ALTA,
        "com_uhr": True,
        "descricao": "Penetração renovável alta - com UHR",
    },
}


# ============================================================
# 3. FUNÇÕES AUXILIARES
# ============================================================

def localizar_arquivo_entrada():
    """Localiza o arquivo processado de entrada."""
    for caminho in ARQUIVOS_ENTRADA_CANDIDATOS:
        if caminho.exists():
            return caminho

    raise FileNotFoundError(
        "Arquivo entradas_processadas.xlsx não encontrado. "
        "Coloque-o em 'Arquivos de Entrada/' ou na pasta "
        "principal do projeto."
    )


def validar_colunas(df, colunas, nome_aba):
    faltantes = [c for c in colunas if c not in df.columns]
    if faltantes:
        raise ValueError(
            f"Aba '{nome_aba}' não contém as colunas: {faltantes}"
        )


def limpar_nome(nome):
    return str(nome).strip().upper()


# ============================================================
# 4. LEITURA DAS ENTRADAS PROCESSADAS
# ============================================================

def ler_entradas(caminho=None):
    """
    Lê somente as abas necessárias para o problema de otimização.
    """

    if caminho is None:
        caminho = localizar_arquivo_entrada()

    print(f"\nLendo arquivo de entrada:\n{caminho}")

    series = pd.read_excel(
        caminho,
        sheet_name="Series_Modelo"
    )

    uhe = pd.read_excel(
        caminho,
        sheet_name="Parametros_UHE"
    )

    ute = pd.read_excel(
        caminho,
        sheet_name="Parametros_UTE"
    )

    uhr = pd.read_excel(
        caminho,
        sheet_name="Parametros_UHR"
    )

    volumes = pd.read_excel(
        caminho,
        sheet_name="Volumes_Iniciais_ONS"
    )

    # --------------------------------------------------------
    # Validações
    # --------------------------------------------------------

    validar_colunas(
        series,
        [
            "hora",
            "data_hora",
            "carga_NE_MW",
            "fc_eolica_NE",
            "fc_solar_NE",
            "capacidade_eolica_NE_MW",
            "capacidade_solar_NE_MW",
            "afluencia_externa_tres_marias_m3s",
            "contrib_lateral_eq_sobradinho_m3s",
            "contrib_lateral_eq_itaparica_m3s",
            "contrib_lateral_eq_paulo_afonso_m3s",
            "contrib_lateral_eq_xingo_m3s",
            "defluencia_min_tres_marias_m3s",
            "defluencia_min_sobradinho_m3s",
            "defluencia_min_itaparica_m3s",
            "defluencia_min_paulo_afonso_m3s",
            "defluencia_min_xingo_m3s",
            "turbinamento_min_tres_marias_m3s",
            "turbinamento_min_sobradinho_m3s",
            "turbinamento_min_itaparica_m3s",
            "turbinamento_min_paulo_afonso_m3s",
            "turbinamento_min_xingo_m3s",
        ],
        "Series_Modelo",
    )

    validar_colunas(
        uhe,
        [
            "nome_modelo",
            "Qtur_max_m3s",
            "produtibilidade_MW_m3s",
            "Vmin_hm3",
            "Vmax_hm3",
            "Pmax_MW",
            "tipo_modelagem",
        ],
        "Parametros_UHE",
    )

    validar_colunas(
        ute,
        ["nome", "Pmax_MW", "CVU_R$_MWh"],
        "Parametros_UTE",
    )

    validar_colunas(
        uhr,
        [
            "Pger_max_MW",
            "Pbomb_max_MW",
            "eta_bomb",
            "eta_turb",
            "Emax_MWh",
            "Einicial_MWh",
            "Efinal_MWh",
            "custo_ciclo_R$_MWh",
        ],
        "Parametros_UHR",
    )

    validar_colunas(
        volumes,
        ["nome_modelo", "Vinicial_hm3"],
        "Volumes_Iniciais_ONS",
    )

    if len(series) != NUM_HORAS:
        raise ValueError(
            f"Esperados {NUM_HORAS} períodos em Series_Modelo; "
            f"encontrados {len(series)}."
        )

    series = (
        series
        .sort_values("hora")
        .reset_index(drop=True)
    )

    series["data_hora"] = pd.to_datetime(series["data_hora"])

    uhe["nome_modelo"] = uhe["nome_modelo"].map(limpar_nome)
    ute["nome"] = ute["nome"].map(limpar_nome)
    volumes["nome_modelo"] = volumes["nome_modelo"].map(limpar_nome)

    # Dicionário de volumes iniciais somente para reservatórios
    vol_ini = (
        volumes
        .dropna(subset=["Vinicial_hm3"])
        .set_index("nome_modelo")["Vinicial_hm3"]
        .astype(float)
        .to_dict()
    )

    # Uma única UHR
    if len(uhr) != 1:
        raise ValueError(
            "A aba Parametros_UHR deve possuir exatamente uma UHR."
        )

    dados = {
        "series": series,
        "uhe": uhe,
        "ute": ute,
        "uhr": uhr.iloc[0],
        "vol_ini": vol_ini,
    }

    return dados


# ============================================================
# 5. CALIBRAÇÃO DA CAPACIDADE RENOVÁVEL
# ============================================================

def calibrar_renovaveis(series, penetracao_alvo):
    """
    Escala conjuntamente as capacidades eólica e solar observadas
    para que a energia renovável DISPONÍVEL na semana corresponda
    à penetração energética desejada.

    Mantém:
      - os FC horários reais;
      - a proporção observada entre capacidade eólica e solar.

    Pen_REN =
        soma(Peol_disp + Psol_disp) / soma(carga)
    """

    carga = series["carga_NE_MW"].to_numpy(dtype=float) * FATOR_CARGA_EQUIVALENTE
    fc_eol = series["fc_eolica_NE"].to_numpy(dtype=float)
    fc_sol = series["fc_solar_NE"].to_numpy(dtype=float)

    cap_eol_obs = float(
        series["capacidade_eolica_NE_MW"].median()
    )
    cap_sol_obs = float(
        series["capacidade_solar_NE_MW"].median()
    )

    energia_carga = carga.sum()

    energia_ren_obs = (
        (fc_eol * cap_eol_obs)
        +
        (fc_sol * cap_sol_obs)
    ).sum()

    if energia_ren_obs <= 0:
        raise ValueError(
            "Energia renovável observada não positiva."
        )

    fator_escala = (
        penetracao_alvo * energia_carga / energia_ren_obs
    )

    cap_eol_modelo = cap_eol_obs * fator_escala
    cap_sol_modelo = cap_sol_obs * fator_escala

    eol_disp = fc_eol * cap_eol_modelo
    sol_disp = fc_sol * cap_sol_modelo

    penetracao_resultante = (
        (eol_disp.sum() + sol_disp.sum())
        / energia_carga
    )

    return {
        "fator_escala": fator_escala,
        "cap_eolica_MW": cap_eol_modelo,
        "cap_solar_MW": cap_sol_modelo,
        "eolica_disponivel_MW": eol_disp,
        "solar_disponivel_MW": sol_disp,
        "penetracao_resultante": penetracao_resultante,
        "energia_ren_disponivel_MWh":
            eol_disp.sum() + sol_disp.sum(),
        "energia_carga_MWh": energia_carga,
    }


# ============================================================
# 6. ÍNDICES DAS VARIÁVEIS
# ============================================================

def criar_indices(dados, com_uhr):
    """
    Cria fatias do vetor x usado pelo scipy.optimize.linprog.
    """

    uhe = dados["uhe"]
    ute = dados["ute"]

    reservatorios = (
        uhe.loc[
            uhe["tipo_modelagem"].str.upper() == "RESERVATORIO",
            "nome_modelo",
        ]
        .tolist()
    )

    fios = (
        uhe.loc[
            uhe["tipo_modelagem"].str.upper() == "FIO_DAGUA",
            "nome_modelo",
        ]
        .tolist()
    )

    nomes_uhe = uhe["nome_modelo"].tolist()
    nomes_ute = ute["nome"].tolist()

    idx = {}
    pos = 0

    def adicionar(chave, tamanho):
        nonlocal pos
        idx[chave] = np.arange(pos, pos + tamanho)
        pos += tamanho

    # UHEs: geração, turbinamento e vertimento
    for h in nomes_uhe:
        adicionar(("PH", h), NUM_HORAS)
        adicionar(("QTUR", h), NUM_HORAS)
        adicionar(("QVERT", h), NUM_HORAS)

    # Estados de volume apenas para reservatórios
    for h in reservatorios:
        adicionar(("V", h), NUM_HORAS + 1)

    # UTEs
    for g in nomes_ute:
        adicionar(("PT", g), NUM_HORAS)

    # Renováveis
    adicionar("PEOL", NUM_HORAS)
    adicionar("CEOL", NUM_HORAS)
    adicionar("PSOL", NUM_HORAS)
    adicionar("CSOL", NUM_HORAS)

    # Déficit
    adicionar("DEF", NUM_HORAS)

    # UHR
    if com_uhr:
        adicionar("PUHR", NUM_HORAS)
        adicionar("PBOMB", NUM_HORAS)
        adicionar("EUHR", NUM_HORAS + 1)

    return idx, pos, reservatorios, fios, nomes_uhe, nomes_ute


# ============================================================
# 7. MONTAGEM DO PROBLEMA
# ============================================================

def montar_problema(dados, penetracao_renovavel, com_uhr):
    """
    Monta c, bounds, A_eq, b_eq, A_ub e b_ub.
    """

    series = dados["series"]
    uhe = dados["uhe"]
    ute = dados["ute"]
    uhr = dados["uhr"]
    vol_ini = dados["vol_ini"]

    (
        idx,
        nvar,
        reservatorios,
        fios,
        nomes_uhe,
        nomes_ute,
    ) = criar_indices(dados, com_uhr)

    renov = calibrar_renovaveis(
        series,
        penetracao_renovavel
    )

    carga = series["carga_NE_MW"].to_numpy(dtype=float) * FATOR_CARGA_EQUIVALENTE
    eol_disp = renov["eolica_disponivel_MW"]
    sol_disp = renov["solar_disponivel_MW"]

    # --------------------------------------------------------
    # Parâmetros por UHE
    # --------------------------------------------------------

    param_uhe = (
        uhe
        .set_index("nome_modelo")
        .to_dict("index")
    )

    # Entrada externa da cabeceira e contribuições laterais
    # equivalentes dos trechos da cascata.
    afluencia_tm = series[
        "afluencia_externa_tres_marias_m3s"
    ].to_numpy(dtype=float)

    coluna_lateral = {
        "SOBRADINHO":
            "contrib_lateral_eq_sobradinho_m3s",
        "ITAPARICA":
            "contrib_lateral_eq_itaparica_m3s",
        "PAULO AFONSO":
            "contrib_lateral_eq_paulo_afonso_m3s",
        "XINGO":
            "contrib_lateral_eq_xingo_m3s",
    }

    laterais = {
        h: series[coluna_lateral[h]].to_numpy(dtype=float)
        for h in coluna_lateral
    }

    coluna_qmin = {
        "TRES MARIAS": "defluencia_min_tres_marias_m3s",
        "SOBRADINHO": "defluencia_min_sobradinho_m3s",
        "ITAPARICA": "defluencia_min_itaparica_m3s",
        "PAULO AFONSO": "defluencia_min_paulo_afonso_m3s",
        "XINGO": "defluencia_min_xingo_m3s",
    }
    qdef_min = {
        h: series[coluna_qmin[h]].to_numpy(dtype=float)
        for h in coluna_qmin
    }

    coluna_qtur_min = {
        "TRES MARIAS": "turbinamento_min_tres_marias_m3s",
        "SOBRADINHO": "turbinamento_min_sobradinho_m3s",
        "ITAPARICA": "turbinamento_min_itaparica_m3s",
        "PAULO AFONSO": "turbinamento_min_paulo_afonso_m3s",
        "XINGO": "turbinamento_min_xingo_m3s",
    }
    qtur_min = {
        h: series[coluna_qtur_min[h]].to_numpy(dtype=float)
        for h in coluna_qtur_min
    }

    # Encadeamento hidráulico adotado.
    montante = {
        "SOBRADINHO": "TRES MARIAS",
        "ITAPARICA": "SOBRADINHO",
        "PAULO AFONSO": "ITAPARICA",
        "XINGO": "PAULO AFONSO",
    }

    # --------------------------------------------------------
    # Função objetivo
    # --------------------------------------------------------

    c = np.zeros(nvar, dtype=float)

    # Custos térmicos
    param_ute = ute.set_index("nome").to_dict("index")

    for g in nomes_ute:
        c[idx[("PT", g)]] = float(
            param_ute[g]["CVU_R$_MWh"]
        )

    # Curtailment
    c[idx["CEOL"]] = CUSTO_CURTAILMENT
    c[idx["CSOL"]] = CUSTO_CURTAILMENT

    # Déficit
    c[idx["DEF"]] = CUSTO_DEFICIT

    # Penalização baixa de vertimento. Como QVERT está em m3/s e cada
    # período dura 1 h, o coeficiente tem unidade R$ por (m3/s)*h.
    for h in nomes_uhe:
        c[idx[("QVERT", h)]] = CUSTO_VERTIMENTO

    # Custo de ciclo da UHR lido diretamente da planilha Parametros_UHR.
    if com_uhr:
        custo_ciclo = float(uhr["custo_ciclo_R$_MWh"])
        c[idx["PUHR"]] = custo_ciclo
        c[idx["PBOMB"]] = custo_ciclo

    # --------------------------------------------------------
    # Limites das variáveis
    # --------------------------------------------------------

    bounds = [(0.0, None)] * nvar

    # UHEs
    for h in nomes_uhe:
        pmax = float(param_uhe[h]["Pmax_MW"])
        qmax = float(param_uhe[h]["Qtur_max_m3s"])

        for t in range(NUM_HORAS):
            bounds[idx[("PH", h)][t]] = (0.0, pmax)
            bounds[idx[("QTUR", h)][t]] = (0.0, qmax)
            bounds[idx[("QVERT", h)][t]] = (0.0, None)

    # Volumes
    for h in reservatorios:
        vmin = float(param_uhe[h]["Vmin_hm3"])
        vmax = float(param_uhe[h]["Vmax_hm3"])

        if h not in vol_ini:
            raise ValueError(
                f"Volume inicial não encontrado para {h}."
            )

        v0 = float(vol_ini[h])

        for t in range(NUM_HORAS + 1):
            bounds[idx[("V", h)][t]] = (vmin, vmax)

        # Estado inicial fixado
        bounds[idx[("V", h)][0]] = (v0, v0)

        # Condição terminal
        vfinal_min = max(
            vmin,
            FATOR_VOLUME_FINAL * v0
        )
        bounds[idx[("V", h)][NUM_HORAS]] = (
            vfinal_min,
            vmax
        )

    # UTEs
    for g in nomes_ute:
        pmax = float(param_ute[g]["Pmax_MW"])
        for t in range(NUM_HORAS):
            bounds[idx[("PT", g)][t]] = (0.0, pmax)

    # Renováveis
    for t in range(NUM_HORAS):
        bounds[idx["PEOL"][t]] = (0.0, float(eol_disp[t]))
        bounds[idx["CEOL"][t]] = (0.0, float(eol_disp[t]))
        bounds[idx["PSOL"][t]] = (0.0, float(sol_disp[t]))
        bounds[idx["CSOL"][t]] = (0.0, float(sol_disp[t]))

    # Déficit
    for t in range(NUM_HORAS):
        bounds[idx["DEF"][t]] = (0.0, float(carga[t]))

    # UHR
    if com_uhr:
        pger_max = float(uhr["Pger_max_MW"])
        pbomb_max = float(uhr["Pbomb_max_MW"])
        emax = float(uhr["Emax_MWh"])
        eini = float(uhr["Einicial_MWh"])
        efinal = float(uhr["Efinal_MWh"])

        for t in range(NUM_HORAS):
            bounds[idx["PUHR"][t]] = (0.0, pger_max)
            bounds[idx["PBOMB"][t]] = (0.0, pbomb_max)

        for t in range(NUM_HORAS + 1):
            bounds[idx["EUHR"][t]] = (0.0, emax)

        bounds[idx["EUHR"][0]] = (eini, eini)
        bounds[idx["EUHR"][NUM_HORAS]] = (
            efinal,
            efinal
        )

    # --------------------------------------------------------
    # Número de restrições de igualdade
    # --------------------------------------------------------

    # PH = rho * Qtur
    n_eq_hidro_prod = len(nomes_uhe) * NUM_HORAS

    # Balanços dos reservatórios
    n_eq_bal_res = len(reservatorios) * NUM_HORAS

    # Fio d'água: Qtur + Qvert = A_rep
    n_eq_fio = len(fios) * NUM_HORAS

    # Disponibilidade renovável
    n_eq_ren = 2 * NUM_HORAS

    # Atendimento à demanda
    n_eq_eletrico = NUM_HORAS

    # UHR
    n_eq_uhr = NUM_HORAS if com_uhr else 0

    n_eq = (
        n_eq_hidro_prod
        + n_eq_bal_res
        + n_eq_fio
        + n_eq_ren
        + n_eq_eletrico
        + n_eq_uhr
    )

    A_eq = lil_matrix((n_eq, nvar), dtype=float)
    b_eq = np.zeros(n_eq, dtype=float)

    linha = 0

    # --------------------------------------------------------
    # 7.1 Produtibilidade hidráulica
    #
    # PH = rho * Qtur
    # --------------------------------------------------------

    for h in nomes_uhe:
        rho = float(
            param_uhe[h]["produtibilidade_MW_m3s"]
        )

        for t in range(NUM_HORAS):
            A_eq[linha, idx[("PH", h)][t]] = 1.0
            A_eq[linha, idx[("QTUR", h)][t]] = -rho
            b_eq[linha] = 0.0
            linha += 1

    # --------------------------------------------------------
    # 7.2 Balanço hidráulico dos reservatórios acoplados
    #
    # Três Marias:
    # V(t+1) = V(t) + k*(Aext - Qdef)
    #
    # Demais reservatórios:
    # V_j(t+1) = V_j(t) + k*(Qdef_i + L_j - Qdef_j)
    #
    # Qdef = Qtur + Qvert
    # Tempo de viagem entre usinas desprezado.
    # --------------------------------------------------------

    for h in reservatorios:
        for t in range(NUM_HORAS):
            A_eq[linha, idx[("V", h)][t + 1]] = 1.0
            A_eq[linha, idx[("V", h)][t]] = -1.0
            A_eq[linha, idx[("QTUR", h)][t]] = FATOR_HM3
            A_eq[linha, idx[("QVERT", h)][t]] = FATOR_HM3

            if h == "TRES MARIAS":
                b_eq[linha] = (
                    FATOR_HM3 * float(afluencia_tm[t])
                )
            else:
                h_mont = montante[h]
                A_eq[linha, idx[("QTUR", h_mont)][t]] = -FATOR_HM3
                A_eq[linha, idx[("QVERT", h_mont)][t]] = -FATOR_HM3
                b_eq[linha] = (
                    FATOR_HM3 * float(laterais[h][t])
                )

            linha += 1

    # --------------------------------------------------------
    # 7.3 UHEs a fio d'água acopladas
    #
    # Qdef_j = Qdef_i + L_j
    # Qdef = Qtur + Qvert
    # --------------------------------------------------------

    for h in fios:
        h_mont = montante[h]

        for t in range(NUM_HORAS):
            A_eq[linha, idx[("QTUR", h)][t]] = 1.0
            A_eq[linha, idx[("QVERT", h)][t]] = 1.0
            A_eq[linha, idx[("QTUR", h_mont)][t]] = -1.0
            A_eq[linha, idx[("QVERT", h_mont)][t]] = -1.0
            b_eq[linha] = float(laterais[h][t])
            linha += 1

    # --------------------------------------------------------
    # 7.4 Disponibilidade renovável
    #
    # Peol + Curt_eol = Peol_disp
    # Psol + Curt_sol = Psol_disp
    # --------------------------------------------------------

    for t in range(NUM_HORAS):
        A_eq[linha, idx["PEOL"][t]] = 1.0
        A_eq[linha, idx["CEOL"][t]] = 1.0
        b_eq[linha] = float(eol_disp[t])
        linha += 1

    for t in range(NUM_HORAS):
        A_eq[linha, idx["PSOL"][t]] = 1.0
        A_eq[linha, idx["CSOL"][t]] = 1.0
        b_eq[linha] = float(sol_disp[t])
        linha += 1

    # --------------------------------------------------------
    # 7.5 Atendimento à demanda
    #
    # UHE + UTE + EOL + SOL + UHR + DEF
    #     = CARGA + BOMBEAMENTO
    # --------------------------------------------------------

    for t in range(NUM_HORAS):

        for h in nomes_uhe:
            A_eq[linha, idx[("PH", h)][t]] = 1.0

        for g in nomes_ute:
            A_eq[linha, idx[("PT", g)][t]] = 1.0

        A_eq[linha, idx["PEOL"][t]] = 1.0
        A_eq[linha, idx["PSOL"][t]] = 1.0
        A_eq[linha, idx["DEF"][t]] = 1.0

        if com_uhr:
            A_eq[linha, idx["PUHR"][t]] = 1.0
            A_eq[linha, idx["PBOMB"][t]] = -1.0

        b_eq[linha] = float(carga[t])
        linha += 1

    # --------------------------------------------------------
    # 7.6 Estado energético da UHR
    #
    # E(t+1) = E(t)
    #          + eta_bomb * Pbomb
    #          - Puhr / eta_turb
    # --------------------------------------------------------

    if com_uhr:

        eta_bomb = float(uhr["eta_bomb"])
        eta_turb = float(uhr["eta_turb"])

        if not (0 < eta_bomb <= 1):
            raise ValueError("Eficiência de bombeamento inválida.")

        if not (0 < eta_turb <= 1):
            raise ValueError("Eficiência de turbinamento inválida.")

        for t in range(NUM_HORAS):

            A_eq[linha, idx["EUHR"][t + 1]] = 1.0
            A_eq[linha, idx["EUHR"][t]] = -1.0
            A_eq[linha, idx["PBOMB"][t]] = -eta_bomb
            A_eq[linha, idx["PUHR"][t]] = 1.0 / eta_turb

            b_eq[linha] = 0.0
            linha += 1

    if linha != n_eq:
        raise RuntimeError(
            f"Erro interno: {linha} restrições montadas, "
            f"mas eram esperadas {n_eq}."
        )

    # --------------------------------------------------------
    # 7.7 Restrição linear de potência total da UHR
    #
    # Pger(t) + Pbomb(t) <= Pmax_UHR
    #
    # Esta restrição não impõe exclusividade exata (isso exigiria
    # variável binária/MILP), mas reduz a possibilidade de operação
    # simultânea mantendo a formulação como Programação Linear.
    # --------------------------------------------------------
    # Além da capacidade compartilhada da UHR, impõe-se um piso
    # de defluência total em cada UHE:
    #     Qtur(h,t) + Qvert(h,t) >= Qdef_min(h,t)
    # ou, na forma A_ub x <= b_ub:
    #    -Qtur - Qvert <= -Qdef_min.
    n_qdef_min = len(nomes_uhe) * NUM_HORAS
    n_qtur_min = len(nomes_uhe) * NUM_HORAS
    n_uhr = NUM_HORAS if com_uhr else 0
    A_ub = lil_matrix((n_qdef_min + n_qtur_min + n_uhr, nvar), dtype=float)
    b_ub = np.zeros(n_qdef_min + n_qtur_min + n_uhr, dtype=float)

    linha = 0
    for h in nomes_uhe:
        for t in range(NUM_HORAS):
            A_ub[linha, idx[("QTUR", h)][t]] = -1.0
            A_ub[linha, idx[("QVERT", h)][t]] = -1.0
            b_ub[linha] = -float(qdef_min[h][t])
            linha += 1

    # Turbinamento mínimo representativo:
    #     Qtur(h,t) >= Qtur_min(h,t)
    # ou, em A_ub x <= b_ub:
    #    -Qtur <= -Qtur_min.
    for h in nomes_uhe:
        for t in range(NUM_HORAS):
            A_ub[linha, idx[("QTUR", h)][t]] = -1.0
            b_ub[linha] = -float(qtur_min[h][t])
            linha += 1

    if com_uhr:
        pger_max = float(uhr["Pger_max_MW"])
        pbomb_max = float(uhr["Pbomb_max_MW"])
        for t in range(NUM_HORAS):
            # Forma normalizada, válida mesmo se as potências nominais
            # de geração e bombeamento forem diferentes.
            A_ub[linha, idx["PUHR"][t]] = 1.0 / pger_max
            A_ub[linha, idx["PBOMB"][t]] = 1.0 / pbomb_max
            b_ub[linha] = 1.0
            linha += 1

    A_ub = A_ub.tocsr()

    metadados = {
        "renovaveis": renov,
        "idx": idx,
        "reservatorios": reservatorios,
        "fios": fios,
        "nomes_uhe": nomes_uhe,
        "nomes_ute": nomes_ute,
        "montante": montante,
        "laterais": laterais,
        "afluencia_tm": afluencia_tm,
        "qdef_min": qdef_min,
    }

    return (
        c,
        bounds,
        A_eq.tocsr(),
        b_eq,
        A_ub,
        b_ub,
        metadados,
    )


# ============================================================
# 8. EXTRAÇÃO DOS RESULTADOS
# ============================================================

def extrair_resultados(
    nome_cenario,
    dados,
    solucao,
    metadados,
    com_uhr,
):
    """
    Converte o vetor ótimo em DataFrame horário e indicadores.
    """

    x = solucao.x
    series = dados["series"].copy()
    uhe = dados["uhe"]
    ute = dados["ute"]
    uhr = dados["uhr"]

    idx = metadados["idx"]
    nomes_uhe = metadados["nomes_uhe"]
    nomes_ute = metadados["nomes_ute"]
    reservatorios = metadados["reservatorios"]
    renov = metadados["renovaveis"]

    df = pd.DataFrame({
        "hora": series["hora"].to_numpy(),
        "data_hora": series["data_hora"].to_numpy(),
        "carga_MW": series["carga_NE_MW"].to_numpy(dtype=float) * FATOR_CARGA_EQUIVALENTE,
        "eolica_disponivel_MW":
            renov["eolica_disponivel_MW"],
        "solar_disponivel_MW":
            renov["solar_disponivel_MW"],
        "geracao_eolica_MW":
            x[idx["PEOL"]],
        "curtailment_eolico_MW":
            x[idx["CEOL"]],
        "geracao_solar_MW":
            x[idx["PSOL"]],
        "curtailment_solar_MW":
            x[idx["CSOL"]],
        "deficit_MW":
            x[idx["DEF"]],
        "afluencia_externa_tres_marias_m3s":
            series["afluencia_externa_tres_marias_m3s"].to_numpy(dtype=float),
        "contrib_lateral_eq_sobradinho_m3s":
            series["contrib_lateral_eq_sobradinho_m3s"].to_numpy(dtype=float),
        "contrib_lateral_eq_itaparica_m3s":
            series["contrib_lateral_eq_itaparica_m3s"].to_numpy(dtype=float),
        "contrib_lateral_eq_paulo_afonso_m3s":
            series["contrib_lateral_eq_paulo_afonso_m3s"].to_numpy(dtype=float),
        "contrib_lateral_eq_xingo_m3s":
            series["contrib_lateral_eq_xingo_m3s"].to_numpy(dtype=float),
    })

    # UHEs
    for h in nomes_uhe:
        chave = h.lower().replace(" ", "_")

        df[f"geracao_{chave}_MW"] = x[idx[("PH", h)]]
        df[f"qtur_{chave}_m3s"] = x[idx[("QTUR", h)]]
        df[f"qvert_{chave}_m3s"] = x[idx[("QVERT", h)]]
        df[f"qdef_{chave}_m3s"] = (
            df[f"qtur_{chave}_m3s"]
            + df[f"qvert_{chave}_m3s"]
        )

    # Vazão total de entrada em cada aproveitamento da cascata.
    df["qentrada_tres_marias_m3s"] = (
        df["afluencia_externa_tres_marias_m3s"]
    )
    df["qentrada_sobradinho_m3s"] = (
        df["qdef_tres_marias_m3s"]
        + df["contrib_lateral_eq_sobradinho_m3s"]
    )
    df["qentrada_itaparica_m3s"] = (
        df["qdef_sobradinho_m3s"]
        + df["contrib_lateral_eq_itaparica_m3s"]
    )
    df["qentrada_paulo_afonso_m3s"] = (
        df["qdef_itaparica_m3s"]
        + df["contrib_lateral_eq_paulo_afonso_m3s"]
    )
    df["qentrada_xingo_m3s"] = (
        df["qdef_paulo_afonso_m3s"]
        + df["contrib_lateral_eq_xingo_m3s"]
    )

    # Estados de volume no início de cada hora
    for h in reservatorios:
        chave = h.lower().replace(" ", "_")
        df[f"volume_{chave}_hm3"] = x[idx[("V", h)]][:-1]

    # UTEs
    for g in nomes_ute:
        chave = g.lower().replace(" ", "_")
        df[f"geracao_{chave}_MW"] = x[idx[("PT", g)]]

    # UHR
    if com_uhr:
        df["geracao_uhr_MW"] = x[idx["PUHR"]]
        df["bombeamento_uhr_MW"] = x[idx["PBOMB"]]
        df["energia_uhr_MWh"] = x[idx["EUHR"]][:-1]
    else:
        df["geracao_uhr_MW"] = 0.0
        df["bombeamento_uhr_MW"] = 0.0
        df["energia_uhr_MWh"] = np.nan

    df["curtailment_total_MW"] = (
        df["curtailment_eolico_MW"]
        + df["curtailment_solar_MW"]
    )

    # Totais horários agregados
    col_hidro = [
        f"geracao_{h.lower().replace(' ', '_')}_MW"
        for h in nomes_uhe
    ]

    col_termica = [
        f"geracao_{g.lower().replace(' ', '_')}_MW"
        for g in nomes_ute
    ]

    df["geracao_hidraulica_total_MW"] = df[col_hidro].sum(axis=1)
    df["geracao_termica_total_MW"] = df[col_termica].sum(axis=1)
    df["geracao_renovavel_total_MW"] = (
        df["geracao_eolica_MW"]
        + df["geracao_solar_MW"]
    )

    # --------------------------------------------------------
    # Indicadores
    # --------------------------------------------------------

    ger_termica = df["geracao_termica_total_MW"].sum()
    curt_eol = df["curtailment_eolico_MW"].sum()
    curt_sol = df["curtailment_solar_MW"].sum()
    deficit = df["deficit_MW"].sum()
    bombeamento = df["bombeamento_uhr_MW"].sum()
    ger_uhr = df["geracao_uhr_MW"].sum()

    energia_ren_disp = (
        df["eolica_disponivel_MW"].sum()
        + df["solar_disponivel_MW"].sum()
    )

    energia_ren_aprov = df["geracao_renovavel_total_MW"].sum()

    taxa_curt = (
        100.0 * (curt_eol + curt_sol) / energia_ren_disp
        if energia_ren_disp > TOL
        else 0.0
    )

    # Custo térmico separado
    param_ute = ute.set_index("nome").to_dict("index")
    custo_termico = 0.0

    for g in nomes_ute:
        chave = g.lower().replace(" ", "_")
        custo_termico += (
            df[f"geracao_{chave}_MW"].sum()
            * float(param_ute[g]["CVU_R$_MWh"])
        )

    custo_curt = (
        (curt_eol + curt_sol)
        * CUSTO_CURTAILMENT
    )

    custo_def = deficit * CUSTO_DEFICIT

    custo_ciclo = 0.0
    if com_uhr:
        custo_ciclo = (
            (bombeamento + ger_uhr)
            * float(uhr["custo_ciclo_R$_MWh"])
        )

    # Custo de vertimento: mesma expressão usada na FOB.
    # QVERT está em m3/s e cada período tem duração de 1 h.
    vertimento_total = sum(
        df[f"qvert_{h.lower().replace(' ', '_')}_m3s"].sum()
        for h in nomes_uhe
    )
    custo_vertimento = vertimento_total * CUSTO_VERTIMENTO

    resumo = {
        "cenario": nome_cenario,
        "status": solucao.message,
        "FOB_R$": float(solucao.fun),
        "custo_termico_R$": custo_termico,
        "custo_curtailment_R$": custo_curt,
        "custo_deficit_R$": custo_def,
        "custo_ciclo_UHR_R$": custo_ciclo,
        "custo_vertimento_R$": custo_vertimento,
        "penetracao_REN_disponivel_pct":
            100.0 * renov["penetracao_resultante"],
        "fator_escala_REN":
            renov["fator_escala"],
        "cap_eolica_equivalente_MW":
            renov["cap_eolica_MW"],
        "cap_solar_equivalente_MW":
            renov["cap_solar_MW"],
        "energia_carga_MWh":
            df["carga_MW"].sum(),
        "energia_REN_disponivel_MWh":
            energia_ren_disp,
        "energia_REN_aproveitada_MWh":
            energia_ren_aprov,
        "curtailment_eolico_MWh":
            curt_eol,
        "curtailment_solar_MWh":
            curt_sol,
        "curtailment_total_MWh":
            curt_eol + curt_sol,
        "taxa_curtailment_pct":
            taxa_curt,
        "geracao_termica_MWh":
            ger_termica,
        "deficit_MWh":
            deficit,
        "geracao_UHR_MWh":
            ger_uhr,
        "bombeamento_UHR_MWh":
            bombeamento,
        "carga_media_MWmed": df["carga_MW"].mean(),
        "geracao_hidraulica_MWmed": df["geracao_hidraulica_total_MW"].mean(),
        "geracao_termica_MWmed": df["geracao_termica_total_MW"].mean(),
        "renovavel_disponivel_MWmed": (df["eolica_disponivel_MW"] + df["solar_disponivel_MW"]).mean(),
        "renovavel_aproveitada_MWmed": df["geracao_renovavel_total_MW"].mean(),
        "curtailment_total_MWmed": df["curtailment_total_MW"].mean(),
        "deficit_MWmed": df["deficit_MW"].mean(),
        "geracao_UHR_MWmed": df["geracao_uhr_MW"].mean(),
        "bombeamento_UHR_MWmed": df["bombeamento_uhr_MW"].mean(),
        "vertimento_total_m3s_med": vertimento_total / NUM_HORAS,
    }

    # Volumes finais
    for h in reservatorios:
        chave = h.lower().replace(" ", "_")
        resumo[f"volume_inicial_{chave}_hm3"] = (
            float(x[idx[("V", h)]][0])
        )
        resumo[f"volume_final_{chave}_hm3"] = (
            float(x[idx[("V", h)]][-1])
        )

    if com_uhr:
        resumo["energia_UHR_inicial_MWh"] = float(
            x[idx["EUHR"]][0]
        )
        resumo["energia_UHR_final_MWh"] = float(
            x[idx["EUHR"]][-1]
        )
    else:
        resumo["energia_UHR_inicial_MWh"] = np.nan
        resumo["energia_UHR_final_MWh"] = np.nan

    return df, resumo


# ============================================================
# 9. VALIDAÇÕES PÓS-SOLUÇÃO
# ============================================================

def validar_solucao(
    nome_cenario,
    df,
    resumo,
    com_uhr,
):
    """
    Faz verificações automáticas importantes para interpretação.
    """

    avisos = []

    # Déficit
    if resumo["deficit_MWh"] > 1e-3:
        avisos.append(
            f"{nome_cenario}: houve déficit de "
            f"{resumo['deficit_MWh']:.2f} MWh."
        )

    # Simultaneidade UHR
    if com_uhr:
        simult = df[
            (df["geracao_uhr_MW"] > TOL_SIMULTANEIDADE)
            &
            (df["bombeamento_uhr_MW"] > TOL_SIMULTANEIDADE)
        ]

        if not simult.empty:
            avisos.append(
                f"{nome_cenario}: operação simultânea da UHR "
                f"em {len(simult)} hora(s)."
            )

            print(
                "\nATENÇÃO - operação simultânea da UHR:"
            )
            print(
                simult[
                    [
                        "hora",
                        "data_hora",
                        "geracao_uhr_MW",
                        "bombeamento_uhr_MW",
                    ]
                ].to_string(index=False)
            )

    # Curtailment
    if resumo["curtailment_total_MWh"] > 1e-3:
        avisos.append(
            f"{nome_cenario}: curtailment total = "
            f"{resumo['curtailment_total_MWh']:.2f} MWh "
            f"({resumo['taxa_curtailment_pct']:.2f}% "
            "da energia renovável disponível)."
        )

    return avisos


# ============================================================
# 10. RESOLUÇÃO DE UM CENÁRIO
# ============================================================

def resolver_cenario(
    nome_cenario,
    dados,
    penetracao_renovavel,
    com_uhr,
):
    """
    Monta e resolve um cenário com HiGHS.
    """

    print("\n" + "=" * 70)
    print(f"RESOLVENDO {nome_cenario}")
    print(
        f"Penetração renovável alvo: "
        f"{100 * penetracao_renovavel:.1f}%"
    )
    print(f"UHR: {'SIM' if com_uhr else 'NÃO'}")
    print("=" * 70)

    (
        c,
        bounds,
        A_eq,
        b_eq,
        A_ub,
        b_ub,
        metadados,
    ) = montar_problema(
        dados,
        penetracao_renovavel,
        com_uhr,
    )

    solucao = linprog(
        c=c,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
        options={
            "presolve": True,
        },
    )

    if not solucao.success:
        raise RuntimeError(
            f"{nome_cenario} não foi resolvido.\n"
            f"Status: {solucao.status}\n"
            f"Mensagem: {solucao.message}"
        )

    df, resumo = extrair_resultados(
        nome_cenario,
        dados,
        solucao,
        metadados,
        com_uhr,
    )

    avisos = validar_solucao(
        nome_cenario,
        df,
        resumo,
        com_uhr,
    )

    print(f"\nFOB: R$ {resumo['FOB_R$']:,.2f}")
    print(
        "Penetração renovável disponível: "
        f"{resumo['penetracao_REN_disponivel_pct']:.2f}%"
    )
    print(
        "Curtailment: "
        f"{resumo['curtailment_total_MWh']:,.2f} MWh"
    )
    print(
        "Geração térmica: "
        f"{resumo['geracao_termica_MWh']:,.2f} MWh"
    )
    print(
        "Déficit: "
        f"{resumo['deficit_MWh']:,.2f} MWh"
    )

    if com_uhr:
        print(
            "Bombeamento UHR: "
            f"{resumo['bombeamento_UHR_MWh']:,.2f} MWh"
        )
        print(
            "Geração UHR: "
            f"{resumo['geracao_UHR_MWh']:,.2f} MWh"
        )

    return {
        "df": df,
        "resumo": resumo,
        "avisos": avisos,
        "solucao": solucao,
    }


# ============================================================
# 11. COMPARAÇÕES ENTRE CENÁRIOS
# ============================================================

def montar_comparacoes(resumos):
    """
    Compara C1 x C2 e C3 x C4 para quantificar o efeito da UHR.
    """

    df = pd.DataFrame(resumos).set_index("cenario")

    linhas = []

    for sem_uhr, com_uhr, nivel in [
        ("C1", "C2", "Base"),
        ("C3", "C4", "Alta"),
    ]:

        a = df.loc[sem_uhr]
        b = df.loc[com_uhr]

        def reducao_percentual(valor_sem, valor_com):
            if abs(valor_sem) <= TOL:
                return np.nan
            return (
                100.0
                * (valor_sem - valor_com)
                / valor_sem
            )

        linhas.append({
            "nivel_renovavel": nivel,
            "cenario_sem_UHR": sem_uhr,
            "cenario_com_UHR": com_uhr,

            "reducao_FOB_R$":
                a["FOB_R$"] - b["FOB_R$"],

            "reducao_FOB_pct":
                reducao_percentual(
                    a["FOB_R$"],
                    b["FOB_R$"],
                ),

            "reducao_curtailment_MWh":
                a["curtailment_total_MWh"]
                - b["curtailment_total_MWh"],

            "reducao_curtailment_pct":
                reducao_percentual(
                    a["curtailment_total_MWh"],
                    b["curtailment_total_MWh"],
                ),

            "reducao_termica_MWh":
                a["geracao_termica_MWh"]
                - b["geracao_termica_MWh"],

            "reducao_termica_pct":
                reducao_percentual(
                    a["geracao_termica_MWh"],
                    b["geracao_termica_MWh"],
                ),

            "reducao_deficit_MWh":
                a["deficit_MWh"]
                - b["deficit_MWh"],

            "bombeamento_UHR_MWh":
                b["bombeamento_UHR_MWh"],

            "geracao_UHR_MWh":
                b["geracao_UHR_MWh"],
        })

    return pd.DataFrame(linhas)


# ============================================================
# 12. EXPORTAÇÃO
# ============================================================

def exportar_resultados(resultados):
    """
    Exporta resultados horários, resumo, comparações e avisos.
    """

    PASTA_SAIDA.mkdir(
        parents=True,
        exist_ok=True
    )

    resumos = [
        resultados[c]["resumo"]
        for c in CENARIOS
    ]

    df_resumo = pd.DataFrame(resumos)
    df_comparacoes = montar_comparacoes(resumos)

    linhas_avisos = []
    for cenario, resultado in resultados.items():
        if resultado["avisos"]:
            for aviso in resultado["avisos"]:
                linhas_avisos.append({
                    "cenario": cenario,
                    "aviso": aviso,
                })
        else:
            linhas_avisos.append({
                "cenario": cenario,
                "aviso": "Nenhum aviso.",
            })

    df_avisos = pd.DataFrame(linhas_avisos)

    # Parâmetros dos cenários
    df_cenarios = pd.DataFrame([
        {
            "cenario": nome,
            "descricao": cfg["descricao"],
            "penetracao_REN_alvo_pct":
                100.0 * cfg["penetracao_renovavel"],
            "com_UHR": cfg["com_uhr"],
        }
        for nome, cfg in CENARIOS.items()
    ])

    with pd.ExcelWriter(
        ARQ_SAIDA,
        engine="openpyxl"
    ) as writer:

        df_resumo.to_excel(
            writer,
            sheet_name="Resumo",
            index=False
        )

        df_comparacoes.to_excel(
            writer,
            sheet_name="Comparacoes",
            index=False
        )

        df_cenarios.to_excel(
            writer,
            sheet_name="Cenarios",
            index=False
        )

        df_avisos.to_excel(
            writer,
            sheet_name="Avisos",
            index=False
        )

        for cenario in CENARIOS:
            resultados[cenario]["df"].to_excel(
                writer,
                sheet_name=cenario,
                index=False
            )

    print(
        f"\nResultados exportados para:\n{ARQ_SAIDA.resolve()}"
    )


# ============================================================
# 13. DIAGNÓSTICO DOS CENÁRIOS RENOVÁVEIS
# ============================================================

def imprimir_diagnostico_renovavel(dados):
    """
    Mostra as capacidades equivalentes produzidas pelos níveis
    de penetração escolhidos antes da resolução.
    """

    series = dados["series"]

    print("\n" + "=" * 70)
    print("CALIBRAÇÃO DOS CENÁRIOS RENOVÁVEIS")
    print("=" * 70)

    for nome, pen in [
        ("BASE", PEN_RENOVAVEL_BASE),
        ("ALTA", PEN_RENOVAVEL_ALTA),
    ]:

        r = calibrar_renovaveis(
            series,
            pen
        )

        print(f"\nCenário {nome}:")
        print(
            f"  Penetração energética: "
            f"{100*r['penetracao_resultante']:.2f}%"
        )
        print(
            f"  Fator sobre capacidade observada: "
            f"{r['fator_escala']:.4f}"
        )
        print(
            f"  Eólica equivalente: "
            f"{r['cap_eolica_MW']:.1f} MW"
        )
        print(
            f"  Solar equivalente: "
            f"{r['cap_solar_MW']:.1f} MW"
        )


# ============================================================
# 14. EXECUÇÃO PRINCIPAL
# ============================================================

def main():

    warnings.filterwarnings(
        "ignore",
        category=RuntimeWarning
    )

    dados = ler_entradas()

    imprimir_diagnostico_renovavel(
        dados
    )

    resultados = {}

    for nome, cfg in CENARIOS.items():

        resultados[nome] = resolver_cenario(
            nome_cenario=nome,
            dados=dados,
            penetracao_renovavel=
                cfg["penetracao_renovavel"],
            com_uhr=
                cfg["com_uhr"],
        )

    exportar_resultados(
        resultados
    )

    print("\n" + "=" * 70)
    print("OTIMIZAÇÃO FINALIZADA")
    print("=" * 70)

    resumo = pd.DataFrame(
        [
            resultados[c]["resumo"]
            for c in CENARIOS
        ]
    )

    colunas = [
        "cenario",
        "FOB_R$",
        "penetracao_REN_disponivel_pct",
        "geracao_termica_MWh",
        "curtailment_total_MWh",
        "deficit_MWh",
        "bombeamento_UHR_MWh",
        "geracao_UHR_MWh",
    ]

    print(
        "\nResumo dos cenários:\n"
    )

    print(
        resumo[colunas].to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
