from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# ============================================================
# GRÁFICOS FINAIS — ARTIGO E APRESENTAÇÃO
# ============================================================
# Entrada: resultados_otimizacao.xlsx
# Saídas:
#   Graficos/Artigo        -> PNG 300 dpi + PDF
#   Graficos/Apresentacao  -> PNG
#
# Convenções:
#   - valores agregados semanais: MWmed
#   - séries horárias / potência máxima: MW
#   - bombeamento da UHR aparece negativo apenas como recurso visual
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

CANDIDATOS = [BASE_DIR / "Arquivos de Saida" / "resultados_otimizacao.xlsx"]
ARQ_RESULTADOS = next((p for p in CANDIDATOS if p.exists()), CANDIDATOS[0])

DIR = BASE_DIR / "Graficos"
DIR_ARTIGO = DIR / "Artigo"
DIR_APRESENTACAO = DIR / "Apresentacao"

CENARIOS = ["C1", "C2", "C3", "C4"]
ROTULOS = {
    "C1": "C1\n45,5% REN\nsem UHR",
    "C2": "C2\n45,5% REN\ncom UHR",
    "C3": "C3\n80% REN\nsem UHR",
    "C4": "C4\n80% REN\ncom UHR",
}

DPI_ARTIGO = 300
DPI_APRESENTACAO = 180
LARGURA_JANELA = 24

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "figure.titlesize": 14,
})


def carregar():
    if not ARQ_RESULTADOS.exists():
        raise FileNotFoundError(
            "Coloque este script na mesma pasta de resultados_otimizacao.xlsx."
        )
    xls = pd.ExcelFile(ARQ_RESULTADOS)
    resumo = pd.read_excel(ARQ_RESULTADOS, sheet_name="Resumo")
    cen = {c: pd.read_excel(ARQ_RESULTADOS, sheet_name=c) for c in CENARIOS}
    return resumo, cen


def preparar_pastas():
    DIR_ARTIGO.mkdir(parents=True, exist_ok=True)
    DIR_APRESENTACAO.mkdir(parents=True, exist_ok=True)


def salvar(fig, nome):
    fig.tight_layout()
    fig.savefig(DIR_ARTIGO / f"{nome}.png", dpi=DPI_ARTIGO, bbox_inches="tight")
    fig.savefig(DIR_ARTIGO / f"{nome}.pdf", bbox_inches="tight")
    fig.savefig(DIR_APRESENTACAO / f"{nome}.png",
                dpi=DPI_APRESENTACAO, bbox_inches="tight")
    plt.close(fig)


def horas(df):
    return pd.to_numeric(df["hora"], errors="coerce").to_numpy(float)


def s(df, col):
    return pd.to_numeric(df[col], errors="coerce").fillna(0).to_numpy(float)


def recorte(df, ini, fim):
    h = horas(df)
    return df.loc[(h >= ini) & (h <= fim)].copy()


def janela_evento(df, coluna, largura=LARGURA_JANELA):
    y = s(df, coluna)
    h = horas(df)
    centro = int(h[int(np.argmax(y))])
    hmin, hmax = int(np.min(h)), int(np.max(h))
    ini = max(hmin, centro - largura // 2)
    fim = min(hmax, ini + largura - 1)
    ini = max(hmin, fim - largura + 1)
    return ini, fim


def fmt_pt(v, casas=1):
    return f"{v:,.{casas}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def resumo_val(resumo, c, col):
    return float(resumo.loc[resumo["cenario"] == c, col].iloc[0])


# ============================================================
# FIGURAS 01A/01B — RESULTADOS AGREGADOS POR PAR DE CENÁRIOS
# A separação evita que a escala de C3/C4 esconda a geração térmica
# e a operação da UHR em C1/C2.
# ============================================================

def fig_resultados_agregados_par(resumo, cenarios, nome, titulo):
    x = np.arange(len(cenarios))
    w = 0.19

    term = [resumo_val(resumo, c, "geracao_termica_MWmed") for c in cenarios]
    curt = [resumo_val(resumo, c, "curtailment_total_MWmed") for c in cenarios]
    guhr = [resumo_val(resumo, c, "geracao_UHR_MWmed") for c in cenarios]
    bomb = [resumo_val(resumo, c, "bombeamento_UHR_MWmed") for c in cenarios]

    fig, ax = plt.subplots(figsize=(8.4, 5.1))
    barras = [
        ax.bar(x - 1.5*w, term, w, label="Geração térmica"),
        ax.bar(x - 0.5*w, curt, w, label="Curtailment simulado"),
        ax.bar(x + 0.5*w, guhr, w, label="Geração UHR"),
        ax.bar(x + 1.5*w, bomb, w, label="Bombeamento UHR"),
    ]

    ax.set_xticks(x)
    ax.set_xticklabels([ROTULOS[c] for c in cenarios])
    ax.set_ylabel("Potência média (MWmed)")
    ax.set_title(titulo)
    ax.grid(axis="y", alpha=.25)
    ax.legend(ncol=2)

    ymax = max(term + curt + guhr + bomb + [1.0])
    ax.set_ylim(0, ymax * 1.22)

    for grupo in barras:
        for b in grupo:
            v = b.get_height()
            if v > 1e-8:
                ax.text(
                    b.get_x() + b.get_width()/2, v,
                    fmt_pt(v, 2 if ymax < 50 else 1),
                    ha="center", va="bottom", fontsize=8
                )

    salvar(fig, nome)


def fig_resultados_agregados(resumo):
    fig_resultados_agregados_par(
        resumo, ["C1", "C2"],
        "01A_resultados_agregados_C1_C2",
        "Indicadores médios — 45,5% de penetração renovável"
    )
    fig_resultados_agregados_par(
        resumo, ["C3", "C4"],
        "01B_resultados_agregados_C3_C4",
        "Indicadores médios — 80% de penetração renovável"
    )


# ============================================================
# FIGURA 2 — C1 x C2: UHR SUBSTITUI TÉRMICA
# Escala dedicada: não mistura carga de 14 GW com UHR de 800 MW.
# ============================================================

def fig_c1_c2_termica_uhr(cen):
    ini, fim = janela_evento(cen["C1"], "geracao_termica_total_MW")
    a, b = recorte(cen["C1"], ini, fim), recorte(cen["C2"], ini, fim)
    h = horas(a)

    fig, ax = plt.subplots(figsize=(11.2, 5.6))
    ax.plot(h, s(a, "geracao_termica_total_MW"),
            marker="o", linewidth=2.2, label="Térmica C1 — sem UHR")
    ax.plot(h, s(b, "geracao_termica_total_MW"),
            marker="o", linewidth=1.8, label="Térmica C2 — com UHR")
    ax.bar(h - .18, s(b, "geracao_uhr_MW"), width=.36,
           alpha=.80, label="Geração UHR — C2")
    ax.bar(h + .18, -s(b, "bombeamento_uhr_MW"), width=.36,
           alpha=.80, label="Bombeamento UHR — C2")

    ax.axhline(0, linewidth=.8)
    ax.set_xlabel("Hora")
    ax.set_ylabel("Potência (MW)")
    ax.set_title("UHR e redução do despacho térmico — C1 × C2")
    ax.grid(axis="y", alpha=.25)
    ax.legend(ncol=2)
    salvar(fig, "02_C1_C2_termica_UHR")
    return ini, fim


# ============================================================
# FIGURA 3 — C3 x C4: BOMBEAMENTO REDUZ CURTAILMENT
# Esta é a figura principal no estilo solicitado.
# ============================================================

def fig_c3_c4_curtailment_uhr(cen):
    ini, fim = janela_evento(cen["C3"], "curtailment_total_MW")
    a, b = recorte(cen["C3"], ini, fim), recorte(cen["C4"], ini, fim)
    h = horas(a)

    fig, ax = plt.subplots(figsize=(11.2, 5.6))
    ax.plot(h, s(a, "curtailment_total_MW"),
            marker="o", linewidth=2.3, label="Curtailment C3 — sem UHR")
    ax.plot(h, s(b, "curtailment_total_MW"),
            marker="o", linewidth=2.0, label="Curtailment C4 — com UHR")
    ax.bar(h - .18, s(b, "geracao_uhr_MW"), width=.36,
           alpha=.80, label="Geração UHR — C4")
    ax.bar(h + .18, -s(b, "bombeamento_uhr_MW"), width=.36,
           alpha=.80, label="Bombeamento UHR — C4")

    ax.axhline(0, linewidth=.8)
    ax.set_xlabel("Hora")
    ax.set_ylabel("Potência (MW)")
    ax.set_title("Bombeamento da UHR e redução do curtailment — C3 × C4")
    ax.grid(axis="y", alpha=.25)
    ax.legend(ncol=2)
    salvar(fig, "03_C3_C4_curtailment_UHR")
    return ini, fim


# ============================================================
# FIGURA 4 — RENOVÁVEIS, CURTAILMENT E UHR (C3/C4)
# Mostra por que há excedente sem achatar a UHR pela escala da carga.
# ============================================================

def fig_renovaveis_c3_c4(cen, janela):
    ini, fim = janela
    a, b = recorte(cen["C3"], ini, fim), recorte(cen["C4"], ini, fim)
    h = horas(a)

    ren_disp = s(a, "eolica_disponivel_MW") + s(a, "solar_disponivel_MW")

    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    ax.plot(h, ren_disp, linewidth=2.0, label="REN disponível")
    ax.plot(h, s(a, "geracao_renovavel_total_MW"),
            linewidth=1.8, label="REN aproveitada — C3")
    ax.plot(h, s(b, "geracao_renovavel_total_MW"),
            linewidth=1.8, label="REN aproveitada — C4")
    ax.fill_between(h, 0, s(a, "curtailment_total_MW"),
                    alpha=.20, label="Curtailment — C3")
    ax.fill_between(h, 0, s(b, "curtailment_total_MW"),
                    alpha=.20, label="Curtailment — C4")

    ax.set_xlabel("Hora")
    ax.set_ylabel("Potência (MW)")
    ax.set_title("Aproveitamento renovável com e sem UHR — alta penetração")
    ax.grid(axis="y", alpha=.25)
    ax.legend(ncol=2)
    salvar(fig, "04_aproveitamento_renovavel_C3_C4")


# ============================================================
# FIGURA 06 — EFEITO DA UHR NA OPERAÇÃO DO SISTEMA
#
# Para tornar visíveis térmica/UHR/curtailment sem deformar as séries
# de carga/REN/hidráulica, cada painel usa:
#   eixo esquerdo  -> grandezas sistêmicas (MW)
#   eixo direito   -> flexibilidade: térmica, UHR e curtailment (MW)
#
# C1/C2 usam uma janela centrada no maior despacho térmico de C1.
# C3/C4 usam uma janela centrada no maior curtailment de C3.
# ============================================================

def _janela_centrada(df, coluna, largura=24):
    y = s(df, coluna)
    h = horas(df)
    centro = int(h[int(np.argmax(y))])
    hmin, hmax = int(np.min(h)), int(np.max(h))
    ini = max(hmin, centro - largura // 2)
    fim = min(hmax, ini + largura - 1)
    ini = max(hmin, fim - largura + 1)
    return ini, fim


def _painel_operacao(ax, df, cenario, janela, resumo, flexmax_compartilhado=None):
    ini, fim = janela
    d = recorte(df, ini, fim)
    h = horas(d)

    # Eixo principal: sistema.
    ax.plot(h, s(d, "carga_MW"), marker="o", markersize=3,
            linewidth=2.0, color="black", label="Carga")
    ax.plot(h, s(d, "eolica_disponivel_MW"), marker="o", markersize=2.5,
            linewidth=1.6, color="red", label="Eólica disponível")
    ax.plot(h, s(d, "solar_disponivel_MW"), marker="o", markersize=2.5,
            linewidth=1.6, color="green", label="Solar disponível")
    ax.plot(h, s(d, "geracao_hidraulica_total_MW"), marker="o", markersize=2.5,
            linewidth=1.7, color="orange", label="Geração hidráulica")

    ax.set_ylabel("Carga / renováveis / hidráulica (MW)", labelpad=8)
    ax.set_xlabel("Hora")
    ax.grid(alpha=.20)

    # Eixo secundário: variáveis menores, ainda em MW.
    ax2 = ax.twinx()
    largura = .28

    term = s(d, "geracao_termica_total_MW")
    guhr = s(d, "geracao_uhr_MW")
    bomb = s(d, "bombeamento_uhr_MW")
    curt = s(d, "curtailment_total_MW")

    
    ax2.plot(h, term, linestyle="--", marker="o", markersize=3,
             linewidth=2.0, label="Geração térmica")
    if np.max(guhr) > 1e-8:
        ax2.bar(h - .18, guhr, width=.36, alpha=.72,
                label="Geração UHR")
    if np.max(bomb) > 1e-8:
        ax2.bar(h + .18, -bomb, width=.36, alpha=.72,
                label="Bombeamento UHR")
    if np.max(curt) > 1e-8:
        ax2.plot(h, curt, linestyle="--", marker="o", markersize=3,
                 linewidth=1.8, color = "brown",label="Curtailment simulado")

    flexmax_local = max(
        float(np.max(term)), float(np.max(guhr)),
        float(np.max(bomb)), float(np.max(curt)), 1.0
    )
    flexmax = flexmax_local if flexmax_compartilhado is None else flexmax_compartilhado
    ax2.set_ylim(-1.20 * flexmax, 1.25 * flexmax)
    ax2.axhline(0, linewidth=.7)
    ax2.set_ylabel("Térmica / UHR / curtailment (MW)", labelpad=10)

    pen = resumo_val(resumo, cenario, "penetracao_REN_disponivel_pct")
    tem_uhr = cenario in ("C2", "C4")
    ax.set_title(
        f"{cenario} — {fmt_pt(pen,1)}% REN "
        f"({'com UHR' if tem_uhr else 'sem UHR'})\n"
        f"Operação do sistema (horas {ini:.0f}–{fim:.0f})",
        fontsize=11, fontweight="bold"
    )

    term_med = resumo_val(resumo, cenario, "geracao_termica_MWmed")
    curt_med = resumo_val(resumo, cenario, "curtailment_total_MWmed")
    deficit = resumo_val(resumo, cenario, "deficit_MWmed")

    linhas = [f"Geração térmica: {fmt_pt(term_med,2)} MWmed"]
    if tem_uhr:
        linhas += [
            f"Geração UHR: {fmt_pt(resumo_val(resumo, cenario, 'geracao_UHR_MWmed'),2)} MWmed",
            f"Bombeamento UHR: {fmt_pt(resumo_val(resumo, cenario, 'bombeamento_UHR_MWmed'),2)} MWmed",
        ]
    linhas += [
        f"Curtailment simulado: {fmt_pt(curt_med,2)} MWmed",
        f"Déficit: {fmt_pt(deficit,2)} MWmed",
    ]

    ax.text(
        .015, .97, "\n".join(linhas),
        transform=ax.transAxes, va="top", ha="left", fontsize=8.5,
        bbox=dict(boxstyle="round,pad=.45", alpha=.80)
    )

    # Junta as duas legendas.
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    return h1 + h2, l1 + l2


def fig_efeito_uhr_sistema(cen, resumo):
    janela_12 = _janela_centrada(cen["C1"], "geracao_termica_total_MW", 24)
    janela_34 = _janela_centrada(cen["C3"], "curtailment_total_MW", 24)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Escala comum do eixo secundário para C3 e C4.
    # Assim, a redução do curtailment com a UHR é visualmente comparável.
    flexmax_34 = 1.0
    for c in ("C3", "C4"):
        d = recorte(cen[c], *janela_34)
        flexmax_34 = max(
            flexmax_34,
            float(np.max(s(d, "geracao_termica_total_MW"))),
            float(np.max(s(d, "geracao_uhr_MW"))),
            float(np.max(s(d, "bombeamento_uhr_MW"))),
            float(np.max(s(d, "curtailment_total_MW")))
        )

    configuracao = [
        ("C1", janela_12, None),
        ("C2", janela_12, None),
        ("C3", janela_34, flexmax_34),
        ("C4", janela_34, flexmax_34),
    ]

    handles, labels = None, None
    for ax, (c, janela, flexmax_comum) in zip(axes.ravel(), configuracao):
        hh, ll = _painel_operacao(
            ax, cen[c], c, janela, resumo,
            flexmax_compartilhado=flexmax_comum
        )
        if c == "C4":
            handles, labels = hh, ll

    # Legenda global sem duplicatas.
    todos_h, todos_l = [], []
    for ax in axes.ravel():
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax.right_ax.get_legend_handles_labels() if hasattr(ax, "right_ax") else ([], [])
        for hh, ll in zip(h1+h2, l1+l2):
            if ll not in todos_l:
                todos_h.append(hh); todos_l.append(ll)

    # Como twinx não é exposto por right_ax em todas as versões, coleta de todos os eixos.
    todos_h, todos_l = [], []
    for a in fig.axes:
        hh, ll = a.get_legend_handles_labels()
        for hhh, lll in zip(hh, ll):
            if lll not in todos_l:
                todos_h.append(hhh); todos_l.append(lll)

    fig.suptitle("Efeito da UHR na operação do sistema",
                 fontsize=18, fontweight="bold", y=.995)
    fig.legend(
        todos_h, todos_l, loc="lower center",
        ncol=4, frameon=False, bbox_to_anchor=(.5, .035)
    )


    fig.subplots_adjust(top=.91, bottom=.10, hspace=.34, wspace=.34)
    fig.savefig(DIR_ARTIGO / "06_efeito_UHR_operacao_sistema.png",
                dpi=DPI_ARTIGO, bbox_inches="tight")
    fig.savefig(DIR_ARTIGO / "06_efeito_UHR_operacao_sistema.pdf",
                bbox_inches="tight")
    fig.savefig(DIR_APRESENTACAO / "06_efeito_UHR_operacao_sistema.png",
                dpi=DPI_APRESENTACAO, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# FIGURA 05 — REDUÇÕES-CHAVE PARA SLIDE
# ============================================================

def fig_reducoes_chave(resumo):
    term_c1 = resumo_val(resumo, "C1", "geracao_termica_MWmed")
    term_c2 = resumo_val(resumo, "C2", "geracao_termica_MWmed")
    curt_c3 = resumo_val(resumo, "C3", "curtailment_total_MWmed")
    curt_c4 = resumo_val(resumo, "C4", "curtailment_total_MWmed")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))

    axes[0].bar(["C1\nsem UHR", "C2\ncom UHR"], [term_c1, term_c2])
    axes[0].set_ylabel("Geração térmica (MWmed)")
    axes[0].set_title("Penetração REN de referência")
    axes[0].grid(axis="y", alpha=.25)

    for i, v in enumerate([term_c1, term_c2]):
        axes[0].text(i, v, fmt_pt(v, 2), ha="center", va="bottom")

    axes[1].bar(["C3\nsem UHR", "C4\ncom UHR"], [curt_c3, curt_c4])
    axes[1].set_ylabel("Curtailment (MWmed)")
    axes[1].set_title("Alta penetração REN")
    axes[1].grid(axis="y", alpha=.25)

    for i, v in enumerate([curt_c3, curt_c4]):
        axes[1].text(i, v, fmt_pt(v, 1), ha="center", va="bottom")

    if term_c1 > 0:
        red_t = 100 * (term_c1 - term_c2) / term_c1
        axes[0].text(.5, .88, f"Redução: {fmt_pt(red_t,1)}%",
                     transform=axes[0].transAxes, ha="center",
                     bbox=dict(boxstyle="round", alpha=.12))

    if curt_c3 > 0:
        red_c = 100 * (curt_c3 - curt_c4) / curt_c3
        axes[1].text(.5, .88, f"Redução: {fmt_pt(red_c,1)}%",
                     transform=axes[1].transAxes, ha="center",
                     bbox=dict(boxstyle="round", alpha=.12))

    fig.suptitle("Principais efeitos da UHR")
    salvar(fig, "05_reducoes_chave")


def main():
    preparar_pastas()
    resumo, cen = carregar()

    # Indicadores auditáveis usados nas figuras.
    cols = [
        "cenario", "FOB_R$", "custo_termico_R$", "custo_curtailment_R$",
        "custo_ciclo_UHR_R$", "custo_vertimento_R$",
        "geracao_hidraulica_MWmed", "geracao_termica_MWmed",
        "curtailment_total_MWmed", "geracao_UHR_MWmed",
        "bombeamento_UHR_MWmed", "vertimento_total_m3s_med"
    ]
    resumo[cols].to_excel(DIR / "indicadores_graficos.xlsx", index=False)

    fig_resultados_agregados(resumo)
    janela_term = fig_c1_c2_termica_uhr(cen)
    janela_curt = fig_c3_c4_curtailment_uhr(cen)
    fig_renovaveis_c3_c4(cen, janela_curt)
    fig_reducoes_chave(resumo)
    fig_efeito_uhr_sistema(cen, resumo)

    print("=" * 70)
    print("GRÁFICOS FINAIS GERADOS")
    print("=" * 70)
    print(f"Resultados: {ARQ_RESULTADOS}")
    print(f"Artigo: {DIR_ARTIGO}")
    print(f"Apresentação: {DIR_APRESENTACAO}")
    print(f"Janela do gráfico 02: {janela_term[0]:.0f}–{janela_term[1]:.0f}")
    print(f"Janela dos gráficos 03/04: {janela_curt[0]:.0f}–{janela_curt[1]:.0f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
