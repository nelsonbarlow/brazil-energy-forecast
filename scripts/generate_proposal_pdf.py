"""Generate 1-page proposal PDF for course submission."""

from fpdf import FPDF

pdf = FPDF()
pdf.add_page()
pdf.set_auto_page_break(auto=False)

# Use DejaVu Sans (Unicode TTF from matplotlib)
import matplotlib
_font_dir = matplotlib.matplotlib_fname().replace("matplotlibrc", "fonts/ttf/")
pdf.add_font("DejaVu", "", _font_dir + "DejaVuSans.ttf")
pdf.add_font("DejaVu", "B", _font_dir + "DejaVuSans-Bold.ttf")

# Title
pdf.set_font("DejaVu", "B", 16)
pdf.cell(0, 12, "Proposta de Projeto", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.ln(2)

pdf.set_font("DejaVu", "B", 13)
pdf.multi_cell(
    0, 7,
    "Brazilian Energy Load Forecasting with Foundation Models",
    align="C",
)
pdf.ln(6)


# Helper
def section(title, body):
    pdf.set_font("DejaVu", "B", 11)
    pdf.cell(0, 7, title, new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("DejaVu", "", 10)
    pdf.multi_cell(0, 5.5, body)
    pdf.ln(3)


# Team
section(
    "Equipe",
    "Nelson Barlow — ngb2@cin.ufpe.br",
)

# Base article
section(
    "Artigo Base",
    "Simeone, L. (2026). Time Series Foundation Models for Energy Load Forecasting "
    "on Consumer Hardware: A Multi-Dimensional Zero-Shot Benchmark.\n"
    "arXiv: https://arxiv.org/abs/2602.10848\n\n"
    "O artigo avalia modelos fundacionais de séries temporais (Chronos-Bolt, Chronos-2, "
    "Moirai-2, TinyTimeMixer) para previsão de carga elétrica em hardware de consumo, "
    "usando dados do ERCOT (Texas) obtidos via EIA Open Data API (43.732 observações "
    "horárias, 2020–2024). Analisa sensibilidade ao comprimento de contexto, calibração "
    "probabilística, robustez a choques de distribuição (COVID-19, Winter Storm Uri) e "
    "aplicações prescritivas (detecção de pico, reserva girante, otimização de "
    "armazenamento).",
)

# Base code
section(
    "Código Base",
    "https://github.com/PhysicsInforMe/tsfm-energy-benchmark\n\n"
    "Repositório oficial do artigo base (Simeone, 2026), contendo a implementação do "
    "benchmark zero-shot de TSFMs em hardware de consumo.",
)

# Data
section(
    "Dados para Validação",
    "Dados do artigo base (referência): ERCOT (grid do Texas, EUA), 43.732 "
    "observações horárias de 2020 a 2024, obtidas via EIA Open Data API.\n\n"
    "Dados desta proposta: ONS (Operador Nacional do Sistema Elétrico do Brasil), "
    "disponíveis em https://dados.ons.org.br/ sob licença CC-BY 4.0.\n"
    "• Série: Curva de Carga Horária (carga média em MW por hora)\n"
    "• Período: 2019–2025 (61.368 registros para o subsistema SE)\n"
    "• Subsistemas: SE (Sudeste), S (Sul), NE (Nordeste), N (Norte)\n"
    "• Resolução: horária, sem necessidade de registro ou API key",
)

# Summary
section(
    "Resumo da Proposta",
    "Este projeto investiga se modelos fundacionais de séries temporais (TSFMs), "
    "pré-treinados em dados globais diversos, conseguem prever a demanda de energia "
    "elétrica no Brasil sem nenhum treinamento em dados locais (zero-shot). "
    "A hipótese central é a 'universalidade da demanda elétrica': os padrões de "
    "consumo são governados por física e comportamento humano, transferindo-se entre "
    "grids sem adaptação.\n\n"
    "Avaliamos quatro TSFMs (Chronos-2, TiRex, Moirai 2.0, TimesFM 2.5) nos quatro "
    "subsistemas do ONS brasileiro, comparando com baselines treinados localmente "
    "(N-BEATS, LSTM) e um baseline naïve. O benchmark inclui métricas pontuais "
    "(MAPE, RMSE) e probabilísticas (CRPS, calibração), análise de erros por "
    "hora/dia/feriado, ablação de contexto e testes de significância estatística. "
    "Simeone (2026) demonstrou resultados promissores no grid ERCOT (Texas), mas "
    "nenhum trabalho avaliou o caso brasileiro — um grid hidro-dependente no "
    "hemisfério sul com calendário de feriados distinto. Auditamos os corpus de "
    "pré-treinamento dos quatro TSFMs — Chronos-2 (arXiv:2510.15821), TiRex "
    "(2505.23719), Moirai 2.0 (2511.11698), TimesFM 2.5 (2310.10688) — e confirmamos "
    "que nenhum contém dados ONS ou carga elétrica brasileira: zero-shot legítimo.",
)

output_path = "proposta_projeto.pdf"
pdf.output(output_path)
print(f"PDF gerado: {output_path}")
