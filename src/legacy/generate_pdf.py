#!/usr/bin/env python3
"""
Generate a publication-quality PDF of the Kazakhstan GDP Nowcasting manuscript.
"""

import os, re, json
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, HRFlowable, KeepTogether
)
from reportlab.lib.colors import HexColor
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

os.chdir('/Users/akzholturdibai/Bayesian research')

# ─── Colours ────────────────────────────────────────────────────────────────
NAVY   = HexColor('#1a3557')
STEEL  = HexColor('#2c5f8a')
ACCENT = HexColor('#d4a017')   # gold line
LIGHT  = HexColor('#f0f4f8')
MID    = HexColor('#4a6a8a')
DARK   = HexColor('#1a1a2e')

# ─── Document setup ─────────────────────────────────────────────────────────
doc = SimpleDocTemplate(
    'manuscript_final.pdf',
    pagesize=A4,
    topMargin=2.5*cm, bottomMargin=2.5*cm,
    leftMargin=2.8*cm, rightMargin=2.8*cm,
    title='Nowcasting GDP in a Data-Constrained Commodity Economy',
    author='Kazakhstan GDP Nowcasting Research',
)

W = A4[0] - 5.6*cm   # usable text width

# ─── Styles ─────────────────────────────────────────────────────────────────
base = getSampleStyleSheet()

def S(name, parent='Normal', **kw):
    s = ParagraphStyle(name, parent=base[parent], **kw)
    return s

styles = {
    'title': S('MyTitle', 'Title',
        fontSize=22, textColor=NAVY, spaceAfter=6,
        leading=28, alignment=TA_LEFT, fontName='Helvetica-Bold'),
    'subtitle': S('MySubtitle',
        fontSize=13, textColor=STEEL, spaceAfter=4,
        leading=18, alignment=TA_LEFT, fontName='Helvetica-Oblique'),
    'author': S('MyAuthor',
        fontSize=11, textColor=MID, spaceAfter=2,
        leading=15, alignment=TA_LEFT, fontName='Helvetica'),
    'date': S('MyDate',
        fontSize=10, textColor=MID, spaceAfter=18,
        leading=14, alignment=TA_LEFT, fontName='Helvetica-Oblique'),
    'abstract_head': S('AbsHead',
        fontSize=11, textColor=NAVY, spaceAfter=4,
        fontName='Helvetica-Bold'),
    'abstract': S('AbsBody',
        fontSize=10, leading=15, textColor=DARK,
        spaceAfter=0, firstLineIndent=0, alignment=TA_JUSTIFY,
        fontName='Helvetica'),
    'h1': S('H1',
        fontSize=14, textColor=NAVY, spaceBefore=18, spaceAfter=6,
        fontName='Helvetica-Bold', keepWithNext=1),
    'h2': S('H2',
        fontSize=12, textColor=STEEL, spaceBefore=14, spaceAfter=4,
        fontName='Helvetica-Bold', keepWithNext=1),
    'h3': S('H3',
        fontSize=11, textColor=MID, spaceBefore=10, spaceAfter=3,
        fontName='Helvetica-BoldOblique', keepWithNext=1),
    'body': S('Body',
        fontSize=10.5, leading=16, textColor=DARK,
        firstLineIndent=14, alignment=TA_JUSTIFY,
        spaceAfter=6, fontName='Helvetica'),
    'body_noi': S('BodyNoIndent',
        fontSize=10.5, leading=16, textColor=DARK,
        firstLineIndent=0, alignment=TA_JUSTIFY,
        spaceAfter=6, fontName='Helvetica'),
    'caption': S('Caption',
        fontSize=9, leading=12, textColor=MID,
        alignment=TA_CENTER, spaceAfter=4,
        fontName='Helvetica-Oblique'),
    'table_hdr': S('TblHdr',
        fontSize=9.5, textColor=colors.white,
        fontName='Helvetica-Bold', alignment=TA_CENTER),
    'table_cell': S('TblCell',
        fontSize=9.5, leading=13, textColor=DARK,
        fontName='Helvetica', alignment=TA_CENTER),
    'table_cell_l': S('TblCellL',
        fontSize=9.5, leading=13, textColor=DARK,
        fontName='Helvetica', alignment=TA_LEFT),
    'ref': S('Ref',
        fontSize=9.5, leading=14, textColor=DARK,
        firstLineIndent=-18, leftIndent=18,
        alignment=TA_JUSTIFY, spaceAfter=4,
        fontName='Helvetica'),
    'eqn': S('Eqn',
        fontSize=10, leading=16, textColor=DARK,
        alignment=TA_CENTER, spaceAfter=6,
        fontName='Helvetica-Oblique'),
    'footnote': S('Footnote',
        fontSize=8.5, leading=12, textColor=MID,
        fontName='Helvetica-Oblique', spaceAfter=2),
    'kw': S('KW',
        fontSize=9.5, leading=14, textColor=DARK,
        fontName='Helvetica', spaceAfter=12),
}

def P(text, style='body'):
    """Shorthand paragraph — strips basic LaTeX markers for display."""
    text = text.replace('\\\\', '<br/>')
    text = re.sub(r'\$\$.*?\$\$', '', text, flags=re.DOTALL)
    text = re.sub(r'\$(.+?)\$', r'<i>\1</i>', text)
    text = re.sub(r'\\textbf\{(.+?)\}', r'<b>\1</b>', text)
    text = re.sub(r'\\textit\{(.+?)\}', r'<i>\1</i>', text)
    text = re.sub(r'\\emph\{(.+?)\}', r'<i>\1</i>', text)
    text = re.sub(r'\\text(superscript|subscript)\{(.+?)\}', r'\2', text)
    text = re.sub(r'\\[a-zA-Z]+\{([^}]*)\}', r'\1', text)
    text = re.sub(r'\\[a-zA-Z]+', '', text)
    return Paragraph(text, styles[style])

def accent_rule():
    return HRFlowable(width=W, thickness=2, color=ACCENT, spaceAfter=4, spaceBefore=2)

def navy_rule():
    return HRFlowable(width=W, thickness=0.5, color=NAVY, spaceAfter=6, spaceBefore=6)

def section_rule():
    return HRFlowable(width=3*cm, thickness=2, color=ACCENT, spaceAfter=8, spaceBefore=2, hAlign='LEFT')

# ─── Accuracy metrics table helper ─────────────────────────────────────────
def metrics_table(rows, col_headers, caption):
    hdr = [Paragraph(h, styles['table_hdr']) for h in col_headers]
    data = [hdr]
    for r in rows:
        row_cells = [Paragraph(str(c), styles['table_cell_l'] if i == 0 else styles['table_cell'])
                     for i, c in enumerate(r)]
        data.append(row_cells)
    col_w = [6*cm, 2.6*cm, 2.6*cm, 2.6*cm]
    tbl = Table(data, colWidths=col_w, repeatRows=1)
    tbl.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), NAVY),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [LIGHT, colors.white]),
        ('GRID', (0,0), (-1,-1), 0.4, colors.white),
        ('LINEBELOW', (0,0), (-1,0), 1.5, ACCENT),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
    ]))
    return [tbl, Spacer(1, 4), P(caption, 'caption'), Spacer(1, 8)]

def lambda_table():
    hdr = [Paragraph(h, styles['table_hdr']) for h in
           ['λ₁', 'B-VAR RMSE', '% Change']]
    rows_data = [
        ['0.10', '0.020501', '—'],
        ['0.15', '0.019939', '−2.7%'],
        ['0.20', '0.019879', '−3.0%'],
    ]
    data = [hdr] + [[Paragraph(c, styles['table_cell']) for c in r] for r in rows_data]
    col_w = [4*cm, 4*cm, 5*cm]
    tbl = Table(data, colWidths=col_w)
    tbl.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), NAVY),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [LIGHT, colors.white]),
        ('GRID', (0,0), (-1,-1), 0.4, colors.white),
        ('LINEBELOW', (0,0), (-1,0), 1.5, ACCENT),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    return [tbl, Spacer(1, 4),
            P('Table 3. Sensitivity of B-VAR Nowcast to Minnesota Prior Tightness.', 'caption'),
            Spacer(1, 8)]

def stei_table():
    hdr = [Paragraph(h, styles['table_hdr']) for h in ['Lag (Months)', 'STEI', 'IPI', 'Freight']]
    rows_data = [
        ['Lag 1', '0.96', '0.64', '−0.41'],
        ['Lag 2', '0.82', '0.56', '−0.42'],
        ['Lag 3', '0.80', '0.53', '−0.44'],
        ['Lag 4', '0.80', '0.59', '−0.46'],
    ]
    data = [hdr] + [[Paragraph(c, styles['table_cell']) for c in r] for r in rows_data]
    col_w = [4*cm, 3*cm, 3*cm, 3.8*cm]
    tbl = Table(data, colWidths=col_w)
    tbl.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), NAVY),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [LIGHT, colors.white]),
        ('GRID', (0,0), (-1,-1), 0.4, colors.white),
        ('LINEBELOW', (0,0), (-1,0), 1.5, ACCENT),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    return [tbl, Spacer(1, 4),
            P('Table 4. Cross-Correlation between High-Frequency Predictors and Quarterly GDP Growth. '
              'Note: Negative freight correlations reflect bulk commodity export composition divergence '
              'from GDP during terms-of-trade adjustments.', 'caption'),
            Spacer(1, 8)]

def appendix_table():
    hdr = [Paragraph(h, styles['table_hdr']) for h in
           ['Month', 'Correction', 'BNS Document', 'Pub. Date', 'Index', 'Growth']]
    rows_data = [
        ['Nov 2017', '+5.0%', 'Kazakhstan_IPI_2016_2020.xlsx', 'Various', '105.0', '+5.0%'],
        ['May 2023', '−1.9%', 'Express Info: Jan–May 2023', 'Jun 15, 2023', '98.1', '−1.9%'],
        ['Jul 2023', '+2.7%', 'Express Info: Jan–Jul 2023', 'Aug 15, 2023', '102.7', '+2.7%'],
        ['Aug 2023*', '+5.4%', 'Express Info: Jan–Aug 2023', 'Sep 15, 2023', '105.4', '+5.4%'],
        ['Nov 2023†', '+0.9%', 'Express Info: Jan–Nov 2023', 'Dec 15, 2023', '100.9', '+0.9%'],
        ['Dec 2025†', '−2.8%', 'Express Info: Jan–Dec 2025', 'Jan 15, 2026', '97.2', '−2.8%'],
    ]
    data = [hdr] + [[Paragraph(c, styles['table_cell_l'] if i == 0 else styles['table_cell'])
                     for i, c in enumerate(r)] for r in rows_data]
    col_w = [2.2*cm, 2.0*cm, 5.0*cm, 2.5*cm, 1.6*cm, 1.7*cm]
    tbl = Table(data, colWidths=col_w)
    tbl.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), NAVY),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [LIGHT, colors.white]),
        ('GRID', (0,0), (-1,-1), 0.4, colors.white),
        ('LINEBELOW', (0,0), (-1,0), 1.5, ACCENT),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('FONTSIZE', (0,0), (-1,-1), 8.5),
    ]))
    return [tbl, Spacer(1, 4),
            P('Table A.1. Deterministic Override Values for IPI Reporting Anomalies. '
              '*August 2023: BNS column header contains typographical error "июню 2022г." '
              'instead of "августу 2022г."; value confirmed via cumulative total. '
              '†Pending final cross-verification against official BNS releases.', 'caption'),
            Spacer(1, 8)]

# ─── Header / Footer ────────────────────────────────────────────────────────
def header_footer(canvas, doc):
    canvas.saveState()
    pn = doc.page
    # Header bar
    canvas.setFillColor(NAVY)
    canvas.rect(2.8*cm, A4[1]-1.8*cm, W, 0.6*cm, fill=1, stroke=0)
    canvas.setFillColor(colors.white)
    canvas.setFont('Helvetica', 8)
    canvas.drawString(2.8*cm + 4, A4[1]-1.45*cm,
                      'Nowcasting GDP in Kazakhstan: Bayesian vs. MIDAS')
    canvas.drawRightString(2.8*cm + W - 4, A4[1]-1.45*cm, 'Working Paper · 2025')
    # Footer
    canvas.setFillColor(MID)
    canvas.setFont('Helvetica', 8)
    canvas.drawCentredString(A4[0]/2, 1.5*cm, f'— {pn} —')
    canvas.setStrokeColor(ACCENT)
    canvas.setLineWidth(1.5)
    canvas.line(2.8*cm, 1.8*cm, 2.8*cm + W, 1.8*cm)
    canvas.restoreState()

# ─── Build story ─────────────────────────────────────────────────────────────
story = []

# ── Title page ──
story += [
    Spacer(1, 1.5*cm),
    accent_rule(),
    Spacer(1, 0.3*cm),
    P('Nowcasting GDP in a Data-Constrained Commodity Economy:', 'title'),
    P('A Bayesian vs. Classical Comparative Framework for Kazakhstan', 'subtitle'),
    Spacer(1, 0.4*cm),
    navy_rule(),
    Spacer(1, 0.3*cm),
    P('Working Paper · 2025', 'date'),
    Spacer(1, 1.2*cm),
    P('<b>Abstract</b>', 'abstract_head'),
]
abstract_text = (
    'This study develops and evaluates a mixed-frequency nowcasting pipeline for Kazakhstan\'s '
    'real GDP growth, comparing a Bayesian Vector Autoregression (B-VAR) with Minnesota Prior '
    'shrinkage against a Mixed Data Sampling (MIDAS) framework with Exponential Almon lag '
    'weighting. Using a 28-quarter recursive pseudo out-of-sample evaluation from Q1 2019 to '
    'Q4 2025, the MIDAS model achieves an RMSE of 0.0078 against the B-VAR\'s 0.0205 — a '
    '61.9 percent reduction — with a Diebold-Mariano p-value of 0.0154. A structural impulse '
    'response analysis identifies a Brent crude oil impact response of 0.47 standard deviations '
    'of quarterly GDP growth, confirming the hydrocarbon transmission channel. An original '
    'empirical finding documents that Kazakhstan\'s Short-Term Economic Indicator (STEI) '
    'concentrates 89 percent of predictive weight on the current month, confirmed by a lag-1 '
    'correlation of r = 0.96 with quarterly GDP — implying STEI functions as a high-frequency '
    'proxy rather than a distributed lag predictor.'
)
story += [
    P(abstract_text, 'abstract'),
    Spacer(1, 0.5*cm),
    P('<b>Keywords:</b> GDP nowcasting, Bayesian VAR, MIDAS, Minnesota prior, Kazakhstan, '
      'commodity economy, impulse response function, Diebold-Mariano test', 'kw'),
    PageBreak(),
]

# ── Section 3: Data and Methodology ──
story += [
    P('3. Data and Methodology', 'h1'), section_rule(),

    P('3.1 Data Sources and Variable Construction', 'h2'),
    P('The dataset comprises macroeconomic, financial, and commodity price indicators for the Republic '
      'of Kazakhstan, spanning Q1 2010 to Q4 2025. The primary data source is the Bureau of National '
      'Statistics of the Agency for Strategic Planning and Reforms of the Republic of Kazakhstan '
      '(stat.gov.kz), which provides the core national accounts, the Short-Term Economic Indicator '
      '(STEI), the Industrial Production Index (IPI), and freight turnover volumes. Monetary and '
      'financial variables, including the USD/KZT exchange rate and aggregate credit to the economy, '
      'were sourced from the National Bank of Kazakhstan (nationalbank.kz). Global commodity price '
      'data, specifically Brent crude oil, copper, zinc, and uranium prices, were obtained from the '
      'World Bank Pink Sheet (World Bank, 2025). All monthly variables were aggregated to quarterly '
      'frequency using the arithmetic mean to align with the quarterly GDP release schedule. The final '
      'sample yields an effective N = 63 quarters. To address historical reporting anomalies in the '
      'Industrial Production Index, specifically in November 2017, May 2023, July 2023, August 2023, '
      'November 2023, and December 2025, as documented in Appendix A, deterministic override values '
      'were applied based on the BNS official revised releases.'),

    P('3.2 Stationarity and Seasonal Adjustment', 'h2'),
    P('Given the relatively short sample size, the presence of structural breaks including the '
      'COVID-19 pandemic, and the symmetry requirements of the MIDAS regression framework, a '
      'first-difference stationary approach was selected over a Vector Error Correction Model (VECM; '
      'Engle & Granger, 1987). The raw real GDP series exhibits strong deterministic seasonality '
      'characteristic of the Kazakhstani economy. Consequently, the X-13ARIMA-SEATS procedure '
      '(Findley et al., 1998; U.S. Census Bureau, 2017) was applied to the natural logarithm of '
      'real GDP. Seasonal adjustment was performed on the log-levels to prevent amplification of '
      'seasonal high-frequency noise during differencing. The transformation sequence is:'),
    P('y_t^SA = X-13-ARIMA-SEATS(ln GDP_t)   ;   Δy_t^SA = y_t^SA − y_{t−1}^SA', 'eqn'),

    P('3.3 Bayesian Vector Autoregression Specification', 'h2'),
    P('The benchmark model is a Bayesian Vector Autoregression (B-VAR; Sims, 1980) specified as:'),
    P('Y_t = Σ_{l=1}^{p} A_l Y_{t−l} + u_t ,  u_t ~ N(0, Σ)', 'eqn'),
    P('The system includes K = 7 variables in Cholesky order: Brent crude oil price, USD/KZT '
      'exchange rate, Industrial Production Index, Short-Term Economic Indicator, freight turnover, '
      'aggregate credit, and real GDP growth. The lag order p = 2 was selected by BIC (Schwarz, 1978). '
      'The Minnesota Prior (Litterman, 1986; Doan et al., 1984) is imposed with hyperparameters '
      'λ₁ = 0.10, λ₂ = 0.50, λ₃ = 1.00. The Cholesky ordering places Brent crude oil first, '
      'reflecting its strict exogeneity to the domestic economy (Cologni & Manera, 2008), and real '
      'GDP last. A ridge regularization guard of 10⁻⁶ was applied to the precision matrix during '
      'full-sample IRF estimation (Kadiyala & Karlsson, 1997).'),

    P('3.4 MIDAS Specification', 'h2'),
    P('The Mixed Data Sampling (MIDAS) framework (Ghysels et al., 2004, 2007) incorporates '
      'high-frequency indicators without pre-aggregation using an Exponential Almon polynomial '
      '(Almon, 1965):'),
    P('Δy_t^SA = α + Σ_k β_k Σ_{j=1}^{m_k} w_j(θ_k) x_{k,3t−j} + γΔy_{t−1}^SA + ε_t', 'eqn'),
    P('Three high-frequency predictors are included: IPI (m = 6 lags), freight (m = 9 lags), and '
      'STEI (m = 6 lags), selected via BIC grid search (Schwarz, 1978). Almon parameters were '
      'estimated using a 49-point grid search to prevent weight degeneracy (Clements & Galvão, 2008). '
      'Estimated values are θ₁ = −0.50, θ₂ = 0.0 for IPI, and θ₁ = −0.25, θ₂ = −0.10 for freight. '
      'For STEI, the grid search converged to a near-degenerate front-loaded profile wherein '
      'approximately 89% of predictive weight is assigned to the most recent monthly observation. '
      'As documented in Section 4.5, this is confirmed as a genuine data property (Giannone et al., '
      '2008) and the estimated distribution is retained.'),
    Spacer(1, 6),
]

# ── Section 4: Empirical Results ──
story += [
    PageBreak(),
    P('4. Empirical Results', 'h1'), section_rule(),

    P('4.1 In-Sample Fit and Model Diagnostics', 'h2'),
    P('The full-sample B-VAR was estimated using p = 2 lags selected by BIC (Schwarz, 1978). '
      'The system satisfies the stationarity condition: the companion matrix maximum eigenvalue '
      'modulus is 0.9302, strictly below unity (Lütkepohl, 2005), confirming the B-VAR is '
      'non-explosive and appropriate for rolling evaluation and IRF analysis.'),

    P('4.2 Pseudo Out-of-Sample Nowcast Evaluation', 'h2'),
    P('The models are evaluated using a recursive pseudo out-of-sample framework (Stock & Watson, 2002). '
      'The initial training window utilises 35 observations from Q1 2010 to Q4 2018, with recursive '
      'nowcasts generated over 28 quarters from Q1 2019 to Q4 2025.'),
    Spacer(1, 4),
]

acc_rows = [
    ['MIDAS Nowcast', '0.0078', '0.0059', '18.3%'],
    ['B-VAR Nowcast', '0.0205', '0.0131', '45.5%'],
    ['AR(1) Nowcast', '0.0193', '0.0111', '36.3%'],
    ['— — —', '— — —', '— — —', '— — —'],
    ['MIDAS Backcast', '0.0063', '0.0047', '14.0%'],
    ['B-VAR Backcast', '0.0177', '0.0114', '40.0%'],
    ['AR(1) Backcast', '0.0176', '0.0102', '33.6%'],
]
story += metrics_table(acc_rows,
    ['Model', 'RMSE', 'MAE', 'MAPE'],
    'Table 1. Recursive Pseudo Out-of-Sample Accuracy Metrics (Q1 2019 – Q4 2025). '
    'Evaluation window: 28 quarters. Training window: Q1 2010 – Q4 2018 (35 observations).')

story += [
    P('MIDAS strictly dominates the B-VAR across all nowcast metrics. MIDAS achieves an RMSE of '
      '0.0078 versus the B-VAR\'s 0.0205, a 61.9% reduction. The MAPE falls from 45.5% (B-VAR) '
      'to 18.3% (MIDAS). A Diebold-Mariano test (Diebold & Mariano, 1995) yields a statistic of '
      '2.423 with p-value 0.0154, confirming MIDAS statistical superiority at the 5% level. '
      'The Harvey et al. (1997) small-sample correction confirms the result holds for T = 28.'),
    P('MIDAS superiority extends to the backcast horizon (RMSE: 0.0063 vs. B-VAR 0.0177), indicating '
      'mixed-frequency aggregation dominates Bayesian shrinkage not only in real-time nowcasting but '
      'also in the revision of the most recently completed quarter.'),

    P('4.3 Structural Analysis via Impulse Response Functions', 'h2'),
    P('Structural IRFs are computed from the full-sample B-VAR using Cholesky decomposition '
      '(Sims, 1980; Blanchard & Quah, 1989; Lütkepohl, 2005). Figure 2 shows the GDP response '
      'to a one-standard-deviation Brent crude shock. The impact response is 0.0068 log points, '
      'equivalent to 0.47 standard deviations of quarterly GDP growth. The 68% confidence band '
      'crosses zero after the seventh quarter, confirming a highly significant and persistent '
      'oil price transmission channel consistent with Kazakhstan\'s hydrocarbon-dependent fiscal '
      'structure (Husain et al., 2008; IMF, 2024).'),
    P('Figure 3 shows the GDP response to a one-standard-deviation IPI shock. The impact response '
      'is 0.0041 log points. The confidence band remains strictly positive for three quarters, '
      'confirming a statistically significant contemporaneous channel from industrial production '
      'to aggregate output.'),

    P('4.4 Robustness of Bayesian Prior Specification', 'h2'),
    P('The B-VAR nowcast is re-evaluated across λ₁ ∈ {0.10, 0.15, 0.20} on the full '
      '28-quarter window (Bańbura et al., 2010):'),
    Spacer(1, 4),
]
story += lambda_table()
story += [
    P('On the extended window, relaxing prior tightness marginally reduces B-VAR RMSE (−3.0% at '
      'λ₁ = 0.20). This is consistent with the COVID-19 episode within the evaluation window, '
      'where tighter shrinkage may over-constrain the model\'s structural response to extreme variance '
      '(Bańbura et al., 2010). Crucially, MIDAS (RMSE 0.0078) strictly dominates all B-VAR '
      'parameterizations, confirming structural robustness of the comparison.'),

    P('4.5 Information Timing and the Within-Quarter Signal Structure of the STEI', 'h2'),
    P('The Exponential Almon weight profile for STEI concentrates 89% of predictive weight on the '
      'most recent monthly observation, with the second lag receiving 11% and all subsequent lags '
      'negligible weight. This is confirmed as a genuine data property, not an optimisation artifact, '
      'by a rolling correlation diagnostic (Ghysels et al., 2007):'),
    Spacer(1, 4),
]
story += stei_table()
story += [
    P('The STEI lag-1 correlation of r = 0.96 with quarterly GDP — substantially higher than IPI '
      '(0.64) — is consistent with the STEI\'s design as a real-time composite tracker of six core '
      'economic sectors (Giannone et al., 2008). Within the MIDAS framework, STEI functions less '
      'as a distributed lag predictor and more as a high-frequency proxy for the quarterly outcome. '
      'The practical implication for the National Bank of Kazakhstan is direct: the first STEI '
      'release within a quarter should be treated as the primary signal, with subsequent within-quarter '
      'releases carrying sharply diminishing informational value.'),
    Spacer(1, 6),
]

# ── Section 5: Limitations ──
story += [
    PageBreak(),
    P('5. Limitations', 'h1'), section_rule(),
    P('<b>Real-time data vintages.</b> The pseudo out-of-sample evaluation uses final revised data, '
      'likely overstating accuracy relative to true real-time performance (Croushore & Stark, 2001). '
      'A vintage database of BNS releases does not currently exist for Kazakhstan; its construction '
      'is a priority for future research.', 'body_noi'),
    P('<b>Evaluation window length.</b> The 28-quarter window remains shorter than the 40–60 quarter '
      'windows typical of advanced-economy nowcasting studies (Harvey et al., 1997). The DM test '
      'result, while significant at 5%, should be interpreted with caution as additional data '
      'accumulates beyond Q4 2025.', 'body_noi'),
    P('<b>MIDAS weight degeneracy.</b> The STEI Almon profile exhibits near-complete concentration '
      'on lag 1, preventing exploitation of the full distributed lag structure motivating MIDAS. '
      'Future work should explore U-MIDAS or Beta-weighted specifications (Foroni et al., 2015; '
      'Ghysels et al., 2007).', 'body_noi'),
    P('<b>Alternative data exclusion.</b> Electricity consumption, satellite nightlight intensity, '
      'and Google Trends mobility data were identified as potentially valuable proxies but were '
      'excluded due to access and harmonisation constraints. Their inclusion is a tractable extension '
      'that could substantially improve nowcast accuracy during structural breaks.', 'body_noi'),
    Spacer(1, 6),
]

# ── Section 6: Conclusion ──
story += [
    P('6. Conclusion', 'h1'), section_rule(),
    P('This study provides a comprehensive comparative analysis of mixed-frequency nowcasting versus '
      'Bayesian shrinkage for Kazakhstan\'s data-constrained commodity economy. Three central findings '
      'emerge.'),
    P('First, MIDAS produces statistically significantly more accurate real-time nowcasts than the '
      'B-VAR, with an RMSE advantage of approximately 61.9% and a Diebold-Mariano p-value of 0.0154 '
      '(Diebold & Mariano, 1995). Both models outperform the naive AR(1) benchmark, confirming '
      'that multi-variable mixed-frequency modelling yields economically meaningful information gains.'),
    P('Second, the Brent crude oil transmission channel produces an impact GDP response of 0.47 '
      'standard deviations per standard deviation shock, confirming a statistically significant and '
      'economically large hydrocarbon dependency channel operating within the first three quarters '
      'following an oil price movement (Husain et al., 2008).'),
    P('Third, the STEI\'s near-instantaneous information decay — evidenced by a lag-1 correlation of '
      'r = 0.96 — implies that real-time nowcasting in Kazakhstan is dominated by the most recent '
      'monthly composite reading. This finding has direct implications for real-time monitoring '
      'system design at the National Bank of Kazakhstan and comparable central banks in '
      'commodity-dependent emerging markets (Giannone et al., 2008).'),
    P('Three specific extensions follow: the construction of a real-time vintage database for BNS '
      'releases; the integration of electricity consumption data as a high-frequency proxy; and '
      'application of the comparative framework to other Central Asian economies, including '
      'Uzbekistan and Azerbaijan, where similar data constraints and commodity dependencies apply.'),
    Spacer(1, 8),
]

# ── References ──
story += [
    PageBreak(),
    P('References', 'h1'), section_rule(),
]

refs = [
    'Almon, S. (1965). The distributed lag between capital appropriations and expenditures. <i>Econometrica</i>, <i>33</i>(1), 178–196. https://doi.org/10.2307/1911894',
    'Bańbura, M., Giannone, D., & Reichlin, L. (2010). Large Bayesian vector auto regressions. <i>Journal of Applied Econometrics</i>, <i>25</i>(1), 71–92. https://doi.org/10.1002/jae.1137',
    'Bell, W. R., & Hillmer, S. C. (1984). Issues involved with the seasonal adjustment of economic time series. <i>Journal of Business & Economic Statistics</i>, <i>2</i>(4), 291–320. https://doi.org/10.1080/07350015.1984.10509398',
    'Blanchard, O. J., & Quah, D. (1989). The dynamic effects of aggregate demand and supply disturbances. <i>American Economic Review</i>, <i>79</i>(4), 655–673.',
    'Bureau of National Statistics of the Agency for Strategic Planning and Reforms of the Republic of Kazakhstan. (2025). <i>Short-term economic indicator, industrial production index, and freight turnover statistics</i> [Data set]. https://stat.gov.kz',
    'Clements, M. P., & Galvão, A. B. (2008). Macroeconomic forecasting with mixed-frequency data. <i>Journal of Business & Economic Statistics</i>, <i>26</i>(4), 546–554. https://doi.org/10.1198/073500108000000015',
    'Cologni, A., & Manera, M. (2008). Oil prices, inflation and interest rates in a structural cointegrated VAR model for the G-7 countries. <i>Energy Economics</i>, <i>30</i>(3), 856–888. https://doi.org/10.1016/j.eneco.2006.11.001',
    'Croushore, D., & Stark, T. (2001). A real-time data set for macroeconomists. <i>Journal of Econometrics</i>, <i>105</i>(1), 111–130. https://doi.org/10.1016/S0304-4076(01)00072-0',
    'Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy. <i>Journal of Business & Economic Statistics</i>, <i>13</i>(3), 253–263. https://doi.org/10.1080/07350015.1995.10524599',
    'Doan, T., Litterman, R., & Sims, C. (1984). Forecasting and conditional projection using realistic prior distributions. <i>Econometric Reviews</i>, <i>3</i>(1), 1–100. https://doi.org/10.1080/07474938408800053',
    'Engle, R. F., & Granger, C. W. J. (1987). Co-integration and error correction: Representation, estimation, and testing. <i>Econometrica</i>, <i>55</i>(2), 251–276. https://doi.org/10.2307/1913236',
    'Findley, D. F., Monsell, B. C., Bell, W. R., Otto, M. C., & Chen, B.-C. (1998). New capabilities and methods of the X-12-ARIMA seasonal-adjustment program. <i>Journal of Business & Economic Statistics</i>, <i>16</i>(2), 127–152. https://doi.org/10.1080/07350015.1998.10524743',
    'Foroni, C., Marcellino, M., & Schumacher, C. (2015). Unrestricted mixed data sampling (MIDAS). <i>Journal of the Royal Statistical Society: Series A</i>, <i>178</i>(1), 57–82. https://doi.org/10.1111/rssa.12043',
    'Ghysels, E., Santa-Clara, P., & Valkanov, R. (2004). <i>The MIDAS touch: Mixed data sampling regression models</i> (CIRANO Working Paper 2004s-20). CIRANO.',
    'Ghysels, E., Sinko, A., & Valkanov, R. (2007). MIDAS regressions: Further results and new directions. <i>Econometric Reviews</i>, <i>26</i>(1), 53–90. https://doi.org/10.1080/07474930600972467',
    'Giannone, D., Reichlin, L., & Small, D. (2008). Nowcasting: The real-time informational content of macroeconomic data. <i>Journal of Monetary Economics</i>, <i>55</i>(4), 665–676. https://doi.org/10.1016/j.jmoneco.2008.05.010',
    'Harvey, D., Leybourne, S., & Newbold, P. (1997). Testing the equality of prediction mean squared errors. <i>International Journal of Forecasting</i>, <i>13</i>(2), 281–291. https://doi.org/10.1016/S0169-2070(96)00719-4',
    'Husain, A. M., Tazhibayeva, K., & Ter-Martirosyan, A. (2008). <i>Fiscal policy and economic cycles in oil-exporting countries</i> (IMF Working Paper WP/08/253). International Monetary Fund.',
    'International Monetary Fund. (2024). <i>Kazakhstan: 2024 Article IV consultation — Staff report</i> (IMF Country Report No. 24/173). International Monetary Fund.',
    'Kadiyala, K. R., & Karlsson, S. (1997). Numerical methods for estimation and inference in Bayesian VAR-models. <i>Journal of Applied Econometrics</i>, <i>12</i>(2), 99–132.',
    'Litterman, R. B. (1986). Forecasting with Bayesian vector autoregressions: Five years of experience. <i>Journal of Business & Economic Statistics</i>, <i>4</i>(1), 25–38. https://doi.org/10.1080/07350015.1986.10509491',
    'Lütkepohl, H. (2005). <i>New introduction to multiple time series analysis</i>. Springer.',
    'National Bank of Kazakhstan. (2025). <i>Exchange rate, monetary indicators, and aggregate credit statistics</i> [Data set]. https://nationalbank.kz',
    'Schwarz, G. (1978). Estimating the dimension of a model. <i>Annals of Statistics</i>, <i>6</i>(2), 461–464.',
    'Sims, C. A. (1980). Macroeconomics and reality. <i>Econometrica</i>, <i>48</i>(1), 1–48. https://doi.org/10.2307/1912017',
    'Stock, J. H., & Watson, M. W. (2002). Forecasting using principal components from a large number of predictors. <i>Journal of the American Statistical Association</i>, <i>97</i>(460), 1167–1179.',
    'U.S. Census Bureau. (2017). <i>X-13ARIMA-SEATS reference manual</i> (Version 1.1). U.S. Department of Commerce. https://www.census.gov/srd/www/x13as/',
    'World Bank. (2025). <i>Commodity markets outlook — Pink Sheet</i> [Data set]. World Bank Group. https://www.worldbank.org/en/research/commodity-markets',
]
for r in refs:
    story.append(Paragraph(r, styles['ref']))
    story.append(Spacer(1, 2))

# ── Appendix A ──
story += [
    PageBreak(),
    P('Appendix A: IPI Data Correction Documentation', 'h1'), section_rule(),
    P('The following table documents the six deterministic override values applied to the '
      'Industrial Production Index series to correct for reporting anomalies identified '
      'against official BNS Express Information publications. All six correction values '
      'are assertion-tested within the data engineering pipeline.'),
    Spacer(1, 8),
]
story += appendix_table()
story += [
    P('* The August 2023 BNS Express Information contains a typographical error in the third column '
      'header reading "июню 2022г." rather than "августу 2022г.". The value of 105.4 is confirmed '
      'as the August year-on-year figure based on consistency with the cumulative January–August '
      'total of 103.8 reported in the same document.', 'footnote'),
    P('† Values for November 2023 and December 2025 reflect BNS override values applied in the '
      'pipeline and are pending final cross-verification against official BNS Express Information '
      'releases for those months.', 'footnote'),
    Spacer(1, 12),
]

# ── Appendix B ──
story += [
    P('Appendix B: Model Specification Details', 'h1'), section_rule(),

    P('B.1 Minnesota Prior — Full Parameterisation', 'h2'),
    P('The posterior mean under the Normal-Wishart conjugate is (Kadiyala & Karlsson, 1997):'),
    P('B̂_post = (X′X + Ω₀⁻¹)⁻¹ (X′Y + Ω₀⁻¹ B₀)', 'eqn'),
    P('A ridge guard δ = 10⁻⁶ is added to the diagonal of the precision matrix when the condition '
      'number exceeds 10¹⁰. This does not materially affect posterior point estimates but ensures '
      'numerical stability during inversion.'),

    P('B.2 MIDAS Exponential Almon Parameters', 'h2'),
    P('The Almon weight function ensures non-negative, unit-summing weights by construction. '
      'Shape is determined by θ₁ (monotone slope) and θ₂ (curvature). Estimation uses a '
      '49-point grid search followed by Nelder-Mead local optimisation with parameter bounds '
      'to prevent degeneracy (Clements & Galvão, 2008).'),
    Spacer(1, 4),
]

almon_data = [
    [Paragraph(h, styles['table_hdr']) for h in ['Predictor', 'Lags (m)', 'θ̂₁', 'θ̂₂', 'Weight on Lag 1']],
    [Paragraph(c, styles['table_cell']) for c in ['IPI', '6', '−0.50', '0.00', '~60%']],
    [Paragraph(c, styles['table_cell']) for c in ['Freight', '9', '−0.25', '−0.10', '~45%']],
    [Paragraph(c, styles['table_cell']) for c in ['STEI', '6', '−0.92*', '0.00', '~89%']],
]
almon_tbl = Table(almon_data, colWidths=[3.5*cm, 2.5*cm, 2.5*cm, 2.5*cm, 3.8*cm])
almon_tbl.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,0), NAVY),
    ('ROWBACKGROUNDS', (0,1), (-1,-1), [LIGHT, colors.white]),
    ('GRID', (0,0), (-1,-1), 0.4, colors.white),
    ('LINEBELOW', (0,0), (-1,0), 1.5, ACCENT),
    ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ('TOPPADDING', (0,0), (-1,-1), 4),
    ('BOTTOMPADDING', (0,0), (-1,-1), 4),
]))
story += [almon_tbl, Spacer(1, 4),
          P('Table B.1. Estimated MIDAS Almon Parameters by Predictor. *STEI convergence '
            'consistent with r = 0.96 lag-1 correlation documented in Section 4.5.', 'caption'),
          Spacer(1, 8)]

story += [
    P('B.3 Diebold-Mariano Test — Small-Sample Correction', 'h2'),
    P('The standard DM statistic (Diebold & Mariano, 1995) can over-reject for small T. '
      'Harvey et al. (1997) propose the correction factor (T+1−2h+h(h−1)/T)/T. For T = 28, '
      'h = 1 (one-step nowcast), the correction multiplier is approximately 1.02, yielding a '
      'corrected DM statistic of 2.47, still significant at 5%. The main results are therefore '
      'robust to small-sample bias in the DM test.'),

    P('B.4 Data Vintage Policy', 'h2'),
    P('All data reflect final revised releases available as of Q4 2025. The Bureau of National '
      'Statistics publishes quarterly GDP in three stages: flash estimate (~30 days after '
      'quarter-end), preliminary estimate (~60 days), and final estimate (~90 days). STEI, IPI, '
      'and freight turnover are published monthly with approximately 15-day lag. All pipeline '
      'assertions verifying the IPI deterministic override values are archived in the data/ '
      'directory of the replication repository.'),
]

# ─── Build ───────────────────────────────────────────────────────────────────
doc.build(story, onFirstPage=header_footer, onLaterPages=header_footer)
print("✓ manuscript_final.pdf generated successfully.")
