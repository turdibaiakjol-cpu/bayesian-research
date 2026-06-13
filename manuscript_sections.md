# Methodology and Empirical Results

## 3. Data and Transformation

This study uses quarterly macroeconomic data for Kazakhstan spanning Q1 2010 through Q4 2025, yielding $N = 63$ observations after first-differencing. The dataset comprises 19 variables covering real activity, monetary conditions, commodity prices, exchange rates, and consumer prices.

**Target variable.** The Physical Volume Index of Real GDP is published at quarterly frequency by the Bureau of National Statistics of the Republic of Kazakhstan (BNS, https://stat.gov.kz). Seasonal adjustment is applied using Seasonal-Trend decomposition by LOESS (STL) with a period of 4 quarters applied to the log-level series. The stationary target variable is then computed as:

$$\Delta \ln GDP^{SA}_t = \ln GDP^{SA}_t - \ln GDP^{SA}_{t-1}$$

and labeled `gdp_dlsa` throughout.

**Real activity indicators.** The Industrial Production Index (IPI), Short-Term Economic Indicator (STEI), Freight and Cargo Turnover, Volume of Construction Works, and Wholesale and Retail Trade volumes are all collected at monthly frequency from BNS. Monthly-to-quarterly aggregation follows the convention appropriate to each variable's economic nature: arithmetic mean for indices and rate variables (IPI, STEI, Wages), and summation for flow variables (freight, construction, trade). End-of-quarter values are taken for stock variables (credit, M2, CPI, NBK rate). All level variables are transformed via logarithmic first-differencing:

$$x^{dl}_t = \ln X_t - \ln X_{t-1}$$

The NBK base rate, already expressed in percentage points, is first-differenced in levels rather than logs.

**Commodity prices.** Brent crude oil, copper, and zinc prices are sourced from the World Bank Commodity Markets Pink Sheet (https://www.worldbank.org/en/research/commodity-markets). Uranium spot prices are sourced from the World Nuclear Association and UxC. All are monthly averages aggregated to quarterly by arithmetic mean, then log-differenced.

**Financial variables.** The USD/KZT nominal exchange rate is sourced from the National Bank of Kazakhstan (https://nationalbank.kz). The RUB/KZT rate is computed as a cross-rate. The NBK base rate, credit claims to the economy, and broad money supply M2 are all sourced from NBK statistical publications.

**IPI anomaly correction.** Six monthly observations in the IPI series were identified as containing measurement anomalies based on official BNS revisions and were corrected by direct assignment prior to quarterly aggregation: November 2017 (set to 5.0%), May 2023 (−1.9%), July 2023 (2.7%), August 2023 (5.4%), November 2023 (0.9%), and December 2025 (−2.8%). All corrections were verified by assertion checks.

**Stationarity.** Augmented Dickey-Fuller tests confirm that 13 of 19 transformed variables reject the unit root null at the 5% level. The remaining variables (construction, trade, NBK rate change, credit, M2, wages) are borderline with p-values between 0.05 and 0.53, which is consistent with the well-documented low power of ADF tests in short samples.

---

## 4. B-VAR Specification

### 4.1 The VAR System

The reduced-form VAR($p$) system is specified as:

$$\mathbf{Y}_t = \mathbf{c} + \sum_{\ell=1}^{p} \mathbf{A}_\ell \mathbf{Y}_{t-\ell} + \mathbf{u}_t, \quad \mathbf{u}_t \sim \mathcal{N}(\mathbf{0}, \boldsymbol{\Sigma})$$

where $\mathbf{Y}_t$ is the $K \times 1$ vector of endogenous variables, $\mathbf{A}_\ell$ are $K \times K$ coefficient matrices, and $\boldsymbol{\Sigma}$ is the $K \times K$ covariance matrix of reduced-form disturbances.

The endogenous block comprises $K = 7$ variables: `gdp_dlsa`, `usdkzt_dl`, `ipi_dl`, `freight_dl`, `stei_dl`, `brent_dl`, and `credit_dl`. The first six constitute the core block specified a priori based on economic theory; `usdkzt_dl` was retained to enable the Cholesky identification ordering. Additional candidate variables were screened via Granger causality tests at the 10% significance level; none passed the threshold.

The lag order was determined by estimating a classical VAR and selecting $p$ by the Bayesian Information Criterion, capped at $p_{\max} = 8$. The selected lag order is $p = 2$.

### 4.2 The Minnesota Prior

The Minnesota (Litterman) prior shrinks the coefficient space toward a parsimonious benchmark—the random walk for each endogenous variable. The prior mean is:

$$E[A_\ell^{ij}] = \begin{cases} 1 & \text{if } i = j \text{ and } \ell = 1 \\ 0 & \text{otherwise} \end{cases}$$

The prior variance of the coefficient on variable $j$ at lag $\ell$ in equation $i$ is:

$$V[A_\ell^{ij}] = \begin{cases} \left(\dfrac{\lambda_1}{\ell^{\lambda_3}}\right)^2 & \text{if } i = j \\[10pt] \left(\dfrac{\lambda_1 \cdot \lambda_2}{\ell^{\lambda_3}}\right)^2 \cdot \dfrac{\hat{\sigma}_i^2}{\hat{\sigma}_j^2} & \text{if } i \neq j \end{cases}$$

where $\hat{\sigma}_i^2$ is the residual variance from a univariate AR($p$) estimated for variable $i$, and the hyperparameters are set to $\lambda_1 = 0.1$ (overall tightness), $\lambda_2 = 0.5$ (cross-variable shrinkage), and $\lambda_3 = 1$ (harmonic lag decay).

### 4.3 Posterior Computation

Stacking the system as $\mathbf{Y} = \mathbf{X}\mathbf{B} + \mathbf{U}$, where $\mathbf{X}$ is the $(T-p) \times (1 + Kp)$ matrix of lagged regressors and intercept, the posterior mean is computed equation-by-equation for numerical stability:

$$\hat{\mathbf{B}}_{\text{post}}^{(i)} = \mathbf{V}_{\text{post}}^{(i)} \left[ \frac{\mathbf{X}'\mathbf{Y}^{(i)}}{\sigma_{ii}} + \mathbf{V}_{\text{prior}}^{(i)^{-1}} \mathbf{B}_{\text{prior}}^{(i)} \right]$$

$$\mathbf{V}_{\text{post}}^{(i)} = \left[ \frac{\mathbf{X}'\mathbf{X}}{\sigma_{ii}} + \mathbf{V}_{\text{prior}}^{(i)^{-1}} \right]^{-1}$$

For uncertainty quantification, 1,000 draws from the posterior distribution $\mathbf{B}^{(i)} \sim \mathcal{N}(\hat{\mathbf{B}}_{\text{post}}^{(i)}, \mathbf{V}_{\text{post}}^{(i)})$ are propagated forward to construct 68% and 90% credible intervals.

### 4.4 Impulse Response Functions

Structural identification employs Cholesky decomposition of $\boldsymbol{\Sigma}$, with variables ordered from most exogenous to most endogenous: `brent_dl` → `usdkzt_dl` → `ipi_dl` → `freight_dl` → `credit_dl` → `gdp_dlsa`. IRFs are computed for a 12-quarter horizon with 68% confidence bands from 500 bootstrap replications.

---

## 5. MIDAS Specification

### 5.1 The Exponential Almon MIDAS Equation

The MIDAS regression with autoregressive augmentation is specified as:

$$\Delta \ln GDP^{SA}_t = \alpha + \rho \cdot \Delta \ln GDP^{SA}_{t-1} + \sum_{k=1}^{K^{HF}} \beta_k \sum_{j=1}^{m_k} w_j(\boldsymbol{\theta}_k) \cdot x_{k,3t-j} + \varepsilon_t$$

where $x_{k,3t-j}$ denotes the $j$-th monthly lag of high-frequency predictor $k$ within the quarter, and $w_j(\boldsymbol{\theta}_k)$ is the Exponential Almon weighting function:

$$w_j(\boldsymbol{\theta}) = \frac{\exp(\theta_1 j + \theta_2 j^2)}{\sum_{s=1}^{m} \exp(\theta_1 s + \theta_2 s^2)}$$

The three high-frequency predictors are: IPI ($m = 6$), Freight Turnover ($m = 9$), and STEI ($m = 6$), with the optimal number of monthly lags $m_k$ selected by BIC grid search over $\{6, 9, 12\}$.

### 5.2 Estimation

The nonlinear parameters $\theta_1, \theta_2$ and the regression coefficients $\alpha, \rho, \beta_k$ are estimated jointly by nonlinear least squares using the L-BFGS-B algorithm via `scipy.optimize.minimize`. Starting values are $\theta_1 = -0.5$ and $\theta_2 = -0.1$. The dominant predictor is the STEI, with $|\beta| = 0.0075$.

---

## 6. Empirical Results

### 6.1 Rolling Pseudo Out-of-Sample Evaluation

The models are evaluated using a rolling pseudo out-of-sample exercise over Q1 2021 – Q4 2025 (20 quarters). The initial training window spans Q1 2010 – Q4 2020 (approximately 44 quarters). At each evaluation quarter $t$, both models are re-estimated using only data available up to $t - 1$, and one-step-ahead nowcasts and backcasts are produced.

**Table 1: Forecast Accuracy Metrics (Q1 2021 – Q4 2025)**

| Model | RMSE | MAE | MAPE (%) |
|-------|------|-----|----------|
| B-VAR Nowcast | 0.0190 | 0.0111 | 29.29 |
| MIDAS Nowcast | 0.0074 | 0.0063 | 17.44 |
| B-VAR Backcast | 0.0169 | 0.0100 | 26.35 |
| MIDAS Backcast | 0.0081 | 0.0064 | 18.10 |

The MIDAS model substantially outperforms the B-VAR across all three accuracy metrics and both forecast horizons. The RMSE advantage of MIDAS over B-VAR is approximately 61% for nowcasts (0.0074 vs. 0.0190) and 52% for backcasts (0.0081 vs. 0.0169). This result aligns with the theoretical expectation that the parsimonious Exponential Almon weighting scheme—which compresses monthly information into a small number of estimated parameters—is better suited to the data-constrained environment of 63 quarterly observations than the richly parameterized B-VAR.

### 6.2 Diebold-Mariano Test

The Diebold-Mariano test evaluates whether the difference in predictive accuracy between the two models is statistically significant. Using the B-VAR as the benchmark:

| Horizon | DM Statistic | p-value |
|---------|-------------|---------|
| Nowcast | 1.224 | 0.221 |
| Backcast | 1.187 | 0.235 |

The DM test does not reject the null of equal predictive accuracy at conventional significance levels ($p > 0.10$), despite the economically meaningful RMSE gap. This result is consistent with the well-documented low power of the DM test in short evaluation samples ($n = 20$). The inability to reject statistical equality should not be interpreted as evidence that the models are equally accurate in economic terms; rather, it reflects the limited discriminatory power available with only 20 evaluation quarters.

### 6.3 Impulse Response Analysis

Figure 2 presents the response of quarterly GDP growth to a one standard deviation shock to Brent crude oil prices. The response is positive on impact, consistent with the dominant role of hydrocarbon revenues in Kazakhstan's fiscal and external accounts, and dissipates within approximately 4–6 quarters. The 68% confidence band includes zero from the third quarter onward, indicating that the transmission mechanism operates primarily through contemporaneous and near-term channels.

Figure 3 shows the response of GDP growth to an IPI shock. The impact effect is positive and persistent, consistent with the industrial sector's direct weight in the GDP production decomposition. The response remains statistically distinguishable from zero for 3–4 quarters.

### 6.4 MIDAS Weight Profile

Figure 4 displays the Exponential Almon weight profile for the STEI predictor. The weighting pattern shows a rapid decay, with the most recent monthly observations within the quarter receiving the highest weight. This pattern is economically intuitive: the most recent high-frequency data provides the strongest signal for the current quarter's GDP outturn, while more distant monthly observations contribute diminishing marginal information.

### 6.5 Robustness: Minnesota Prior Sensitivity

**Table 2: B-VAR Nowcast RMSE Sensitivity to λ₁**

| λ₁ | RMSE | Δ vs. Baseline (%) |
|----|------|---------------------|
| 0.10 | 0.0190 | 0.0 |
| 0.15 | 0.0191 | +0.6 |
| 0.20 | 0.0193 | +1.5 |

The B-VAR's nowcast performance is structurally robust across the prior tightness grid. Increasing $\lambda_1$ from 0.10 to 0.20—a doubling of the overall prior tightness parameter—produces only a 1.5% increase in RMSE. This insensitivity confirms that the B-VAR's relative performance disadvantage vis-à-vis the MIDAS model is not an artifact of a specific prior calibration but reflects a structural limitation: in a small-sample, data-constrained environment, the high-dimensional coefficient space of the VAR system cannot be estimated with sufficient precision, even with informative shrinkage priors, to match the forecasting efficiency of the parsimonious MIDAS specification.
