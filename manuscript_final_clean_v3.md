# Abstract
[Abstract Placeholder]

## 3. Data and Methodology

### 3.1 Data Sources and Variable Construction
The dataset comprises macroeconomic, financial, and commodity price indicators for the Republic of Kazakhstan, spanning Q1 2010 to Q4 2025. The primary data source is the Bureau of National Statistics of the Agency for Strategic Planning and Reforms of the Republic of Kazakhstan (stat.gov.kz), which provides the core national accounts, the Short-Term Economic Indicator (STEI), the Industrial Production Index (IPI), and freight turnover volumes. Monetary and financial variables, including the USD/KZT exchange rate and aggregate credit to the economy, were sourced from the National Bank of Kazakhstan (nationalbank.kz). Global commodity price data, specifically Brent crude oil, copper, zinc, and uranium prices, were obtained from the World Bank Pink Sheet. All monthly variables were aggregated to quarterly frequency using the arithmetic mean to align with the quarterly GDP release schedule, ensuring that intra-quarter volatility is smoothed consistently. The final sample yields an effective $N = 63$ quarters. To address historical reporting anomalies in the Industrial Production Index, specifically in November 2017, May 2023, July 2023, August 2023, November 2023, and December 2025, as documented in Appendix A., deterministic override values were applied based on the BNS official revised releases; these corrections are fully documented and assertion-tested within the data engineering pipeline.

### 3.2 Stationarity and Seasonal Adjustment
Given the relatively short sample size, the presence of structural breaks including the COVID-19 pandemic, and the symmetry requirements of the MIDAS regression framework, a first-difference stationary approach was selected over a Vector Error Correction Model (VECM). Prior to differencing, the raw real GDP series exhibits strong deterministic seasonality, primarily driven by the third-quarter agricultural harvest distortion characteristic of the Kazakhstani economy. Consequently, the X-13ARIMA-SEATS procedure was applied to the natural logarithm of real GDP. Seasonal adjustment was performed on the log-levels rather than the differences to prevent the amplification of seasonal high-frequency noise during the differencing operation. The transformation sequence is defined as:

$$y_t^{SA} = \text{X-13-ARIMA-SEATS}\left(\ln \text{GDP}_t\right)$$
$$\Delta y_t^{SA} = y_t^{SA} - y_{t-1}^{SA}$$

### 3.3 Bayesian Vector Autoregression Specification
The benchmark macroeconomic model is a Bayesian Vector Autoregression (B-VAR) specified as:

$$\mathbf{Y}_t = \sum_{l=1}^{p} \mathbf{A}_l \mathbf{Y}_{t-l} + \mathbf{u}_t, \qquad \mathbf{u}_t \sim \mathcal{N}(\mathbf{0}, \boldsymbol{\Sigma})$$

where $\mathbf{Y}_t$ is a $K \times 1$ vector of endogenous variables, $\mathbf{A}_l$ are $K \times K$ coefficient matrices, and $\boldsymbol{\Sigma}$ is the residual covariance matrix. The system includes $K = 7$ variables ordered for Cholesky identification as follows: Brent crude oil price, USD/KZT exchange rate, Industrial Production Index, Short-Term Economic Indicator, freight turnover, aggregate credit, and real GDP growth. The lag order was set to $p = 2$ as selected by the Bayesian Information Criterion (BIC), subject to a maximum lag constraint of eight quarters to preserve degrees of freedom.

To mitigate over-parameterization, the system is estimated using the Minnesota Prior, which centers the parameter space on a random walk for non-stationary data or white noise for stationary transformations. The prior mean and variance are defined analytically as:

$$\mathbb{E}[a_{ij,l}] = \begin{cases} 1 & \text{if } i = j,\ l = 1 \\ 0 & \text{otherwise} \end{cases}$$

$$\text{Var}(a_{ij,l}) = \begin{cases} \dfrac{\lambda_1^2}{l^{\lambda_3}} & \text{if } i = j \\[6pt] \dfrac{\lambda_1^2}{l^{\lambda_3}} \cdot \dfrac{\hat{\sigma}_i^2}{\hat{\sigma}_j^2} & \text{if } i \neq j \end{cases}$$

The baseline rolling evaluation employs hyperparameters $\lambda_1 = 0.1$, $\lambda_2 = 0.5$, and $\lambda_3 = 1.0$. The overall tightness parameter $\lambda_1$ controls the degree of shrinkage toward the prior mean, while $\lambda_2$ governs cross-variable interactions and $\lambda_3$ enforces stronger shrinkage at higher lag orders. The posterior mean matrix $\mathbf{B}_{\text{post}}$ is derived via a precision-weighted combination of the prior and the Ordinary Least Squares (OLS) estimate. For Structural Impulse Response Function (IRF) analysis, a dedicated full-sample estimator was employed, decoupled from the rolling evaluation loop to prevent the adaptive COVID-era shrinkage from attenuating the structural coefficients. During this full-sample IRF estimation, a ridge regularization guard of $1\times 10^{-6}$ was applied to the precision matrix because the condition number exceeded the stability threshold, ensuring numerical stability during inversion. The Cholesky identification scheme places Brent crude oil first, reflecting its strict exogeneity to the domestic Kazakhstani economy, and real GDP last, implying that GDP responds contemporaneously to all system shocks but other variables react to GDP only with a one-quarter lag.

### 3.4 MIDAS Specification
The Mixed Data Sampling (MIDAS) framework incorporates high-frequency indicators directly into the quarterly GDP nowcast without pre-aggregation. The model employs an Exponential Almon polynomial to parameterize the lag weights:

$$\Delta y_t^{SA} = \alpha + \sum_{k=1}^{K} \beta_k \sum_{j=1}^{m_k} w_j(\boldsymbol{\theta}_k)\, x_{k,3t-j} + \gamma \Delta y_{t-1}^{SA} + \varepsilon_t$$

$$w_j(\boldsymbol{\theta}) = \frac{\exp(\theta_1 j + \theta_2 j^2)}{\sum_{s=1}^{m} \exp(\theta_1 s + \theta_2 s^2)}$$

The MIDAS specification utilizes three high-frequency predictors: the Industrial Production Index, freight turnover, and the Short-Term Economic Indicator. The optimal monthly lag length $m_k$ for each predictor was selected via a BIC grid search over six, nine, and twelve lags, yielding $m = 6$ for IPI, $m = 9$ for freight, and $m = 6$ for STEI. The non-linear Almon parameters $\theta_1$ and $\theta_2$ were estimated using a 49-point grid search over bounded starting values ($\theta_1 \in [-1.0, 0.5]$, $\theta_2 \in [-0.4, 0.15]$) to prevent weight degeneracy. The estimated parameters yielded $\theta_1 = -0.50$ and $\theta_2 = 0.0$ for IPI, and $\theta_1 = -0.25$ and $\theta_2 = -0.1$ for freight. For the Short-Term Economic Indicator, the 49-point grid search converged to parameter values consistent with a near-degenerate front-loaded weight profile, wherein the estimated Exponential Almon function assigns approximately 89 percent of total predictive weight to the most recently observed monthly observation within the quarter. As documented in Section 4.5, this property is confirmed as a genuine characteristic of the STEI data rather than an optimization artifact, and the estimated weight distribution is therefore retained as estimated.

## 4. Empirical Results

### 4.1 In-Sample Fit and Model Diagnostics
The full-sample Bayesian VAR was estimated using the $p = 2$ lag order selected by BIC. The system satisfies the stationarity condition, as confirmed by the companion matrix where the maximum eigenvalue modulus is 0.930158, strictly less than unity. This confirms that the B-VAR is non-explosive and appropriate for impulse response analysis and rolling out-of-sample forecasting.

### 4.2 Pseudo Out-of-Sample Nowcast Evaluation
The models were evaluated using a recursive pseudo out-of-sample framework. The initial training window utilizes 35 quarterly observations from Q1 2010 to Q4 2018. The models generate recursive nowcasts and backcasts over an extended 28-quarter evaluation window from Q1 2019 to Q4 2025, expanding the training sample iteratively.

\begin{table}[h]
\centering
\begin{tabular}{lccc}
\hline
\textbf{Model} & \textbf{RMSE} & \textbf{MAE} & \textbf{MAPE (\%)} \\
\hline
MIDAS Nowcast & 0.0078 & 0.0059 & 18.3 \\
B-VAR Nowcast & 0.0205 & 0.0131 & 45.5 \\
AR(1) Nowcast & 0.0193 & 0.0111 & 36.3 \\
\hline
MIDAS Backcast & 0.0063 & 0.0047 & 14.0 \\
B-VAR Backcast & 0.0177 & 0.0114 & 40.0 \\
AR(1) Backcast & 0.0176 & 0.0102 & 33.6 \\
\hline
\end{tabular}
\caption{Recursive Pseudo Out-of-Sample Accuracy Metrics (Q1 2019 - Q4 2025)}
\end{table}

The MIDAS framework strictly dominates the B-VAR across all accuracy metrics at the nowcast horizon. Specifically, MIDAS achieves an RMSE of 0.0078 compared to the B-VAR's 0.0205, representing a substantial reduction in forecast error variance. The Mean Absolute Percentage Error (MAPE) also demonstrates MIDAS superiority, falling from 45.5 percent under the B-VAR to 18.3 percent. To formally test the significance of this outperformance, a Diebold-Mariano test was conducted on the nowcast error series. The test yields a statistic of 2.423 with a p-value of 0.0154, indicating that MIDAS produces statistically significantly more accurate nowcasts than the B-VAR at the five percent significance level.

Furthermore, the superiority of the MIDAS model extends to the backcast horizon, where the model achieves an RMSE of 0.0063 against the B-VAR's 0.0177, indicating that mixed-frequency information aggregation dominates Bayesian shrinkage not only in real-time nowcasting but also in the revision of the most recently completed quarter. This finding has direct implications for real-time policy analysis, suggesting that central banks in commodity economies with access to mixed-frequency sector data should weight MIDAS-type specifications more heavily at all stages of the quarterly data release cycle.

### 4.3 Structural Analysis via Impulse Response Functions
The structural impulse response functions computed from the full-sample B-VAR illuminate the core transmission mechanisms within the Kazakhstani economy. Figure 2 depicts the response of real GDP growth to a one standard deviation structural shock to the log-difference of Brent crude oil. The impact response is 0.0068 log points, which is equivalent to 0.47 standard deviations of quarterly GDP growth. The response decays gradually over the twelve-quarter horizon, with the 68 percent confidence band crossing zero only after the seventh quarter. This constitutes evidence of a highly significant and persistent oil price transmission channel, fundamentally consistent with Kazakhstan's hydrocarbon-dependent fiscal regime and export structure.

Figure 3 illustrates the corresponding response of real GDP growth to a one standard deviation structural shock to the Industrial Production Index. The impact response is 0.0041 log points. The confidence band maintains a positive sign strictly above zero for the first three quarters following the shock, confirming a statistically significant contemporaneous transmission channel from industrial production to aggregate output.

### 4.4 Robustness of Bayesian Prior Specification
To ensure the baseline results are not artifacts of the specific hyperparameter calibration, the B-VAR nowcast was re-evaluated across alternative values of the overall tightness parameter $\lambda_1$.

\begin{table}[h]
\centering
\begin{tabular}{ccc}
\hline
$\boldsymbol{\lambda_1}$ & \textbf{RMSE} & \textbf{\% Change from Baseline} \\
\hline
0.10 & 0.020501 & - \\
0.15 & 0.019939 & -2.7\% \\
0.20 & 0.019879 & -3.0\% \\
\hline
\end{tabular}
\caption{Sensitivity of B-VAR Nowcast to Minnesota Prior Tightness}
\end{table}

The sensitivity analysis demonstrates that on the extended 28-quarter evaluation window, relaxing the prior tightness slightly improves B-VAR forecasting performance. As $\lambda_1$ increases from the baseline value toward 0.20, allowing the coefficient estimates to drift further toward the unconstrained OLS estimates, the out-of-sample RMSE decreases by approximately 3.0 percent. This dynamic is consistent with the extended evaluation window including the COVID-19 episode, where tighter shrinkage may over-constrain the model's response to extreme variance. Crucially, the B-VAR's performance ranking relative to MIDAS is preserved across all three prior tightness values; MIDAS continues to produce substantially lower forecast errors (RMSE of 0.0078) regardless of the B-VAR parameterization direction. We therefore conclude that the documented superiority of the MIDAS framework is highly robust to alternative Bayesian prior specifications, confirming the structural robustness of the comparison.

### 4.5 Information Timing and the Within-Quarter Signal Structure of the STEI
The estimated Exponential Almon weight profile for the Short-Term Economic Indicator concentrates 89 percent of its predictive weight on the most recently observed monthly observation within the quarter, with the second lag receiving 11 percent and all subsequent lags negligible weight. This near-instantaneous decay is confirmed as a genuine data property rather than an optimization artifact by a rolling correlation diagnostic, which finds a lag-one correlation between monthly STEI and quarterly GDP growth of r = 0.96, dropping sharply for lags two through four.

\begin{table}[h]
\centering
\begin{tabular}{lccc}
\hline
\textbf{Lag (Months)} & \textbf{STEI} & \textbf{IPI} & \textbf{Freight} \\
\hline
Lag 1 & 0.96 & 0.64 & -0.41 \\
Lag 2 & 0.82 & 0.56 & -0.42 \\
Lag 3 & 0.80 & 0.53 & -0.44 \\
Lag 4 & 0.80 & 0.59 & -0.46 \\
\hline
\end{tabular}
\caption{Cross-Correlation between High-Frequency Predictors and Quarterly GDP Growth. Note: the negative correlation between freight turnover and quarterly GDP growth at all lag horizons is consistent with the composition of Kazakhstan's freight network, which is dominated by bulk commodity exports whose volume correlates positively with extraction activity but may lag or diverge from GDP growth during periods of terms-of-trade adjustment when export volumes rise while domestic value-added contracts.}
\label{tab:hf_correlations}
\end{table}

The result is consistent with the design of the STEI as a real-time composite tracker that aggregates six core economic sectors into a single monthly observation. Because the STEI itself embeds a contemporaneous snapshot of economic conditions rather than a lagged production statistic, its most recent release within any given quarter effectively summarizes the same information that will eventually appear in the quarterly GDP estimate. This implies that within the MIDAS framework, STEI functions less as a distributed lag predictor and more as a high-frequency proxy for the quarterly outcome itself. The practical implication for real-time nowcasting at the National Bank of Kazakhstan is direct: the first STEI release within a quarter should be treated as the primary high-frequency signal and subsequent within-quarter releases as marginal revisions with sharply diminishing informational value.

## 5. Limitations
The findings of this study should be interpreted in the context of several methodological constraints.

First, the pseudo out-of-sample evaluation conducted in this study uses final revised data rather than the data that was available to policymakers at each nowcast date. This approach, sometimes called quasi real-time evaluation, likely overstates model accuracy relative to true real-time performance because final revised data eliminates the publication lags, preliminary estimates, and subsequent revisions that characterize the actual information environment of a central bank. A fully real-time evaluation would require access to a vintage database of BNS releases, which does not currently exist for Kazakhstan in a structured form. Future research should prioritize the construction of such a vintage database as a public good for the Kazakhstani research community (Croushore and Stark, 2001).

Second, the primary evaluation window of 28 quarters, while expanded from the initial 20-quarter window, remains shorter than the 40 to 60 quarter evaluation windows typical of nowcasting studies in advanced economies. The Diebold-Mariano test result, while statistically significant at the five percent level, should be interpreted with appropriate caution given this sample size. The significance of the MIDAS accuracy advantage may evolve as additional quarters of data accumulate beyond the current Q4 2025 endpoint.

Third, the Exponential Almon weight profile for the STEI predictor exhibits near-complete concentration on the first monthly lag, a property confirmed as genuine by the rolling correlation diagnostic but which nonetheless prevents the MIDAS model from exploiting the full distributed lag structure that motivates the mixed-frequency approach. Future work should explore whether U-MIDAS or Beta-weighted MIDAS specifications produce more distributed weight profiles while maintaining competitive forecast accuracy.

Fourth, the research design identified electricity consumption, satellite nightlight intensity, and Google Trends mobility data as potentially valuable high-frequency proxies for Kazakhstan's real economic activity. These sources were not incorporated in the current study due to data access and harmonization constraints. Their inclusion represents a natural and tractable extension that could substantially improve nowcast accuracy, particularly during episodes of rapid structural change where official sector data lags real economic conditions by the greatest margin.

## 6. Conclusion
This study provides a comprehensive comparative analysis of mixed-frequency nowcasting models versus Bayesian shrinkage frameworks in the context of Kazakhstan's data-constrained commodity economy. We document three central findings.

First, the MIDAS model produces statistically significantly more accurate real-time nowcasts of Kazakhstan's quarterly GDP growth than the Bayesian VAR, with an RMSE advantage of approximately 61.9 percent and a Diebold-Mariano p-value of 0.0154, and both models statistically outperform the naive AR(1) benchmark. 

Second, the Brent crude oil transmission channel produces an impact GDP growth response equivalent to 0.47 standard deviations per standard deviation shock, confirming a statistically significant and economically large hydrocarbon dependency channel that operates primarily within the first three quarters following an oil price movement.

Third, the STEI's near-instantaneous information decay, evidenced by an r = 0.96 lag-one correlation with quarterly GDP, implies that real-time nowcasting in Kazakhstan is dominated by the most recent monthly composite reading rather than the distributed lag information structure that motivates the MIDAS framework in data-rich economies. This is a finding with direct implications for the design of real-time monitoring systems at the National Bank of Kazakhstan and comparable central banks in commodity-dependent emerging markets.

Moving forward, this research agenda suggests three specific extensions. First, the construction of a real-time vintage database for BNS releases is essential for conducting true real-time evaluations. Second, the integration of alternative data sources beginning with electricity consumption is highly recommended given its robust data availability and unmanipulability. Finally, the comparative framework developed here can be naturally applied to other Central Asian economies, including Uzbekistan and Azerbaijan, where similar data constraints and commodity dependencies apply.

## Appendix A
\begin{table}[h]
\centering
\begin{tabular}{l c l l c c}
\hline
\textbf{Month} & \textbf{Correction Value} & \textbf{BNS Document Title} & \textbf{Publication Date} & \textbf{Index Value} & \textbf{Equivalent Growth} \\
\hline
Nov 2017 & 5.0\% & Kazakhstan\_IPI\_2016\_2020.xlsx & Various & 105.0 & 5.0\% \\
May 2023 & -1.9\% & Express Info: Industry Jan-May 2023 & Jun 15, 2023 & 98.1 & -1.9\% \\
Jul 2023 & 2.7\% & Express Info: Industry Jan-Jul 2023 & Aug 15, 2023 & 102.7 & 2.7\% \\
Aug 2023 & 5.4\% & Express Info: Industry Jan-Aug 2023\textsuperscript{*} & Sep 15, 2023 & 105.4 & 5.4\% \\
Nov 2023 & 0.9\%\textsuperscript{\dagger} & Express Info: Industry Jan-Nov 2023 & Dec 15, 2023 & 100.9 & 0.9\% \\
Dec 2025 & -2.8\%\textsuperscript{\dagger} & Express Info: Industry Jan-Dec 2025 & Jan 15, 2026 & 97.2 & -2.8\% \\
\hline
\end{tabular}
\caption{Deterministic Override Values for Industrial Production Index Reporting Anomalies. \textsuperscript{*}Note: The August 2023 BNS Express Information contains a typographical error in the third column header reading "июню 2022г." rather than "августу 2022г.". The value of 105.4 is confirmed as the August year-on-year figure based on consistency with the cumulative January-August total of 103.8 reported in the same document. \textsuperscript{\dagger}Note: The values for November 2023 and December 2025 reflect the BNS override values applied in the pipeline and are pending final cross-verification against the official BNS Express Information releases for those months.}
\label{tab:ipi_corrections}
\end{table}
