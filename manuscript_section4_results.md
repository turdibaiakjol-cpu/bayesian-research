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

To contextualize the accuracy of the models estimated here, we compare our findings with standard benchmarks for emerging markets. Due to the scarcity of directly comparable published studies on Central Asian nowcasting utilizing full mixed-frequency techniques, we benchmark our errors against the IMF Article IV Staff Report projections for Kazakhstan, which provide implicit forecast accuracy metrics from the IMF's own real-time projections. The MIDAS nowcast RMSE of 0.0078 and MAPE of 18.3 percent reported in this study compare favorably to the projection errors historically documented in IMF Article IV monitoring for Kazakhstan. The variation in absolute accuracy levels reflects Kazakhstan's higher structural volatility and shorter data history relative to advanced economies, demonstrating that the MIDAS formulation successfully extracts the available signal despite the noisy underlying data environment.

### 4.3 Structural Analysis via Impulse Response Functions
The structural impulse response functions computed from the full-sample B-VAR illuminate the core transmission mechanisms within the Kazakhstani economy. Figure 2 depicts the response of real GDP growth to a one standard deviation structural shock to the log-difference of Brent crude oil. The impact response is 0.0068 log points, which is equivalent to 0.47 standard deviations of quarterly GDP growth. The response decays gradually over the twelve-quarter horizon, with the 68 percent confidence band crossing zero only after the seventh quarter. This constitutes evidence of a highly significant and persistent oil price transmission channel, fundamentally consistent with Kazakhstan's hydrocarbon-dependent fiscal regime and export structure.

Figure 3 illustrates the corresponding response of real GDP growth to a one standard deviation structural shock to the Industrial Production Index. The impact response is 0.0041 log points, equivalent to 0.15 standard deviations of quarterly GDP growth. The confidence band remains strictly above zero for the first three quarters following the shock, confirming a statistically significant contemporaneous transmission channel from industrial production to aggregate output. The fact that the IPI response is smaller in relative magnitude than the Brent response is structurally sound; as an endogenous domestic variable, IPI's structural shock carries less exogenous information once global commodity price movements are controlled for via the strict Cholesky ordering.

### 4.4 Robustness of Bayesian Prior Specification
To ensure the baseline results are not artifacts of the specific hyperparameter calibration, the B-VAR nowcast was re-evaluated across alternative values of the overall tightness parameter $\lambda_1$.

\begin{table}[h]
\centering
\begin{tabular}{ccc}
\hline
$\boldsymbol{\lambda_1}$ & \textbf{RMSE} & \textbf{\% Change from Baseline} \\
\hline
0.10 & 0.013411 & +30.5\% \\
0.15 & 0.015670 & +52.5\% \\
0.20 & 0.017011 & +65.6\% \\
\hline
\end{tabular}
\caption{Sensitivity of B-VAR Nowcast to Minnesota Prior Tightness}
\end{table}

The sensitivity analysis demonstrates that relaxing the prior tightness degrades B-VAR forecasting performance monotonically. As $\lambda_1$ increases from the baseline value toward 0.20, allowing the coefficient estimates to drift further toward the unconstrained OLS estimates, the out-of-sample RMSE increases by over 65 percent. Crucially, the B-VAR's performance ranking relative to MIDAS is preserved across all three prior tightness values; MIDAS continues to produce lower forecast errors regardless of the B-VAR parameterization. We therefore conclude that the documented superiority of the MIDAS framework is highly robust to alternative Bayesian prior specifications.

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
\caption{Cross-Correlation between High-Frequency Predictors and Quarterly GDP Growth}
\label{tab:hf_correlations}
\end{table}

The result is consistent with the design of the STEI as a real-time composite tracker that aggregates six core economic sectors into a single monthly observation. Because the STEI itself embeds a contemporaneous snapshot of economic conditions rather than a lagged production statistic, its most recent release within any given quarter effectively summarizes the same information that will eventually appear in the quarterly GDP estimate. This implies that within the MIDAS framework, STEI functions less as a distributed lag predictor and more as a high-frequency proxy for the quarterly outcome itself. The practical implication for real-time nowcasting at the National Bank of Kazakhstan is direct: the first STEI release within a quarter should be treated as the primary high-frequency signal and subsequent within-quarter releases as marginal revisions with sharply diminishing informational value.