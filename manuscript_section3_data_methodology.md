## 3. Data and Methodology

### 3.1 Data Sources and Variable Construction
The dataset comprises macroeconomic, financial, and commodity price indicators for the Republic of Kazakhstan, spanning Q1 2010 to Q4 2025. The primary data source is the Bureau of National Statistics of the Agency for Strategic Planning and Reforms of the Republic of Kazakhstan (stat.gov.kz), which provides the core national accounts, the Short-Term Economic Indicator (STEI), the Industrial Production Index (IPI), and freight turnover volumes. Monetary and financial variables, including the USD/KZT exchange rate and aggregate credit to the economy, were sourced from the National Bank of Kazakhstan (nationalbank.kz). Global commodity price data, specifically Brent crude oil, copper, zinc, and uranium prices, were obtained from the World Bank Pink Sheet. All monthly variables were aggregated to quarterly frequency using the arithmetic mean to align with the quarterly GDP release schedule, ensuring that intra-quarter volatility is smoothed consistently. The final sample yields an effective $N = 63$ quarters. To address historical reporting anomalies in the Industrial Production Index, specifically the discontinuities in 2015, deterministic override values were applied based on the BNS official revised releases; these corrections are fully documented and assertion-tested within the data engineering pipeline.

### 3.2 Stationarity and Seasonal Adjustment
Given the relatively short sample size, the presence of structural breaks including the COVID-19 pandemic, and the symmetry requirements of the MIDAS regression framework, a first-difference stationary approach was selected over a Vector Error Correction Model (VECM). Prior to differencing, the raw real GDP series exhibits strong deterministic seasonality, primarily driven by the third-quarter agricultural harvest distortion characteristic of the Kazakhstani economy. Consequently, the X-13ARIMA-SEATS procedure was applied to the natural logarithm of real GDP. Seasonal adjustment was performed on the log-levels rather than the differences to prevent the amplification of seasonal high-frequency noise during the differencing operation. The transformation sequence is defined as:

$$y_t^{SA} = \text{X-13-ARIMA-SEATS}\left(\ln \text{GDP}_t\right)$$
$$\Delta y_t^{SA} = y_t^{SA} - y_{t-1}^{SA}$$

### 3.3 Bayesian Vector Autoregression Specification
The benchmark macroeconomic model is a Bayesian Vector Autoregression (B-VAR) specified as:

$$\mathbf{Y}_t = \sum_{l=1}^{p} \mathbf{A}_l \mathbf{Y}_{t-l} + \mathbf{u}_t, \qquad \mathbf{u}_t \sim \mathcal{N}(\mathbf{0}, \boldsymbol{\Sigma})$$

where $\mathbf{Y}_t$ is a $K \times 1$ vector of endogenous variables, $\mathbf{A}_l$ are $K \times K$ coefficient matrices, and $\boldsymbol{\Sigma}$ is the residual covariance matrix. The system includes $K = 7$ variables ordered for Cholesky identification as follows: Brent crude oil price, USD/KZT exchange rate, Industrial Production Index, freight turnover, aggregate credit, and real GDP growth. The lag order was set to $p = 2$ as selected by the Bayesian Information Criterion (BIC), subject to a maximum lag constraint of eight quarters to preserve degrees of freedom.

To mitigate over-parameterization, the system is estimated using the Minnesota Prior, which centers the parameter space on a random walk for non-stationary data or white noise for stationary transformations. The prior mean and variance are defined analytically as:

$$\mathbb{E}[a_{ij,l}] = \begin{cases} 1 & \text{if } i = j,\ l = 1 \\ 0 & \text{otherwise} \end{cases}$$

$$\text{Var}(a_{ij,l}) = \begin{cases} \dfrac{\lambda_1^2}{l^{\lambda_3}} & \text{if } i = j \\[6pt] \dfrac{\lambda_1^2}{l^{\lambda_3}} \cdot \dfrac{\hat{\sigma}_i^2}{\hat{\sigma}_j^2} & \text{if } i \neq j \end{cases}$$

The baseline rolling evaluation employs hyperparameters $\lambda_1 = 0.1$, $\lambda_2 = 0.5$, and $\lambda_3 = 1.0$. The overall tightness parameter $\lambda_1$ controls the degree of shrinkage toward the prior mean, while $\lambda_2$ governs cross-variable interactions and $\lambda_3$ enforces stronger shrinkage at higher lag orders. The posterior mean matrix $\mathbf{B}_{\text{post}}$ is derived via a precision-weighted combination of the prior and the Ordinary Least Squares (OLS) estimate. For Structural Impulse Response Function (IRF) analysis, a dedicated full-sample estimator was employed, decoupled from the rolling evaluation loop to prevent the adaptive COVID-era shrinkage from attenuating the structural coefficients. During this full-sample IRF estimation, a ridge regularization guard of $1\times 10^{-6}$ was applied to the precision matrix because the condition number exceeded the stability threshold, ensuring numerical stability during inversion. The Cholesky identification scheme places Brent crude oil first, reflecting its strict exogeneity to the domestic Kazakhstani economy, and real GDP last, implying that GDP responds contemporaneously to all system shocks but other variables react to GDP only with a one-quarter lag.

### 3.4 MIDAS Specification
The Mixed Data Sampling (MIDAS) framework incorporates high-frequency indicators directly into the quarterly GDP nowcast without pre-aggregation. The model employs an Exponential Almon polynomial to parameterize the lag weights:

$$\Delta y_t^{SA} = \alpha + \sum_{k=1}^{K} \beta_k \sum_{j=1}^{m_k} w_j(\boldsymbol{\theta}_k)\, x_{k,3t-j} + \gamma \Delta y_{t-1}^{SA} + \varepsilon_t$$

$$w_j(\boldsymbol{\theta}) = \frac{\exp(\theta_1 j + \theta_2 j^2)}{\sum_{s=1}^{m} \exp(\theta_1 s + \theta_2 s^2)}$$

The MIDAS specification utilizes three high-frequency predictors: the Industrial Production Index, freight turnover, and the Short-Term Economic Indicator. The optimal monthly lag length $m_k$ for each predictor was selected via a BIC grid search over six, nine, and twelve lags, yielding $m = 6$ for IPI, $m = 9$ for freight, and $m = 6$ for STEI. The non-linear Almon parameters $\theta_1$ and $\theta_2$ were estimated using a 49-point grid search over bounded starting values ($\theta_1 \in [-1.0, 0.5]$, $\theta_2 \in [-0.4, 0.15]$) to prevent weight degeneracy. The estimated parameters yielded $\theta_1 = -0.50$ and $\theta_2 = 0.0$ for IPI, and $\theta_1 = -0.25$ and $\theta_2 = -0.1$ for freight.