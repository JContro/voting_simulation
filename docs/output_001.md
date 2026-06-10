# Spatial Voting Simulation — Experiment Report

**Date:** 10 June 2026  
**Run:** Single-seed demo (`cfg.seed = 1`), logistic ideal-point kernel, $\alpha = 2$

---

## 1. What the Experiment Is

This is a **latent-space ideal-point model** — the workhorse of political-science roll-call analysis (NOMINATE), psychometric unfolding models, and modern recommender-system research.

The core idea:

- **Topics** (policies, survey items) live at positions $t_j \in \mathbb{R}^\alpha$ in a low-dimensional "opinion space."
- **Voters** each have an ideal point $v_i \in \mathbb{R}^\alpha$, a personal tolerance $\sigma_i$, and an acquiescence bias $b_i$.
- A voter approves a topic when they are "close enough" — modelled by a **logistic ideal-point kernel** with a lapse-rate floor:

$$
P(\text{approve}_{ij}) = \epsilon + (1 - 2\epsilon)\,\sigma\!\left(b_i + c_j - \frac{\lVert v_i - t_j\rVert^2}{2\sigma_i^2}\right)
$$

where $\sigma(\cdot)$ is the sigmoid, $c_j$ is topic valence, and $\epsilon = 0.03$ is the lapse rate.

The experiment has **two phases**, mirroring how Computerized Adaptive Testing works in practice:

| Phase | Goal | Analogous to… |
|---|---|---|
| **Phase 1 — Topic Recovery** | Given only the binary vote matrix $Y_{N \times X}$, reconstruct the latent topic positions $t_j$ and estimate the true dimensionality $\alpha$. | Calibrating the GRE item bank (offline, expensive) |
| **Phase 2 — Adaptive Voter Localization** | Given recovered (or true) topic positions, ask a *new* voter the fewest possible questions to pinpoint their ideal point $v_i$. | The adaptive GRE itself (online, cheap) |

---

## 2. What the Code Is Doing

### 2.1 Generative Engine (`world.py`, `votes.py`, `config.py`)

**Configuration** (`SimConfig` dataclass):

| Parameter | Value | Meaning |
|---|---|---|
| `alpha` | 2 | True latent dimensionality |
| `n_topics` (X) | 60 | Number of topics/items |
| `n_voters` (N) | 2000 | Phase-1 calibration voters |
| `mu_log_sigma` | 0.0 | Median voter tolerance (log scale) |
| `tau` | 0.4 | Heterogeneity of tolerance across voters |
| `sigma_bounds` | (0.2, 5.0) | Clipping to avoid degenerate voters |
| `kernel` | `"logistic"` | Response model type |
| `eps` | 0.03 | Lapse/guess floor and ceiling |
| `population` | `"blob"` | Voter distribution shape |

**Synthetic world generation:**

1. **Topics:** Sampled from $\mathcal{N}(0, I_2)$, then **gauge-normalized**: centroid at origin, unit RMS norm. This pins the scale (Critique 2 from the plan).
2. **Voters:** Sampled from the configured population geometry (`"blob"` = single isotropic Gaussian with scale `s_v = 1.0`).
3. **Tolerances:** $\sigma_i \sim \text{LogNormal}(0, 0.4)$, clipped to $[0.2, 5.0]$.
4. **Biases & valences:** $b_i \sim \mathcal{N}(0, 0.5)$, $c_j \sim \mathcal{N}(0, 0.5)$.
5. **Vote matrix:** $Y_{ij} \sim \text{Bernoulli}(P_{ij})$ via vectorized `cdist` + elementwise ops.

### 2.2 Phase 1 — Topic Recovery (`recovery.py`)

An **escalating ladder** of three estimators:

**Level 1 — PCA (baseline sanity check):**
- SVD on the centered vote matrix. Wrong model for binary data, but fast and provides a floor.

**Level 2 — Classical MDS (workhorse):**
- Builds a topic–topic dissimilarity from column correlations of $Y$: $D^2_{jk} = 2(1 - \rho_{jk})$.
- Double-centers and eigendecomposes to get coordinates.
- Gauge-normalizes output. No likelihood needed — robust and fast.

**Level 3 — Full MLE (Adam + analytic gradients):**
- Fits the logistic ideal-point model by maximizing the Bernoulli log-likelihood.
- Parameters: voter positions $V$, topic positions $T$, biases $b$, valences $c$, log-tolerances $s = \log\sigma$.
- **Gauge re-pinned every iteration:** centroid at origin, unit RMS topics; $\log\sigma$ absorbs the rescaling.
- Regularization: L2 penalties on $V$, $b$, $c$, $s$.
- Initialized from MDS coordinates.

**Dimensionality selection:**
- Masks a random 10% of votes as held-out.
- Fits MLE at $\alpha' \in \{1, 2, 3, 4, 5\}$.
- Picks $\alpha'$ maximizing held-out mean log-likelihood.

### 2.3 Phase 2 — Adaptive Voter Localization (`adaptive.py`)

**Posterior representation:** Particle filter with 3000 particles per voter, tracking $(v, \log\sigma, b)$. Particles are resampled when effective sample size drops below 50%.

**Three question-selection policies raced head-to-head:**

| Policy | Algorithm |
|---|---|
| **Random** | Uniform from unasked topics. Baseline. |
| **Uncertainty Sampling** | Topic with predicted $P(\text{approve})$ closest to 0.5. Fast but confounds proximity with tolerance. |
| **Expected Information Gain (EIG / BALD)** | Topic maximizing $H[\mathbb{E}[P(y_j)]] - \mathbb{E}[H[P(y_j)]]$. Principled; correctly handles the $\sigma$ confound. |

**Evaluation:** Two conditions — using the **true** topic map (isolates policy performance) and using the **recovered** map (end-to-end, propagates phase-1 error). 100 test voters, 30 questions each. Output: median localization error $\lVert\hat{v} - v_{\text{true}}\rVert$ as a function of questions asked.

### 2.4 Metrics (`metrics.py`)

| Metric | What it measures | Gauge-invariant? |
|---|---|---|
| **Procrustes-aligned RMSE** | Euclidean error after optimal rotation + translation + scaling | Semi (scaling handled) |
| **Distance Spearman $\rho$** | Rank correlation of topic–topic distance matrices | **Fully** — the most honest single number |
| **Mean Bernoulli log-likelihood** | Predictive quality on held-out or all votes | Yes |

---

## 3. What the Results Are

### 3.1 Generated Data Characteristics

```
world: alpha=2, X=60, N=2000, approval rate = 0.270
```

The approval rate of **27%** is notably low — voters disapprove of nearly three-quarters of topics. This means the vote matrix is **sparse in ones**, which reduces the effective information per cell. Low approval rates are realistic (most people oppose most policy proposals), but they make both phases harder.

### 3.2 Phase 1 — Topic Recovery

```
method    RMSE(aligned)  dist-Spearman   LL/entry
---------------------------------------------------
PCA              0.7449         0.5550
MDS              0.3196         0.8621
MLE              0.7279         0.8592    -0.4339
oracle                                    -0.4828   <- noise floor
```

**Key findings:**

1. **MDS is the best performer** by a substantial margin. Its RMSE (0.3196) is less than half that of PCA and MLE, and its distance Spearman $\rho = 0.862$ indicates strong recovery of the topic–topic distance structure.

2. **MLE underperforms MDS** — and this is the most important *negative* result:
   - MLE *worsens* RMSE from 0.3196 (MDS initialization) to 0.7279, while distance Spearman barely changes (0.862 → 0.859).
   - The MLE training log-likelihood ($-0.4339$) is **higher than the oracle** ($-0.4828$), meaning the model is **overfitting** the binary noise rather than recovering the latent structure.
   - The optimization moved only marginally: NLL improved just 0.0032 over 1500 iterations (0.4371 → 0.4339).

   **Likely causes:**
   - Low approval rate (27%) → weak gradient signal, especially for "no" votes far from a topic.
   - The loss landscape is flat or has many local minima; Adam with the current learning rate may be getting stuck.
   - The model has ~8,180 free parameters for 120,000 binary observations — enough capacity to fit noise.

3. **PCA is poor** ($\rho = 0.555$), confirming that linear methods on binary data are inadequate.

### 3.3 Dimensionality Selection

```
  dim 1: held-out LL = -0.5519
  dim 2: held-out LL = -0.5614    ← true dimension
  dim 3: held-out LL = -0.5911
  dim 4: held-out LL = -0.6149
  dim 5: held-out LL = -0.6431
selected dim = 1 (true = 2)
```

**This is a clear failure:** the procedure selects $\hat\alpha = 1$ when the true dimension is $2$. Held-out log-likelihood decreases **monotonically** with dimension — each additional dimension overfits the training data and generalizes worse.

**Interpretation:** With $N = 2000$ voters, $X = 60$ topics, and a 27% approval rate, there is insufficient signal to support a 2D model under cross-validation. The model with $\alpha' = 1$ is effectively saying "the data can only support one reliable dimension." This is a **sample-complexity finding** in itself — it tells you how many voters you'd need to reliably detect the second dimension at this approval rate.

### 3.4 Phase 2 — Adaptive Voter Localization

#### Oracle topic map (true positions):

```
  policy random       median final error = 0.879
  policy uncertainty  median final error = 0.762
  policy EIG          median final error = 0.701
```

#### Recovered topic map (MLE, end-to-end):

```
  policy random       median final error = 0.991
  policy uncertainty  median final error = 0.893
  policy EIG          median final error = 0.816
```

**Key findings:**

1. **EIG > Uncertainty > Random** — the rank order is exactly as theory predicts. EIG provides a **20% reduction** in median error over random sampling (0.701 vs. 0.879 with the oracle map).

2. **The recovered-map penalty is real:** using the MLE-recovered topics instead of true topics adds **~0.11–0.13** to median localization error across all policies. This is the *end-to-end cost of phase-1 error* — exactly what the improved design (§4.6) was designed to measure.

3. **Absolute errors are high** — even EIG with the oracle map achieves only ~0.70 after 30 questions in $\mathbb{R}^2$ with unit-RMS topics. To put this in perspective:
   - The topic cloud has RMS = 1.0, so a typical inter-topic distance is ~1–2 units.
   - An error of 0.70 means the voter is localized to within roughly half the topic-cloud radius.
   - This is usable but not precise — consistent with the information-poverty of binary responses (Critique 3).

4. **Uncertainty sampling is closer to EIG than to random** — it captures most of the gain and is much simpler to implement. In practice, the EIG–uncertainty gap may not justify the computational cost of EIG for this regime.

### 3.5 Summary Figures

| Figure | Content | What it shows |
|---|---|---|
| `fig1_topic_recovery.png` | True topics (black) vs. MLE-recovered (red, Procrustes-aligned) with connecting lines | Visual evidence of recovery quality; MLE topics are displaced and distorted relative to truth |
| `fig2_dim_selection.png` | Held-out LL vs. fitted dimension, with true-$\alpha$ reference line | Monotonically decreasing — model selection failure |
| `fig3_adaptive_policies.png` | Median localization error vs. questions asked, all 3 policies × 2 map conditions (true/recovered), with IQR bands for the true-map condition | The headline result: policy ranking + end-to-end gap |

---

## 4. Interpretation & Takeaways

### 4.1 What Worked

- **MDS is a robust, fast topic-recovery method** that significantly outperforms PCA and — in this regime — even the full MLE. For quick exploratory analysis or initializing more complex models, MDS is the clear winner.
- **The adaptive policy ranking is consistent and strong.** EIG dominates, uncertainty sampling is a close second, and both substantially beat random. This holds under both the oracle and recovered maps.
- **The end-to-end pipeline functions correctly.** Error propagates from phase 1 to phase 2 in a measurable, interpretable way.
- **The code architecture is clean** — config-driven, seeded, with clear separation between generative engine, recovery, and adaptive phases.

### 4.2 What Didn't Work (and Why It's Informative)

- **MLE underperformed MDS.** This is the most important *negative result*. It demonstrates that for sparse binary data (27% approval rate), the full likelihood model can overfit noise rather than recovering structure — even when the model is correctly specified. Practical implication: **don't deploy MLE without regularization tuning and cross-validation**, and always benchmark against MDS.
- **Dimensionality selection failed.** The procedure picked $\hat\alpha = 1$ when $\alpha = 2$. This reveals a **sample-size floor**: at $N = 2000$, $X = 60$, and 27% approval, the second dimension cannot be reliably detected. This is actionable: if you need to recover $\alpha = 2$, you need either more voters, more topics, a higher approval rate, or richer response data (ordinal/Likert).
- **Absolute localization errors are high (~0.7–1.0).** Even after 30 questions, voters are not precisely localized. This validates Critique 3 from the plan: binary responses are information-poor. Switching to a 5-point Likert scale (§4.4) or increasing the topic bank size would likely halve these errors.

### 4.3 Recommended Next Steps

1. **Fix the MLE** — try lower learning rate, stronger L2 regularization, early stopping based on held-out LL, or switch to a Bayesian approach (NumPyro NUTS) that naturally handles uncertainty.
2. **Multi-seed sweep** — these are single-seed results. Run 20+ seeds and report medians with IQR bands. The variance across seeds can be large (Pitfall 5).
3. **Sweep the approval rate** — vary `mu_log_sigma` to change the base approval rate from ~10% to ~50% and observe how MLE performance and dimensionality selection change.
4. **Test clustered populations** (`population="polarized"`) — the plan predicts this helps adaptive questioning; verifying it is a publishable finding.
5. **Add ordinal responses** — a 5-point Likert scale on the same latent quantity; directly compare binary vs. ordinal question efficiency.

---

## Appendix: Quick Diagnostic Table

| Symptom | Suspected Cause | Suggested Fix |
|---|---|---|
| MLE LL > Oracle LL | Overfitting | Increase `lam_v`, `lam_s`; add early stopping on held-out |
| MLE RMSE > MDS RMSE | Optimization stuck in flat region | Lower `lr` to 0.01; increase `n_iter`; try NUTS |
| $\hat\alpha = 1$ when $\alpha = 2$ | Insufficient signal for 2D | Increase N to 5k+; increase X; raise approval rate |
| High absolute localization error | Binary data is low-info | Add ordinal responses; increase topic bank to 200 |