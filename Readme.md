
# How Much Non-Probability Data Is Enough?

### An Entropy-Based Stopping Rule for Sample Integration

This repository accompanies the paper:

**Pérez Ruiz, D.A. & Avalos Valdebenito, C. (2026)**  
*How Much Non-Probability Data Is Enough? An Entropy-Based Stopping Rule for
Sample Integration, with an Application to Political Efficacy in the ANES 2012.*

*Diego A. Pérez Ruiz, Department of Social Statistics, University of Manchester.*  
*Constanza Avalos Valdebenito, Cathie Marsh Institute, University of Manchester,
and the Alan Turing Institute.*

---

## Abstract

Survey practitioners increasingly supplement probability samples with
non-probability sources such as online panels, administrative records, or web
data. A basic unanswered question is: **how many non-probability units should be
incorporated before additional data yields negligible benefit?**

This repository implements the **Entropy-Regulated Integration (ERI)** stopping
rule. ERI monitors the Shannon entropy of the posterior of a *bias parameter*
$\delta_{NP}$, the gap in outcome means between the non-probability source and
the probability anchor, as non-probability units are added sequentially along a
pre-specified geometric grid under a propensity-score ordering. Because this
entropy has a natural lower bound set by the probability sample size, the curve
is elbow-shaped (analogous to a scree plot), and the stopping point $k^*$ is
the first grid point at which the marginal entropy reduction falls below a
user-specified tolerance $\epsilon$.

---

## Methodological Summary

We combine two samples measuring the same continuous outcome $Y$:

* a **probability sample** $S_P$ of size $n_P$ with a known design, treated as
  the (approximately unbiased) anchor;
* a **non-probability pool** $S_{NP}$ of size $N_{NP}$ with an unknown selection
  mechanism.

**Bias parameter.** The estimand ERI tracks is

$$
\delta_{NP} = \mathbb{E}[Y \mid S_{NP}] - \mathbb{E}[Y \mid S_P].
$$

**Posterior entropy.** At each grid point $g_j$, the first $g_j$ ordered
non-probability units are combined with $S_P$, a Bayesian model is fitted, and
the differential entropy of the posterior of $\delta_{NP}$ is estimated from
the MCMC draws by kernel density estimation:

$$
H_j \approx -\frac{1}{M}
\sum_{m=1}^{M}
\ln \hat{p}_j\left(\delta_{NP}^{(m)}\right).
$$
**The entropy floor (Proposition 1).** Under Normal-Normal conjugate updating
with independent posteriors, the posterior variance of $\delta_{NP}$ is additive,
so its entropy is bounded below by a floor fixed by the probability sample:

$$
\mathrm{Var}(\delta_{NP})
=
\frac{\sigma_{NP}^2}{g_j}
+
\frac{\sigma^2}{n_P}
\geq
\frac{\sigma^2}{n_P}.
$$

Therefore,

$$
H_j
\geq
H_{\mathrm{floor}}
=
\frac{1}{2}
\ln\left(
\frac{2\pi e\,\sigma^2}{n_P}
\right).
$$

Adding more non-probability units cannot lower this floor; only a larger
probability sample can. This is *why* an elbow appears and how large a pool is
needed for one to be visible.

**Complex survey designs.** The floor above uses the simple-random-sampling
variance $\sigma^2/n_P$. Under a complex probability-sample design
(stratification, clustering, unequal probabilities), the true variance of
$\hat{\mu}_P$ is inflated by the design effect $\mathrm{DEFF} \geq 1$:

$$
\mathrm{Var}(\hat{\mu}_P)
\approx
\frac{\sigma^2 \mathrm{DEFF}}{n_P}.
$$

Ignoring DEFF underestimates the floor and gives an over-optimistic read of how
much non-probability data is needed. The ANES 2012 application below applies
$\mathrm{DEFF}=1.3$ to both the floor and $k_{\mathrm{bal}}$; this is a scalar
correction to the theoretical benchmark and does not itself re-weight the
outcome model (see *Limitations*).

**Stopping rule.** With grid-adjacent marginal reduction

$$
\Delta_j^{(g)}
=
\hat{H}_{j-1}
-
\hat{H}_j
$$

and total range

$$
\Delta_{\mathrm{total}}
=
\hat{H}_1
-
\min_j \hat{H}_j,
$$

the ERI stopping point is

$$
k^*
=
g_{j^*},
$$

where

$$
j^*
=
\min
\left\{
j \geq 2 :
0 \leq
\frac{\Delta_j^{(g)}}{\Delta_{\mathrm{total}}}
\mathrel{\lt}
\epsilon
\right\}.
$$

The non-negativity constraint excludes grid points where estimation noise
causes entropy to *rise* between adjacent points (KDE/MCMC noise); without it,
such a rise would otherwise satisfy the criterion for any positive $\epsilon$
and trigger a spurious early stop.

The default $\epsilon=0.05$ is chosen by analogy with the 5% rule for
scree-plot practice in factor analysis. If no grid point qualifies, $k^*$ is
set to the last grid point (the criterion was "not reached").

**Pre-flight balance point.** Before running ERI, the balance-point heuristic
indicates whether the pool is large enough for a visible elbow. Under a complex
probability-sample design with design effect $\mathrm{DEFF} \geq 1$:

$$
k_{\mathrm{bal}}
=
\left\lceil
\frac{
n_P
\left(
\hat{\sigma}_{NP}/\hat{\sigma}_P
\right)^2
}{
\mathrm{DEFF}
}
\right\rceil.
$$

If $N_{NP} \ll k_{\mathrm{bal}}$, the pool is likely insufficient and ERI
should not proceed.
---

## Computational Approach

* **Bayesian models** are fitted with `brms` (Stan) using a Normal likelihood.
  Three specifications are used:
  * **Model A**: `Y ~ 1 + np` (unadjusted); the coefficient on the source
    indicator is $\delta_{NP}$ and drives the primary ERI criterion.
  * **Model B**: adds covariates; the source coefficient is the residual
    difference after adjustment.
  * **Model C**: a source-by-covariate interaction model for
    coefficient-specific diagnostics.
* **Entropy estimation** uses a KDE with the **Sheather-Jones** bandwidth
  (with an `nrd0` fallback). Differential entropy can be negative for
  concentrated posteriors. This is expected, not an error.
* **Bootstrap.** A $B=500$ resample-the-draws procedure gives a stability
  interval for $k^*$. This reflects entropy-estimation (MCMC) noise only; it
  is *not* a sampling confidence interval, and it does not capture ordering or
  model uncertainty.
* **Ordering.** ERI is **ordering-dependent**. Propensity-score ordering
  (highest-overlap units first) is recommended; the code also supports random,
  Mahalanobis-distance, and reverse-propensity orderings for sensitivity checks.

```{r core-functions}
# Core primitives (see ERI_full_analysis.R)

# KDE-based Shannon entropy of posterior draws (Sheather-Jones bandwidth)
compute_entropy <- function(samples) {
  samples <- as.numeric(samples)
  if (length(samples) < 20 || var(samples) < 1e-12) return(NA_real_)

  d <- tryCatch(
    density(samples, bw = "SJ"),
    error = function(e) density(samples, bw = "nrd0")
  )

  p_vals <- approx(d$x, d$y, xout = samples)$y
  -mean(log(pmax(p_vals, 1e-10)))
}

# ERI stopping point: first grid point below the epsilon tolerance
# pct >= 0 guards against noise-driven entropy increases spuriously
# triggering an early stop; see "Stopping rule" above.
eri_kstar <- function(H_vec, k_vec, epsilon = 0.05) {
  delta_total <- H_vec[1] - min(H_vec, na.rm = TRUE)

  if (delta_total < 1e-6) {
    return(max(k_vec))
  }

  pct <- c(NA, -diff(H_vec)) / delta_total
  idx <- which(!is.na(pct) & pct >= 0 & pct < epsilon)[1]

  if (is.na(idx)) {
    return(max(k_vec))
  }

  k_vec[idx]
}
```

---

## Reproducibility

All results in the paper are reproducible from this repository.

### Requirements

* R (≥ 4.0)
* Packages for the R analysis scripts (`ERI_full_analysis.R`,
  `ERI_ANES2012_application.R`):
  * `tidyverse`
  * `brms` (with a working Stan / C++ toolchain)
  * `patchwork`
  * `scales`
  * `haven` (to read the ANES `.dta` / `.sav` files)
  * `Cairo` (optional; higher-quality figure export on macOS)
* Additional packages for the Shiny dashboard (`app.R`):
  * `shiny`, `bslib`, `plotly`, `bsicons`, `rintrojs`, `DT`
  * `readxl`, `writexl` (optional; Excel upload/export in the "Your Data" tab)
  * `rmarkdown` (for the downloadable HTML/PDF report)
  * `tinytex` (optional; required only for the PDF report download, not the
    HTML one)

```{r install}
pkgs <- c(
  "tidyverse", "brms", "patchwork", "scales", "haven", "Cairo",
  "shiny", "bslib", "plotly", "bsicons", "rintrojs", "DT",
  "readxl", "writexl", "rmarkdown", "tinytex"
)

install.packages(setdiff(pkgs, rownames(installed.packages())))
```

### Repository structure

| File | Purpose |
|------|---------|
| `ERI_full_analysis.R` | Full simulation study (DGP-1 to DGP-5), $\epsilon$-sensitivity, PST/DR benchmarks, the independence stress-test, and the structural-break toy example (Section 2.3 of the paper); regenerates the simulation and toy-example figures and tables. |
| `ERI_ANES2012_application.R` | The ANES 2012 political-efficacy application: data preparation, the DEFF-adjusted pre-flight $k_{\mathrm{bal}}$ and floor-proximity ($k_\eta$) diagnostics, Models A/B/C, sensitivity and benchmark tables. |
| `app.R` | Interactive Shiny dashboard (User Guide, live demonstration, and an "apply to your own data" module); see below. |
| `www/` | Static assets for the Shiny dashboard (University of Manchester branding). Not required to run the analysis scripts. |

### Data access

The ANES 2012 microdata are **not redistributed** here. Download the Time Series
Study from the American National Election Studies and place the file in the
working directory before running the application script:

* https://electionstudies.org/data-center/2012-time-series-study/

The 2012 study is offered in Stata (`.dta`) and SPSS (`.sav`) formats (there is
no CSV). The analysis restricts to standard-splice respondents
(`randsplice_revisedstandard == 2`) so that a comparable four-item efficacy
index is available in both the face-to-face and web modes.

---

## Simulation Design

Five data-generating processes vary the strength and structure of the bias
(probability sample $n_P=80$, non-probability pool $N_{NP}=4{,}000$, grid
$\{20,50,100,200,400,700,1000,1500,2000,3000,4000\}$):

| Scenario | $\lambda$ | Feature |
|----------|-----------|---------|
| DGP-1 (base) | 1.25 | none |
| DGP-2 (strong bias) | 2.00 | none |
| DGP-3 (weak bias) | 1.05 | none |
| DGP-4 (covariate shift) | 1.25 | shifted NP covariate mean |
| DGP-5 (misspecification) | 1.25 | quadratic term omitted from the fitted model |

Across scenarios, posterior means at $k^*$ land within 6% of the true bias and
bootstrap intervals span at most two adjacent grid points. A sensitivity
analysis across four ordering schemes shows that although the exact grid point
$k^*$ shifts with ordering, the **stabilised posterior mean of
$\delta_{NP}$ is consistent across orderings within the plateau**.

### Toy example: why sequential auditing beats a static formula

A separate structural-break simulation (paper, Section 2.3) illustrates why ERI
monitors integration sequentially rather than relying on the static
balance-point formula alone. A probability anchor ($n_P=100$) is paired with a
$N_{NP}=1{,}500$-unit non-probability panel that has a hidden break: the first
500 units are high-overlap, the remaining 1,000 are severely biased with much
larger variance.

The static formula, blind to this structure, averages the two variances together
and yields $k_{\mathrm{bal}} \approx 758$, recommending that 258 additional
severely biased units be ingested past the true break point. ERI's sequential
audit instead detects a stable plateau up to $k=500$ and visibly breaks as the
501st (biased) unit enters, correctly signalling a stop at the true break point.

---

## Application: Political Efficacy in the ANES 2012

The 2012 ANES dual-mode design pairs a face-to-face probability sample with a
GfK KnowledgePanel web sample answering identical efficacy items, an ideal
setting for ERI.

The **raw web-minus-face-to-face gap is essentially zero** ($+0.015$ points,
$p=0.857$), suggesting equivalence. ERI with covariate adjustment tells a
different story: the near-zero raw gap conceals two offsetting parts,

$$
+0.016
=
\underbrace{+0.184}_{\mathrm{compositional}}
+
\underbrace{-0.167}_{\mathrm{residual}}.
$$

Thus, the raw gap can be represented as

$$
\underbrace{+0.016}_{\mathrm{raw\ gap}}
=
\underbrace{+0.184}_{\mathrm{compositional}}
+
\underbrace{-0.167}_{\mathrm{residual}}.
$$

The compositional component reflects the panel's older, higher-income,
more-educated profile (traits associated with *lower* efficacy in the
face-to-face sample), and the negative residual (90% CI entirely below zero) is
consistent with, but does not uniquely identify, volunteer self-selection,
panel conditioning, or differential item functioning.

The DEFF-adjusted pre-flight diagnostic gives $k_{\mathrm{bal}}=525$ with

$$
\frac{N_{NP}}{k_{\mathrm{bal}}} = 3.59,
$$

comfortably sufficient for a visible entropy elbow. ERI identifies
$k^*=1{,}884$ (the full web panel); the observed entropy plateau
($-1.112$ nats) closely approaches the design-adjusted theoretical floor
($-1.094$ nats).

---

## Two Notions of "Enough"

A distinctive contribution of the paper is separating two operationally
distinct thresholds that are easily conflated:

* **Balance point $k_{\mathrm{bal}}$**: when the pool is large enough for a
  *visible elbow* (the entropy curve has entered its flat region). This is the
  practical pre-flight check.
* **Floor-proximity $k_\eta$**: the (larger) pool size needed to bring the
  entropy *within $\eta$ nats of the floor*.

A pool can satisfy the first without satisfying the second: enough for an elbow,
but not enough to sit on the theoretical floor. In the ANES 2012 application,

$$
k_{0.10} \approx 2{,}371,
$$

the pool size needed to close to within 0.10 nats of the design-adjusted floor
at a fixed conventional tolerance, while $N_{NP}=1{,}884$.

The formula predicts a residual gap of approximately $0.123$ nats at the full
panel. The gap actually observed is smaller ($0.018$ nats, with the plateau
slightly *undercutting* the floor), consistent with finite-sample KDE noise
around the asymptotic floor rather than the panel falling short.

---

## Interactive Dashboard

An interactive Shiny application (`app.R`) reproduces the workflow end to end:

* a **User Guide** with the motivation, theory, glossary, and rendered equations;
* a **Demonstration** tab running the five simulation scenarios live, with
  interactive entropy curves, the scree plot, the bootstrap distribution of
  $k^*$, ordering comparisons, and the Model A/B/C outputs;
* an **Apply to Your Data** module that ingests CSV/Excel/RDS or the native ANES
  `.dta`/`.sav`, guides variable mapping, runs the full ERI pipeline, and exports
  figures, tables, and an HTML/PDF report.

```{r run-app}
shiny::runApp("app.R")
```

By default the dashboard draws $\delta_{NP}$ from the closed-form Normal
reference-prior posterior, the exact model under which the entropy floor holds,
and runs the paper's own entropy, $k^*$, and bootstrap functions on those
draws; a `brms` engine is available for bit-faithful reproduction where a Stan
toolchain is present.

---

## Interpretation

The key principle of the framework is:

> Adding non-probability units stops helping once the posterior entropy of the
> bias parameter reaches its floor, a bound set by the **probability** sample,
> not by the size of the non-probability pool. Tracking the target mean gives no
> such signal; only the bias parameter has a model-implied lower bound on
> entropy.

ERI turns "how much is enough?" into a reproducible, auditable stopping point
that practitioners can interpret and override.

---

## Limitations

* The entropy floor assumes Normal-Normal updating; in finite, non-Gaussian
  samples it is a **heuristic** lower bound, not a guarantee.
* Finite-sample plateaus can arise from KDE artefacts, grid spacing, or MCMC
  noise. The elbow is a signal to interpret, not a mathematical certificate.
* ERI is **ordering-dependent**: it estimates how quickly a given ordering
  resolves uncertainty about $\delta_{NP}$, not an intrinsic property of the
  source.
* The DEFF adjustment applied here is a **scalar correction** to the floor and
  $k_{\mathrm{bal}}$ only; the outcome model itself (Models A/B/C) remains
  fitted on unweighted complete cases. A fully design-consistent implementation,
  using survey weights, replicate weights, or Taylor linearisation in the
  outcome model, and DEFF estimated directly from the sample design rather than
  a literature value, is a natural extension for production use.
* The simulations share a Gaussian, homoskedastic, linear structure; heavy
  tails, strong positivity violations, and clustered designs remain to be
  stress-tested.

---

## Citation

If you use this code, please cite:

```bibtex
@article{PerezRuizAvalos2026ERI,
  author  = {P{\'e}rez Ruiz, Diego A. and Avalos Valdebenito, Constanza},
  title   = {How Much Non-Probability Data Is Enough? An Entropy-Based
             Stopping Rule for Sample Integration, with an Application to
             Political Efficacy in the ANES 2012},
  year    = {2026},
  note    = {Working paper. University of Manchester.}
}
```

*(Update the year, venue, and volume once the paper is published.)*

---

## Acknowledgements

The authors thank the American National Election Studies for making the 2012
Time Series Study publicly available. This work was supported by the British
Academy [grant number SRG2122\211173].

---

## License

Specify a license (e.g. MIT).

---

## License

Specify a license (e.g. MIT).
