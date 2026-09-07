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

### Bias parameter

The estimand ERI tracks is

```math
\delta_{NP}
=
\mathbb{E}[Y \mid S_{NP}]
-
\mathbb{E}[Y \mid S_P].
```

### Posterior entropy

At each grid point $g_j$, the first $g_j$ ordered non-probability units are
combined with $S_P$, a Bayesian model is fitted, and the differential entropy
of the posterior of $\delta_{NP}$ is estimated from the MCMC draws by kernel
density estimation:

```math
H_j
\approx
-\frac{1}{M}
\sum_{m=1}^{M}
\ln
\hat{p}_j
\left(
\delta_{NP}^{(m)}
\right).
```

### The entropy floor

Under Normal-Normal conjugate updating with independent posteriors, the
posterior variance of $\delta_{NP}$ is additive:

```math
\mathrm{Var}(\delta_{NP})
=
\frac{\sigma_{NP}^{2}}{g_j}
+
\frac{\sigma^{2}}{n_P}.
```

Since the first term is non-negative,

```math
\mathrm{Var}(\delta_{NP})
\geq
\frac{\sigma^{2}}{n_P}.
```

The posterior entropy therefore has the lower bound

```math
H_j
\geq
H_{\mathrm{floor}}
=
\frac{1}{2}
\ln
\left(
\frac{2\pi e\,\sigma^{2}}{n_P}
\right).
```

Adding more non-probability units cannot lower this floor; only a larger
probability sample can. This is *why* an elbow appears and how large a pool is
needed for one to be visible.

### Complex survey designs

The floor above uses the simple-random-sampling variance $\sigma^2/n_P$. Under
a complex probability-sample design involving stratification, clustering, or
unequal selection probabilities, the true variance of $\hat{\mu}_P$ is inflated
by the design effect $\mathrm{DEFF} \geq 1$:

```math
\mathrm{Var}(\hat{\mu}_P)
\approx
\frac{\sigma^{2}\mathrm{DEFF}}{n_P}.
```

Ignoring DEFF underestimates the floor and gives an over-optimistic read of how
much non-probability data is needed. The ANES 2012 application below applies
$\mathrm{DEFF}=1.3$ to both the floor and $k_{\mathrm{bal}}$; this is a scalar
correction to the theoretical benchmark and does not itself re-weight the
outcome model (see *Limitations*).

### Stopping rule

Let the grid-adjacent marginal reduction in posterior entropy be

```math
\Delta_j^{(g)}
=
\hat{H}_{j-1}
-
\hat{H}_j.
```

Define the total observed entropy reduction as

```math
\Delta_{\mathrm{total}}
=
\hat{H}_1
-
\min_j \hat{H}_j.
```

The ERI stopping point is

```math
k^*
=
g_{j^*},
```

where $j^*$ is the first grid point satisfying

```math
j^*
=
\min
\left\{
j \geq 2 :
0 \leq
\frac{\Delta_j^{(g)}}{\Delta_{\mathrm{total}}}
\lt
\epsilon
\right\}.
```

The non-negativity constraint excludes grid points where estimation noise
causes entropy to *rise* between adjacent points (KDE/MCMC noise). Without this
constraint, such a rise could satisfy the criterion for any positive $\epsilon$
and trigger a spurious early stop.

The default $\epsilon=0.05$ is chosen by analogy with the 5% rule for
scree-plot practice in factor analysis. If no grid point qualifies, $k^*$ is
set to the last grid point and the criterion is recorded as **not reached**.

### Pre-flight balance point

Before running ERI, the balance-point heuristic indicates whether the
non-probability pool is large enough for a visible elbow. Under a complex
probability-sample design with design effect $\mathrm{DEFF} \geq 1$:

```math
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
```

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

### Core functions

```r
# KDE-based Shannon entropy of posterior draws
# (Sheather-Jones bandwidth)
compute_entropy <- function(samples) {

  samples <- as.numeric(samples)

  if (length(samples) < 20 || var(samples) < 1e-12) {
    return(NA_real_)
  }

  d <- tryCatch(
    density(samples, bw = "SJ"),
    error = function(e) density(samples, bw = "nrd0")
  )

  p_vals <- approx(
    d$x,
    d$y,
    xout = samples
  )$y

  -mean(log(pmax(p_vals, 1e-10)))
}


# ERI stopping point:
# first grid point below the epsilon tolerance
eri_kstar <- function(H_vec, k_vec, epsilon = 0.05) {

  delta_total <- H_vec[1] - min(H_vec, na.rm = TRUE)

  if (delta_total < 1e-6) {
    return(max(k_vec))
  }

  # pct >= 0 guards against noise-driven entropy increases
  # spuriously triggering an early stop.
  pct <- c(NA, -diff(H_vec)) / delta_total

  idx <- which(
    !is.na(pct) &
    pct >= 0 &
    pct < epsilon
  )[1]

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

* **R** (≥ 4.0)
* Packages for the R analysis scripts (`ERI_full_analysis.R`,
  `ERI_ANES2012_application.R`):
  * `tidyverse`
  * `brms` (with a working Stan / C++ toolchain)
  * `patchwork`
  * `scales`
  * `haven` (to read the ANES `.dta` / `.sav` files)
  * `Cairo` (optional; higher-quality figure export on macOS)
* Additional packages for the Shiny dashboard (`app.R`):
  * `shiny`
  * `bslib`
  * `plotly`
  * `bsicons`
  * `rintrojs`
  * `DT`
  * `readxl`
  * `writexl`
  * `rmarkdown`
  * `tinytex` (optional; required only for PDF report generation)

Install the required packages with:

```r
pkgs <- c(
  "tidyverse",
  "brms",
  "patchwork",
  "scales",
  "haven",
  "Cairo",
  "shiny",
  "bslib",
  "plotly",
  "bsicons",
  "rintrojs",
  "DT",
  "readxl",
  "writexl",
  "rmarkdown",
  "tinytex"
)

install.packages(
  setdiff(pkgs, rownames(installed.packages()))
)
```

---

## Repository Structure

| File | Purpose |
|---|---|
| `ERI_full_analysis.R` | Full simulation study (DGP-1 to DGP-5), $\epsilon$-sensitivity, PST/DR benchmarks, the independence stress-test, and the structural-break toy example (Section 2.3 of the paper). Regenerates the simulation and toy-example figures and tables. |
| `ERI_ANES2012_application.R` | ANES 2012 political-efficacy application: data preparation, DEFF-adjusted pre-flight $k_{\mathrm{bal}}$ and floor-proximity $k_\eta$ diagnostics, Models A/B/C, sensitivity analyses, and benchmark tables. |
| `app.R` | Interactive Shiny dashboard containing the User Guide, live demonstration, and Apply to Your Data module. |
| `www/` | Static assets for the Shiny dashboard, including University of Manchester branding. Not required to run the analysis scripts. |

---

## Data Access

The ANES 2012 microdata are **not redistributed** in this repository.

Download the **ANES 2012 Time Series Study** from:

https://electionstudies.org/data-center/2012-time-series-study/

Place the downloaded file in the working directory before running
`ERI_ANES2012_application.R`.

The study is available in Stata (`.dta`) and SPSS (`.sav`) formats. The analysis
restricts the data to standard-splice respondents:

```r
randsplice_revisedstandard == 2
```

This ensures that a comparable four-item political-efficacy index is available
for both the face-to-face and web modes.

---

## Simulation Design

Five data-generating processes vary the strength and structure of the bias.

The simulations use:

* probability sample size: $n_P=80$;
* non-probability pool size: $N_{NP}=4{,}000$;
* sequential grid:

```math
\{20,\ 50,\ 100,\ 200,\ 400,\ 700,\ 1000,\ 1500,\ 2000,\ 3000,\ 4000\}.
```

| Scenario | $\lambda$ | Feature |
|---|---:|---|
| DGP-1 (base) | 1.25 | none |
| DGP-2 (strong bias) | 2.00 | none |
| DGP-3 (weak bias) | 1.05 | none |
| DGP-4 (covariate shift) | 1.25 | shifted NP covariate mean |
| DGP-5 (misspecification) | 1.25 | quadratic term omitted from the fitted model |

Across scenarios, posterior means at $k^*$ land within 6% of the true bias and
bootstrap intervals span at most two adjacent grid points.

A sensitivity analysis across four ordering schemes shows that although the
exact grid point $k^*$ shifts with ordering, the **stabilised posterior mean of
$\delta_{NP}$ is consistent across orderings within the plateau**.

### Toy example: why sequential auditing beats a static formula

A separate structural-break simulation (Section 2.3 of the paper) illustrates
why ERI monitors integration sequentially rather than relying on the static
balance-point formula alone.

A probability anchor ($n_P=100$) is paired with a $N_{NP}=1{,}500$-unit
non-probability panel containing a hidden break:

* the first 500 units are high-overlap;
* the remaining 1,000 units are severely biased and have substantially larger
  variance.

The static formula, which is blind to this structure, averages the two
variances and gives

```math
k_{\mathrm{bal}} \approx 758.
```

It therefore recommends incorporating 258 additional severely biased units
beyond the true break point.

ERI's sequential audit instead detects a stable plateau up to $k=500$ and
visibly breaks as the 501st biased unit enters, correctly signalling a stop at
the true break point.

---

## Application: Political Efficacy in the ANES 2012

The 2012 ANES dual-mode design pairs a face-to-face probability sample with a
GfK KnowledgePanel web sample answering identical efficacy items, providing an
ideal empirical setting for ERI.

The **raw web-minus-face-to-face gap is essentially zero**:

```math
\delta_{\mathrm{raw}} = +0.015,
\qquad
p = 0.857.
```

At face value, this suggests equivalence. ERI with covariate adjustment,
however, reveals that the near-zero raw gap conceals two offsetting components:

```math
\underbrace{+0.016}_{\mathrm{raw\ gap}}
=
\underbrace{+0.184}_{\mathrm{compositional}}
+
\underbrace{-0.167}_{\mathrm{residual}}.
```

The compositional component reflects the panel's older, higher-income,
more-educated profile — characteristics associated with *lower* efficacy in
the face-to-face sample.

The negative residual, whose 90% credible interval lies entirely below zero, is
consistent with, but does not uniquely identify, mechanisms such as volunteer
self-selection, panel conditioning, or differential item functioning.

### Pre-flight diagnostic

The DEFF-adjusted pre-flight diagnostic gives

```math
k_{\mathrm{bal}} = 525
```

and

```math
\frac{N_{NP}}{k_{\mathrm{bal}}}
=
3.59.
```

The available non-probability pool is therefore comfortably large enough for a
visible entropy elbow.

ERI identifies

```math
k^* = 1{,}884,
```

corresponding to the full web panel.

The observed entropy plateau is

```math
H_{\mathrm{plateau}} = -1.112
```

nats, closely approaching the design-adjusted theoretical floor

```math
H_{\mathrm{floor}} = -1.094.
```

---

## Two Notions of "Enough"

A distinctive contribution of the paper is to separate two operationally
different thresholds that can otherwise be conflated.

### 1. Balance point

The **balance point**, $k_{\mathrm{bal}}$, indicates when the non-probability
pool is large enough for a *visible elbow* — that is, when the entropy curve
has entered its relatively flat region.

This is the practical **pre-flight diagnostic**.

### 2. Floor proximity

The **floor-proximity threshold**, $k_\eta$, is the generally larger pool size
needed to bring the entropy to within $\eta$ nats of its theoretical floor.

A pool can therefore satisfy the first condition without satisfying the second:
it can be large enough for a visible elbow without being large enough to sit
essentially on the theoretical entropy floor.

In the ANES 2012 application,

```math
k_{0.10}
\approx
2{,}371,
```

whereas the available pool contains

```math
N_{NP}
=
1{,}884.
```

The theoretical formula predicts a residual gap of approximately

```math
0.123
```

nats at the full panel.

The actually observed gap is smaller:

```math
0.018
```

nats, with the plateau slightly *undercutting* the theoretical floor. This is
consistent with finite-sample KDE noise around the asymptotic floor rather than
evidence that the panel materially falls short.

---

## Interactive Dashboard

An interactive Shiny application is available at:

[ERI - Interactive Dashboard](https://dapr12.shinyapps.io/ERI-Toolkit/)

The dashboard reproduces the ERI workflow end to end.
### User Guide

Provides:

* motivation and intuition;
* methodological framework;
* glossary;
* mathematical definitions and equations.

### Demonstration

Runs the five simulation scenarios live and provides:

* interactive entropy curves;
* scree plots;
* bootstrap distributions of $k^*$;
* ordering comparisons;
* Model A/B/C outputs.

### Apply to Your Data

The dashboard can ingest:

* CSV;
* Excel;
* RDS;
* Stata (`.dta`);
* SPSS (`.sav`).

The module guides the user through variable mapping, runs the ERI pipeline, and
exports figures, tables, and an HTML/PDF report.

Run the application with:

```r
shiny::runApp("app.R")
```

By default, the dashboard draws $\delta_{NP}$ from the closed-form Normal
reference-prior posterior — the exact model under which the entropy floor
holds — and applies the paper's entropy, $k^*$, and bootstrap functions to
those draws.

A `brms` engine is also available for bit-faithful reproduction where a
working Stan toolchain is installed.

---

## Interpretation

The central principle of ERI is:

> **Adding non-probability units stops helping once the posterior entropy of the
> bias parameter reaches its floor — a bound determined by the probability
> sample, not by the size of the non-probability pool.**

Tracking the target mean itself provides no equivalent signal; the bias
parameter is the quantity with a model-implied lower bound on posterior
entropy.

ERI therefore turns the practical question

> **How much non-probability data is enough?**

into a reproducible and auditable stopping point that practitioners can
interpret, diagnose, and override when substantive considerations require it.

---

## Limitations

* The entropy floor assumes Normal-Normal updating. In finite,
  non-Gaussian samples it should be interpreted as a **heuristic lower bound**,
  not a guarantee.

* Finite-sample plateaus can arise from KDE artefacts, grid spacing, or MCMC
  noise. The elbow is therefore a diagnostic signal to interpret rather than a
  mathematical certificate.

* ERI is **ordering-dependent**. It estimates how quickly a particular ordering
  resolves uncertainty about $\delta_{NP}$; $k^*$ is not an intrinsic property
  of the non-probability source.

* The DEFF adjustment implemented here is a **scalar correction** to the entropy
  floor and $k_{\mathrm{bal}}$ only. The outcome models (Models A/B/C) remain
  fitted to unweighted complete cases.

* A fully design-consistent implementation using survey weights, replicate
  weights, or Taylor linearisation in the outcome model — together with DEFF
  estimated directly from the sample design rather than from a literature
  value — is a natural extension for production use.

* The simulations use a Gaussian, homoskedastic, linear structure. Heavy tails,
  strong positivity violations, and clustered designs remain to be
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

*Update the year, venue, volume, and publication details once the paper is
published.*

---

## Acknowledgements

The authors thank the American National Election Studies for making the 2012
Time Series Study publicly available.

This work was supported by the **British Academy**
[grant number `SRG2122\211173`].

---

## License

Specify a license for the repository (e.g. MIT).
