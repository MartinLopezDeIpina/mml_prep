# Mathematics for Machine Learning — Exam Study Guide

## Overview

The exam follows a consistent structure: **4 exercises, 3 hours**, covering Linear Algebra, Calculus, Optimization, and Probability. A reference sheet with key definitions/theorems is provided (e.g., Chebyshev, Taylor, Hoeffding, KL divergence, Cauchy-Schwarz, Jensen).

---

## 1. Linear Algebra

### 1.1 Eigenvalues of Structured Matrices ⭐⭐⭐

The most recurring exam pattern: given a matrix with known structure, find eigenvalues, eigenvectors, and derived quantities.

**Key structures to master:**

- **Rank-one:** $A = vv^T$ has eigenvalue $\|v\|^2$ (eigenvector $v$) and eigenvalue $0$ with multiplicity $n-1$ (eigenspace $v^\perp$).
- **Rank-two:** $A = aa^T + bb^T$. Work in $\text{span}(a, b)$: restrict the $2 \times 2$ problem there. Everything in $\text{span}(a,b)^\perp$ has eigenvalue $0$.
- **Shifted identity:** $A = I + vv^T$ has eigenvalue $1 + \|v\|^2$ (eigenvector $v$) and eigenvalue $1$ with multiplicity $n-1$.

**Workflow:** Structure → Eigenvalues → Trace, Det, Singular Values, Rank.

### 1.2 Trace and Determinant Identities ⭐⭐⭐

Once eigenvalues $\lambda_1, \dots, \lambda_n$ are known:

$$\text{tr}(A) = \sum_{i=1}^n \lambda_i, \qquad \det(A) = \prod_{i=1}^n \lambda_i$$

Essential identities:

- $\text{tr}(AB) = \text{tr}(BA)$ (cyclic property)
- $\text{tr}(A^k) = \sum \lambda_i^k$ for Hermitian $A$
- $\det\begin{pmatrix} A & B \\ C & D \end{pmatrix} = \det(A)\det(D - CA^{-1}B)$ when $A$ invertible
- $\det(I_m - FG) = \det(I_n - GF)$

### 1.3 Rayleigh Quotient and Spectral Optimization ⭐⭐⭐

For symmetric $A$ with eigenvalues $\lambda_1 \leq \dots \leq \lambda_n$:

$$\max_{\|x\|=1} x^TAx = \lambda_n, \qquad \min_{\|x\|=1} x^TAx = \lambda_1, \qquad \max_{\|x\|=1} \|Ax\|_2 = \sigma_{\max}$$

**Exam trigger:** Any optimization on the unit sphere → decompose spectrally.

### 1.4 Singular Values ⭐⭐

- For symmetric PSD: $\sigma_i = \lambda_i$. For symmetric: $\sigma_i = |\lambda_i|$.
- Perturbation: $|\sigma_k(A+E) - \sigma_k(A)| \leq \|E\|_2$.
- For rank-one $A = uv^*$: unique nonzero singular value is $\sigma = \|u\|\|v\|$.
- $\|A\|_2 = \sigma_{\max}$, $\|A\|_F = \sqrt{\sum \sigma_i^2}$.

### 1.5 Orthogonal Matrices ⭐⭐

- $Q^TQ = QQ^T = I$, so $Q^{-1} = Q^T$.
- $\det(Q) = \pm 1$ (proof: $\det(Q^TQ) = \det(Q)^2 = 1$).
- Preserves norms, inner products, eigenvalues have $|\lambda|=1$.

### 1.6 Rank, Kernel, Range ⭐

- $\text{rank}(A) = \dim(\text{ran}(A)) = n - \dim(\ker(A))$
- For $A = uv^T$: $\text{ran}(A) = \text{span}(u)$, $\ker(A) = v^\perp$, $\text{rank} = 1$.

---

## 2. Calculus

### 2.1 Matrix Calculus Identities ⭐⭐⭐

The most directly tested topic. Memorize these:

| Expression | Derivative |
|---|---|
| $\text{tr}(AB)$ w.r.t. $A$ | $B^T$ |
| $\log\det(A)$ w.r.t. $a_{rs}$ | $(A^{-1})_{sr}$ |
| $\det(A^{-1})$ w.r.t. $a_{rs}$ | $-\det(A^{-1})(A^{-1})_{sr}$ |
| $A(\alpha)^{-1}$ w.r.t. $\alpha$ | $-A^{-1}\frac{\partial A}{\partial \alpha}A^{-1}$ |
| $\det(I + \alpha T)$ w.r.t. $\alpha$ at $\alpha=0$ | $\text{tr}(T)$ |

**Exam 2026 asked directly:** Compute $\frac{\partial}{\partial \alpha}\text{tr}((B+\alpha X)^{-1})\big|_{\alpha=0}$. Apply the inverse rule, then the trace.

### 2.2 Taylor Expansion (Multivariate, Second Order) ⭐⭐⭐

$$f(x) \approx f(x_0) + \nabla f(x_0)^T(x - x_0) + \frac{1}{2}(x - x_0)^T Hf(x_0)(x - x_0)$$

**Workflow for functions like** $f(x) = \sqrt{1 + \|x\|^2}$ **at** $x_0 = 0$:

1. Compute $f(0)$.
2. Compute $\nabla f$ and evaluate at $0$.
3. Compute the Hessian $Hf$ and evaluate at $0$.
4. Plug into the formula.

### 2.3 Gradient and Hessian of Common Functions ⭐⭐

- **Quadratic form:** $\nabla_x(x^TAx) = (A + A^T)x$, and if $A$ symmetric: $= 2Ax$.
- **Log-sum-exp:** $f(x) = \log\sum_i e^{x_i}$ → $\nabla f = \text{softmax}(x)$, Hessian $= \text{diag}(s) - ss^T$ where $s = \text{softmax}(x)$.
- **Squared norm:** $\nabla_x\|x\|^2 = 2x$, $\nabla_x\|Ax - b\|^2 = 2A^T(Ax-b)$.

### 2.4 Critical Points and Classification ⭐⭐

1. Solve $\nabla f(x_0) = 0$.
2. Compute Hessian $Hf(x_0)$.
3. $Hf \succ 0$ → local min, $Hf \prec 0$ → local max, indefinite → saddle.
4. If $Hf$ is semidefinite (degenerate), argue from the definition directly.

### 2.5 Directional Derivative ⭐

$$D_v f(x) = \nabla f(x)^T v$$

Check whether the problem expects $v$ to be a unit vector. If not stated, the formula above applies to any $v$.

---

## 3. Optimization

### 3.1 Lagrangian and KKT Conditions ⭐⭐⭐

**The single most important exam topic.** For:

$$\min_x f(x) \quad \text{s.t.} \quad g_i(x) \leq 0, \quad h_j(x) = 0$$

**Lagrangian:**

$$\mathcal{L}(x, \lambda, \mu) = f(x) + \sum_i \lambda_i g_i(x) + \sum_j \mu_j h_j(x)$$

**KKT conditions (all four):**

1. **Stationarity:** $\nabla_x \mathcal{L} = 0$
2. **Primal feasibility:** $g_i(x) \leq 0$, $h_j(x) = 0$
3. **Dual feasibility:** $\lambda_i \geq 0$
4. **Complementary slackness:** $\lambda_i g_i(x) = 0$ for all $i$

### 3.2 Entropy Maximization on the Simplex ⭐⭐⭐

This problem appeared in **Sheet 6, Sheet 7, Sheet 10, Exam 2026**. Know it by heart:

$$\max -\sum x_i \log x_i \quad \text{s.t.} \quad \sum x_i = 1, \; x_i \geq 0$$

**Solution sketch:**

- Lagrangian: $\mathcal{L} = -\sum x_i\log x_i - \mu(\sum x_i - 1) - \sum \lambda_i(-x_i)$
- Stationarity: $-\log x_i - 1 + \mu + \lambda_i = 0$
- Complementary slackness + feasibility → $x_i = \frac{1}{n}$ for all $i$.

The Exam 2026 variant uses the KL divergence form $\sum x_i(\log x_i - \log p_i)$. Same technique, solution becomes $x_i = p_i$.

### 3.3 Convexity of Functions ⭐⭐⭐

Three equivalent characterizations (for $f \in C^2$):

| Condition | Statement |
|---|---|
| **Definition** | $f(\lambda x + (1-\lambda)y) \leq \lambda f(x) + (1-\lambda)f(y)$ |
| **First-order** | $f(y) \geq f(x) + \nabla f(x)^T(y - x)$ (lies above tangent) |
| **Second-order** | $Hf(x) \succeq 0$ for all $x$ |

### 3.4 Strong Convexity ⭐⭐

$f$ is $m$-strongly convex ($m > 0$) iff any of these hold:

- $g(x) = f(x) - \frac{m}{2}\|x\|^2$ is convex.
- $f(\lambda x + (1-\lambda)y) \leq \lambda f(x) + (1-\lambda)f(y) - \frac{m}{2}\lambda(1-\lambda)\|x-y\|^2$.
- $f(y) \geq f(x) + \nabla f(x)^T(y-x) + \frac{m}{2}\|x-y\|^2$.
- $Hf(x) \succeq mI$ for all $x$.

**Key consequence:** strongly convex → unique global minimum.

### 3.5 Local Min = Global Min for Convex Problems ⭐⭐

**Proof by contradiction:** Suppose $x$ is local min but $\exists y$ with $f(y) < f(x)$. By convexity, $f(\lambda y + (1-\lambda)x) \leq \lambda f(y) + (1-\lambda)f(x) < f(x)$ for any $\lambda \in (0,1)$. Taking $\lambda$ small enough, this point is in the local neighborhood, contradicting local minimality.

### 3.6 Duality ⭐⭐

**Workflow:**

1. Write Lagrangian $\mathcal{L}(x, \lambda, \mu)$.
2. Minimize over $x$: $g(\lambda, \mu) = \inf_x \mathcal{L}(x, \lambda, \mu)$ (dual function).
3. Dual problem: $\max_{\lambda \geq 0, \mu} g(\lambda, \mu)$.
4. If strong duality holds: primal optimum = dual optimum.

### 3.7 Convexity of Sets ⭐

- **Intersection** of convex sets → convex (even uncountable intersections).
- **Union** → generally **not** convex.
- **Minkowski sum** $C_1 + C_2$ → convex.
- **Balls** $\{x : \|x\| \leq r\}$ under any norm → convex.

---

## 4. Probability

### 4.1 Gaussian Norm Concentration ⭐⭐⭐

For $X \sim \mathcal{N}(0, \sigma^2 I_d)$:

$$\mathbb{E}[\|X\|^2] = d\sigma^2, \qquad \text{Var}[\|X\|^2] = 2d\sigma^4$$

Then apply Chebyshev to bound $P\big(|\|X\|^2 - d\sigma^2| > t\big) \leq \frac{2d\sigma^4}{t^2}$.

**Punchline:** In high dimensions, $\|X\|^2 \approx d\sigma^2$ with high probability. Samples concentrate on a thin shell of radius $\approx \sigma\sqrt{d}$.

### 4.2 Concentration Inequalities ⭐⭐⭐

**Chebyshev** (needs only mean and variance):

$$P(|Y - \mathbb{E}[Y]| \geq k) \leq \frac{\text{Var}(Y)}{k^2}$$

**Hoeffding** (for bounded independent r.v.s $X_i \in [a_i, b_i]$):

$$P\!\left(\left|\frac{1}{n}\sum X_i - \mathbb{E}\!\left[\frac{1}{n}\sum X_i\right]\right| \geq t\right) \leq 2\exp\!\left(\frac{-2n^2t^2}{\sum(b_i - a_i)^2}\right)$$

**When to use which:** Chebyshev for Gaussian-based problems (unbounded). Hoeffding for bounded r.v.s (e.g., uniform on $[0,1]$, coin flips).

### 4.3 Volume Concentration in High Dimensions ⭐⭐⭐

**Shell argument:** For $0 < \varepsilon \ll 1$:

$$\frac{\text{vol}(B(0, 1-\varepsilon))}{\text{vol}(B(0,1))} = (1 - \varepsilon)^d \xrightarrow{d \to \infty} 0$$

Almost all volume is in the thin outer shell $B(0,1) \setminus B(0, 1-\varepsilon)$.

**Cube equator:** For $X$ uniform on $[0,1]^d$, Hoeffding gives $P\big(|\sum X_i - d/2| > \sqrt{d\log d}\big) \leq \frac{2}{d^2}$.

### 4.4 Quadratic Forms of Gaussians ⭐⭐

For $X \sim \mathcal{N}(0, I_d)$ and symmetric $A$:

$$\mathbb{E}[\langle X, AX \rangle] = \text{tr}(A)$$

**Proof:** Expand $\langle X, AX\rangle = \sum_{i,j} a_{ij}X_iX_j$, use $\mathbb{E}[X_iX_j] = \delta_{ij}$.

**Corollary:** For any $B \in \mathbb{R}^{m \times d}$: $\mathbb{E}[\|BX\|^2] = \|B\|_F^2$.

### 4.5 KL Divergence and ELBO ⭐⭐

$$D_{KL}(q \| p) = \mathbb{E}_q\!\left[\log\frac{q(w)}{p(w)}\right] \geq 0$$

**ELBO derivation:**

$$\log p(y) = \mathbb{E}_q\!\left[\log\frac{p(y|w)p(w)}{q(w)}\right] + D_{KL}(q(w) \| p(w|y))$$

Since $D_{KL} \geq 0$:

$$\log p(y) \geq \mathbb{E}_q[\log p(y|w)] - D_{KL}(q \| p)$$

### 4.6 Variance of Linear Combinations ⭐⭐

$$\text{Var}\!\left(\sum a_i X_i\right) = \sum a_i^2 \text{Var}(X_i) + \sum_{i \neq j} a_i a_j \text{Cov}(X_i, X_j)$$

If independent: $= \sum a_i^2 \text{Var}(X_i)$.

### 4.7 MLE for Linear Regression ⭐⭐

Given $y = X\theta + \varepsilon$, $\varepsilon \sim \mathcal{N}(0, \sigma^2 I)$:

1. Likelihood: $P(Y|X;\theta) = \prod_i \frac{1}{\sqrt{2\pi}\sigma}\exp\!\left(-\frac{(y_i - x_i^T\theta)^2}{2\sigma^2}\right)$
2. Log-likelihood: $\ell(\theta) = -\frac{1}{2\sigma^2}\|Y - X\theta\|^2 + \text{const}$
3. Set $\nabla_\theta \ell = 0$: $X^TX\theta = X^TY$
4. Solution: $\hat{\theta} = (X^TX)^{-1}X^TY$

---

## Quick Reference: What to Expect Per Exercise

| Exercise | Core Topic | Most Likely Question Type |
|---|---|---|
| **1** | Linear Algebra | Eigenvalues of structured matrix → trace, det, singular values → spectral optimization |
| **2** | Calculus | Matrix derivative identity + Taylor expansion + possibly a Gaussian concentration bound |
| **3** | Optimization | Lagrangian / KKT on a constrained problem (entropy-like or geometric) + convexity proof |
| **4** | Mixed / Probability | True/False with proofs + high-dimensional volume + KL/ELBO or MLE |
