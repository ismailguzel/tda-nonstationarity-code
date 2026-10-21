# Independent R implementation of the classical tests of Section 6.3 / Appendix E, used to
# cross-check experiments/exploratory/stat_tests.py (Python). Standard packages where one
# exists: urca (ADF), FinTS (ARCH-LM), sandwich (Newey-West), outliers (Grubbs); the sup-LM
# kink test and the two long-run-variance CUSUM statistics have no package version and are
# re-implemented here from their formulas. strucchange's textbook OLS-CUSUM (i.i.d. scale) is
# added for reference. Input: series.csv (standardised series), meta.csv. Output: r_tests.csv.
suppressPackageStartupMessages({
  library(data.table); library(urca); library(FinTS); library(sandwich)
  library(strucchange); library(outliers); library(parallel)
})
set.seed(42)
Z <- as.matrix(fread("series.csv", header = FALSE)); n <- ncol(Z)
L <- floor(4 * (n / 100)^(2 / 9))                       # Newey-West lags (5 for n = 500)
tt <- seq_len(n) - 1
lrv <- function(s) {                                     # Bartlett-kernel long-run variance
  v <- sum(s^2) / length(s)
  for (k in seq_len(L)) v <- v + 2 * (1 - k / (L + 1)) * sum(s[-(1:k)] * s[1:(length(s) - k)]) / length(s)
  max(v, 1e-12)
}
pkolm <- function(x) { k <- 1:100; min(1, max(0, 2 * sum((-1)^(k - 1) * exp(-2 * k^2 * x^2)))) }
ar_resid <- function(z, p = 5) { E <- embed(z, p + 1); resid(lm(E[, 1] ~ E[, -1])) }
taus <- seq(floor(0.15 * n), floor(0.85 * n))
X0 <- cbind(1, tt); P0 <- X0 %*% solve(crossprod(X0), t(X0))
W <- sapply(taus, function(tau) pmax(tt - tau, 0)); U <- W - P0 %*% W
suplm <- function(z) {
  e0 <- as.vector(z - P0 %*% z); S <- U * e0
  max(colSums(S)^2 / (n * apply(S, 2, lrv)))
}
null <- replicate(5000, suplm(rnorm(n))); crit_R <- unname(quantile(null, 0.95, type = 7))
cat(sprintf("R: sup-LM 5%% critical value %.3f\n", crit_R))

one <- function(i) {
  z <- Z[i, ]
  adf <- ur.df(z, type = "drift", lags = floor(12 * (n / 100)^0.25), selectlags = "AIC")
  e <- ar_resid(z)
  arch <- ArchTest(e, lags = 5, demean = FALSE)
  fit <- lm(z ~ tt); V <- NeweyWest(fit, lag = L, prewhite = FALSE, adjust = FALSE)
  wT <- coef(fit)[2]^2 / V[2, 2]
  ed <- z - mean(z); cm <- max(abs(cumsum(ed))) / sqrt(n * lrv(ed))
  x <- e^2 - mean(e^2); cq <- max(abs(cumsum(x))) / sqrt(length(x) * lrv(x))
  g <- grubbs.test(e, type = 10, two.sided = TRUE)
  sc <- sctest(efp(z ~ 1, type = "OLS-CUSUM"))
  c(S_stat = adf@teststat[1], S_cv5 = adf@cval[1, "5pct"],
    V_stat = unname(arch$statistic), V_p = unname(arch$p.value),
    T_stat = unname(wT), T_p = pchisq(unname(wT), 1, lower.tail = FALSE),
    R_stat = suplm(z),
    M_stat = cm, M_p = pkolm(cm), M_iid_p = unname(sc$p.value),
    Q_stat = cq, Q_p = pkolm(cq),
    P_stat = unname(g$statistic[1]), P_p = unname(g$p.value))
}
t0 <- Sys.time()
res <- do.call(rbind, mclapply(seq_len(nrow(Z)), one, mc.cores = 12))
res <- as.data.table(res); res[, R_crit := crit_R]
fwrite(res, "r_tests.csv")
cat(sprintf("R: %d series in %.0f s\n", nrow(res), as.numeric(difftime(Sys.time(), t0, units = "secs"))))
