# Size of the Grubbs test at the 5% level under H0 (i.i.d. normal), n = 495 as for the AR(5)
# residuals: (a) p value of outliers::grubbs.test, (b) the Bonferroni t-based p value used in
# the Python implementation (Grubbs 1950; NIST), (c) the empirical 95% quantile of G.
suppressPackageStartupMessages(library(outliers))
set.seed(1); m <- 495; R <- 20000
G <- numeric(R); pa <- numeric(R); pb <- numeric(R)
for (r in 1:R) {
  x <- rnorm(m); g <- max(abs(x - mean(x))) / sd(x); G[r] <- g
  pa[r] <- grubbs.test(x, type = 10, two.sided = TRUE)$p.value
  t2 <- m * (m - 2) * g^2 / max((m - 1)^2 - m * g^2, 1e-12)
  pb[r] <- min(1, 2 * m * pt(sqrt(t2), m - 2, lower.tail = FALSE))
}
tc <- qt(0.05 / (2 * m), m - 2, lower.tail = FALSE)
gcrit <- (m - 1) / sqrt(m) * sqrt(tc^2 / (m - 2 + tc^2))
cat(sprintf("rejection rate at 5%%: outliers::grubbs.test %.4f; Bonferroni t (Python) %.4f\n", mean(pa < 0.05), mean(pb < 0.05)))
cat(sprintf("critical G: empirical 95%% quantile %.4f; Bonferroni t formula %.4f; MC s.e. of a rate ~ %.4f\n",
            quantile(G, 0.95), gcrit, sqrt(0.05 * 0.95 / R)))
