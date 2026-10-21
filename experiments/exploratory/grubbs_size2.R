suppressPackageStartupMessages({library(outliers); library(data.table)})
set.seed(1); m <- 495; R <- 20000
pa <- pf <- numeric(R)
for (r in 1:R) { x <- rnorm(m)
  pa[r] <- grubbs.test(x, type = 10, two.sided = TRUE)$p.value
  pf[r] <- grubbs.test(x, type = 10, two.sided = FALSE)$p.value }
cat(sprintf("H0 rejection at 5%%: two.sided=TRUE %.4f; two.sided=FALSE %.4f; ratio of p values %.3f\n",
            mean(pa < 0.05), mean(pf < 0.05), median(pa / pf)))
# Re-score P on the data with two.sided = FALSE: p = p(two.sided=TRUE) / 2 if the ratio is exactly 2
