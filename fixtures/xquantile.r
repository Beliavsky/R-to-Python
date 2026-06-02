# test50_empirical_cdf_quantile.R
x <- c(5, 1, 3, 2, 4, 100)

print(quantile(x))
print(quantile(x, probs = c(0.1, 0.5, 0.9)))
f <- ecdf(x)
print(f(c(1, 3, 10, 100)))
print(summary(x))
