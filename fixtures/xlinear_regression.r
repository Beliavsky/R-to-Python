# test37_linear_regression_lm.R
set.seed(123)

n <- 50
x <- rnorm(n)
e <- rnorm(n, sd = 0.5)
y <- 1 + 2 * x + e

fit <- lm(y ~ x)
print(coef(fit))
print(fitted(fit)[1:5])
print(resid(fit)[1:5])
print(summary(fit))
