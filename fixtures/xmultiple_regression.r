# test38_multiple_regression_lm.R
set.seed(123)

n <- 100
x1 <- rnorm(n)
x2 <- runif(n)
e <- rnorm(n, sd = 0.25)
y <- 1 + 2 * x1 - 3 * x2 + e

df <- data.frame(y = y, x1 = x1, x2 = x2)
fit <- lm(y ~ x1 + x2, data = df)

print(coef(fit))
print(summary(fit))
