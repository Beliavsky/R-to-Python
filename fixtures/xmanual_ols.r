# test40_manual_ols.R
set.seed(123)

n <- 50
x <- rnorm(n)
y <- 1 + 2 * x + rnorm(n)

X <- cbind(1, x)
b <- solve(t(X) %*% X) %*% t(X) %*% y

print(b)
print(coef(lm(y ~ x)))
