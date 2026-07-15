# xlm.r
# simple multiple linear regression example for transpiler testing

set.seed(42)

n <- 200
x1 <- rnorm(n)
x2 <- runif(n, -1, 1)

eps <- rnorm(n, sd = 0.5)
y <- 1.5 + 2.0 * x1 - 0.7 * x2 + eps

df <- data.frame(y = y, x1 = x1, x2 = x2)
fit <- lm(y ~ x1 + x2, data = df)

cat("coefficients:\n")
print(coef(fit))

cat("\nmodel summary:\n")
print(summary(fit))

# small prediction set
newdf <- data.frame(
  x1 = c(-1.0, 0.0, 1.0),
  x2 = c(0.5, 0.0, -0.5)
)
yp <- predict(fit, newdata = newdf)

cat("\npredictions:\n")
print(yp)

# simple scalar diagnostics
res <- residuals(fit)
cat("\nRMSE:", sqrt(mean(res^2)), "\n")
cat("R-squared:", summary(fit)$r.squared, "\n")
