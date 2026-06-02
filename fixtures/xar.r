# ar1_sim_fit_base_r.R
# Simulate and fit a univariate AR(1) model using base R.

simulate_ar1 <- function(n, mu, phi, sigma, x0 = mu) {
  x <- numeric(n)
  x[1] <- mu + phi * (x0 - mu) + sigma * rnorm(1)

  for (i in 2:n) {
    x[i] <- mu + phi * (x[i - 1] - mu) + sigma * rnorm(1)
  }

  return(x)
}

fit_ar1 <- function(x) {
  n <- length(x)

  y <- x[2:n]
  lagx <- x[1:(n - 1)]

  fit <- lm(y ~ lagx)

  intercept <- coef(fit)[1]
  phi <- coef(fit)[2]
  mu <- intercept / (1 - phi)

  resid <- residuals(fit)
  sigma <- sqrt(sum(resid^2) / length(resid))

  y <- list(
    intercept = intercept,
    mu = mu,
    phi = phi,
    sigma = sigma,
    fitted = fitted(fit),
    resid = resid,
    lm_fit = fit
  )

  return(y)
}

set.seed(123)

n <- 1000
mu_true <- 2.0
phi_true <- 0.8
sigma_true <- 1.0

x <- simulate_ar1(n, mu_true, phi_true, sigma_true)

fit <- fit_ar1(x)

cat("true parameters\n")
cat("mu    =", mu_true, "\n")
cat("phi   =", phi_true, "\n")
cat("sigma =", sigma_true, "\n\n")

cat("fitted parameters\n")
cat("mu    =", fit$mu, "\n")
cat("phi   =", fit$phi, "\n")
cat("sigma =", fit$sigma, "\n\n")

cat("lm summary\n")
print(summary(fit$lm_fit))
