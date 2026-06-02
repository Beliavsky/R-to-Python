# ar_sim_fit_base_r_no_data_frame.R
# Simulate and fit a univariate AR(p) model using base R, without data frames.

simulate_ar <- function(n, mu, phi, sigma, x0 = NULL) {
  p <- length(phi)
  x <- numeric(n)

  if (n <= p) {
    stop("n must be greater than length(phi)")
  }

  if (is.null(x0)) {
    x[1:p] <- mu
  } else {
    if (length(x0) != p) {
      stop("x0 must have length equal to length(phi)")
    }
    x[1:p] <- x0
  }

  for (t in (p + 1):n) {
    lag_values <- x[(t - 1):(t - p)]
    x[t] <- mu + sum(phi * (lag_values - mu)) + sigma * rnorm(1)
  }

  return(x)
}

fit_ar <- function(x, p) {
  n <- length(x)

  if (p < 1) {
    stop("p must be at least 1")
  }

  if (n <= p) {
    stop("length(x) must be greater than p")
  }

  y <- x[(p + 1):n]

  xlag <- matrix(NA_real_, nrow = n - p, ncol = p)

  for (j in 1:p) {
    xlag[, j] <- x[(p + 1 - j):(n - j)]
  }

  design <- cbind(1.0, xlag)

  coef_vec <- solve(crossprod(design), crossprod(design, y))

  intercept <- coef_vec[1]
  phi <- as.numeric(coef_vec[-1])
  names(phi) <- paste0("phi", 1:p)

  phi_sum <- sum(phi)

  if (abs(1 - phi_sum) < 1.0e-12) {
    mu <- NA_real_
  } else {
    mu <- intercept / (1 - phi_sum)
  }

  fitted <- as.numeric(design %*% coef_vec)
  resid <- y - fitted
  sigma <- sqrt(sum(resid^2) / length(resid))

  y <- list(
    intercept = intercept,
    mu = mu,
    phi = phi,
    sigma = sigma,
    fitted = fitted,
    resid = resid,
    coef = coef_vec
  )

  return(y)
}

set.seed(123)

n <- 1000
mu_true <- 2.0
phi_true <- c(0.6, -0.2, 0.1)
sigma_true <- 1.0

x <- simulate_ar(n, mu_true, phi_true, sigma_true)

fit <- fit_ar(x, p = length(phi_true))

cat("true parameters\n")
cat("mu    =", mu_true, "\n")
cat("phi   =", phi_true, "\n")
cat("sigma =", sigma_true, "\n\n")

cat("fitted parameters\n")
cat("mu    =", fit$mu, "\n")
cat("phi   =", fit$phi, "\n")
cat("sigma =", fit$sigma, "\n\n")

cat("regression coefficients\n")
print(fit$coef)
