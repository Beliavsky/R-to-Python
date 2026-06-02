# K-component univariate Gaussian mixture: simulate + EM fit (base R only)

fmt_vec <- function(x) {
  paste0("(", paste(sprintf("%.6f", x), collapse = ", "), ")")
}

simulate_mixnorm <- function(n, pi = c(0.6, 0.4), mu = c(-2, 2),
                             sigma = c(1, 1.5), seed = NULL) {
  # simulate from a K-component normal mixture

  if (!is.null(seed)) set.seed(seed)

  pi <- as.numeric(pi)
  mu <- as.numeric(mu)
  sigma <- as.numeric(sigma)
  k <- length(pi)

  stopifnot(k >= 1, length(mu) == k, length(sigma) == k)
  stopifnot(all(pi > 0), all(sigma > 0), n >= 1)
  pi <- pi / sum(pi)

  # sample component labels then sample observations
  z <- findInterval(runif(n), cumsum(pi)) + 1L
  z <- pmin(z, k)
  x <- rnorm(n, mean = mu[z], sd = sigma[z])

  list(x = x, z = z, true = list(n = n, pi = pi, mu = mu, sigma = sigma))
}

em_mixnorm <- function(x, k = NULL,
                       pi_init = NULL, mu_init = NULL, sigma_init = NULL,
                       max_iter = 500, tol = 1e-8, min_var = 1e-8,
                       verbose = FALSE) {
  # EM fit for a K-component normal mixture

  x <- as.numeric(x)
  x <- x[is.finite(x)]
  n <- length(x)
  stopifnot(n >= 2)

  if (!is.null(pi_init)) {
    k <- length(pi_init)
  } else if (!is.null(mu_init)) {
    k <- length(mu_init)
  } else if (!is.null(sigma_init)) {
    k <- length(sigma_init)
  } else if (is.null(k)) {
    k <- 2L
  }
  k <- as.integer(k)
  stopifnot(k >= 1)

  # initialize pi (weights)
  if (is.null(pi_init)) {
    pi <- rep(1 / k, k)
  } else {
    pi <- as.numeric(pi_init)
  }
  pi <- pi / sum(pi)

  # initialize mu (means)
  if (is.null(mu_init)) {
    probs <- seq(1 / (k + 1), k / (k + 1), length.out = k)
    mu <- as.numeric(quantile(x, probs = probs, names = FALSE, type = 7))
  } else {
    mu <- as.numeric(mu_init)
  }

  # initialize sigma (sds)
  if (is.null(sigma_init)) {
    s <- sd(x)
    if (!is.finite(s) || s <= 0) s <- 1
    sigma <- rep(s, k)
  } else {
    sigma <- as.numeric(sigma_init)
  }

  # validate + enforce minimum variance
  stopifnot(length(pi) == k, length(mu) == k, length(sigma) == k)
  stopifnot(all(pi > 0), all(sigma > 0))
  sigma <- pmax(sigma, sqrt(min_var))

  # EM iterations
  loglik <- numeric(max_iter)
  resp <- matrix(0, n, k)
  log_dens <- matrix(0, n, k)

  for (it in 1:max_iter) {

    # E-step: stable responsibilities via row-wise log-sum-exp
    for (j in 1:k) {
      log_dens[, j] <- log(pi[j]) + dnorm(x, mean = mu[j], sd = sigma[j], log = TRUE)
    }

    row_max <- apply(log_dens, 1, max)
    dens_shifted <- exp(log_dens - row_max)
    denom <- row_max + log(rowSums(dens_shifted))
    resp <- dens_shifted / rowSums(dens_shifted)

    # log-likelihood
    loglik[it] <- sum(denom)

    # M-step: update pi, mu, sigma
    nk <- colSums(resp)
    nk <- pmax(nk, .Machine$double.eps)
    pi <- nk / n
    mu <- colSums(resp * x) / nk

    for (j in 1:k) {
      sigma[j] <- sqrt(max(sum(resp[, j] * (x - mu[j])^2) / nk[j], min_var))
    }

    if (verbose) {
      cat(sprintf("iter %d  loglik=%.6f  pi=%s  mu=%s  sd=%s\n",
                  it, loglik[it], fmt_vec(pi), fmt_vec(mu), fmt_vec(sigma)))
    }

    if (it >= 2 && abs(loglik[it] - loglik[it - 1]) < tol * (1 + abs(loglik[it - 1]))) {
      loglik <- loglik[1:it]
      break
    }
  }

  cluster <- max.col(resp, ties.method = "first")

  list(pi = pi, mu = mu, sigma = sigma,
       loglik = loglik,
       resp = resp,
       cluster = cluster)
}

# ---- example ----
n_true <- 1000
pi_true <- c(0.7, 0.2, 0.1)
mu_true <- c(-1.5, 0.0, 1.0)
sd_true <- c(0.8, 1.4, 2.0)
k_true <- length(pi_true)

# ---- simulate data from known parameters ----
sim <- simulate_mixnorm(n = n_true, pi = pi_true, mu = mu_true, sigma = sd_true, seed = 1)

# ---- print true parameters ----
cat(sprintf("n = %d\n", sim$true$n))
cat("true pi    =", fmt_vec(sim$true$pi), "\n")
cat("true mu    =", fmt_vec(sim$true$mu), "\n")
cat("true sigma =", fmt_vec(sim$true$sigma), "\n\n")

# ---- fit K-component mixture by EM ----
x <- sim$x
fit <- em_mixnorm(x, k = k_true, max_iter = 500, tol = 1e-9, verbose = FALSE)

# ---- print fitted parameters and final log-likelihood ----
cat("estimated pi    =", fmt_vec(fit$pi), "\n")
cat("estimated mu    =", fmt_vec(fit$mu), "\n")
cat("estimated sigma =", fmt_vec(fit$sigma), "\n")
cat("final loglik:", tail(fit$loglik, 1), "\n")
