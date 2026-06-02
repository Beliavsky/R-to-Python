# mvn_mixture_em_base_r.R
# Simulate and fit a multivariate normal mixture using base R.

rmvnorm_chol <- function(n, mu, sigma) {
  p <- length(mu)
  z <- matrix(rnorm(n * p), nrow = n, ncol = p)
  y <- sweep(z %*% chol(sigma), 2, mu, "+")
  return(y)
}

dmvnorm_chol <- function(x, mu, sigma) {
  x <- as.matrix(x)
  p <- ncol(x)
  r <- chol(sigma)
  z <- backsolve(r, t(sweep(x, 2, mu, "-")), transpose = TRUE)
  q <- colSums(z^2)
  logdet <- 2 * sum(log(diag(r)))
  logdens <- -0.5 * (p * log(2 * pi) + logdet + q)
  y <- exp(logdens)
  return(y)
}

simulate_mvn_mixture <- function(n, prob, mu_list, sigma_list) {
  k <- length(prob)
  p <- length(mu_list[[1]])
  comp <- sample.int(k, size = n, replace = TRUE, prob = prob)
  x <- matrix(NA_real_, nrow = n, ncol = p)

  for (j in seq_len(k)) {
    idx <- which(comp == j)
    if (length(idx) > 0) {
      x[idx, ] <- rmvnorm_chol(length(idx), mu_list[[j]], sigma_list[[j]])
    }
  }

  colnames(x) <- paste0("x", seq_len(p))
  y <- list(x = x, comp = comp)
  return(y)
}

fit_mvn_mixture_em <- function(x, k, max_iter = 200, tol = 1.0e-8,
                               ridge = 1.0e-6, seed = 123) {
  set.seed(seed)

  x <- as.matrix(x)
  n <- nrow(x)
  p <- ncol(x)

  prob <- rep(1 / k, k)

  idx <- sample.int(n, k)
  mu <- vector("list", k)
  sigma <- vector("list", k)

  for (j in seq_len(k)) {
    mu[[j]] <- as.numeric(x[idx[j], ])
    sigma[[j]] <- var(x) + ridge * diag(p)
  }

  loglik_old <- -Inf

  for (iter in seq_len(max_iter)) {

    dens <- matrix(0.0, nrow = n, ncol = k)

    for (j in seq_len(k)) {
      dens[, j] <- prob[j] * dmvnorm_chol(x, mu[[j]], sigma[[j]])
    }

    denom <- rowSums(dens)
    denom[denom <= 0] <- .Machine$double.xmin

    tau <- dens / denom
    loglik <- sum(log(denom))

    nk <- colSums(tau)
    prob <- nk / n

    for (j in seq_len(k)) {
      w <- tau[, j]
      mu[[j]] <- colSums(x * w) / nk[j]

      xc <- sweep(x, 2, mu[[j]], "-")
      sigma[[j]] <- crossprod(xc, xc * w) / nk[j]
      sigma[[j]] <- sigma[[j]] + ridge * diag(p)
    }

    if (abs(loglik - loglik_old) < tol * (1.0 + abs(loglik_old))) {
      break
    }

    loglik_old <- loglik
  }

  class <- max.col(tau)

  y <- list(
    prob = prob,
    mu = mu,
    sigma = sigma,
    posterior = tau,
    class = class,
    loglik = loglik,
    n_iter = iter
  )

  return(y)
}

print_fit <- function(fit) {
  k <- length(fit$prob)

  cat("number of EM iterations:", fit$n_iter, "\n")
  cat("log likelihood:", fit$loglik, "\n\n")

  for (j in seq_len(k)) {
    cat("component", j, "\n")
    cat("probability:", fit$prob[j], "\n")
    cat("mean:\n")
    print(fit$mu[[j]])
    cat("covariance:\n")
    print(fit$sigma[[j]])
    cat("\n")
  }

  return(invisible(NULL))
}

set.seed(1)

n <- 1000
prob_true <- c(0.35, 0.65)

mu_true <- list(
  c(-2.0, 0.0),
  c(2.0, 1.0)
)

sigma_true <- list(
  matrix(c(1.0, 0.5,
           0.5, 1.5), nrow = 2, byrow = TRUE),
  matrix(c(1.2, -0.4,
          -0.4, 0.8), nrow = 2, byrow = TRUE)
)

sim <- simulate_mvn_mixture(n, prob_true, mu_true, sigma_true)

fit <- fit_mvn_mixture_em(sim$x, k = 2, max_iter = 500, tol = 1.0e-10)

cat("true probabilities:\n")
print(prob_true)
cat("\ntrue means:\n")
print(mu_true)
cat("\ntrue covariances:\n")
print(sigma_true)
cat("\n")

cat("fitted parameters:\n")
print_fit(fit)

cat("classification table, true component by fitted component:\n")
print(table(true = sim$comp, fitted = fit$class))
