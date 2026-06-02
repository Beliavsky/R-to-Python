# var1_sim_fit_base_r.R
# Simulate and fit a first-order vector autoregression using base R.

rmvnorm_chol <- function(n, mu, sigma) {
  p <- length(mu)
  z <- matrix(rnorm(n * p), nrow = n, ncol = p)
  y <- sweep(z %*% chol(sigma), 2, mu, "+")
  return(y)
}

simulate_var1 <- function(n, mu, a, sigma, x0 = NULL) {
  p <- length(mu)

  if (!all(dim(a) == c(p, p))) {
    stop("a must be a p by p matrix, where p = length(mu)")
  }

  if (!all(dim(sigma) == c(p, p))) {
    stop("sigma must be a p by p matrix, where p = length(mu)")
  }

  x <- matrix(NA_real_, nrow = n, ncol = p)

  if (is.null(x0)) {
    x[1, ] <- mu
  } else {
    if (length(x0) != p) {
      stop("x0 must have length equal to length(mu)")
    }
    x[1, ] <- x0
  }

  for (t in 2:n) {
    xlag_centered <- x[t - 1, ] - mu
    mean_t <- mu + as.numeric(a %*% xlag_centered)
    x[t, ] <- rmvnorm_chol(1, mean_t, sigma)
  }

  colnames(x) <- paste0("x", seq_len(p))
  return(x)
}

fit_var1 <- function(x) {
  x <- as.matrix(x)
  n <- nrow(x)
  p <- ncol(x)

  if (n < 3) {
    stop("x must have at least 3 rows")
  }

  y <- x[2:n, , drop = FALSE]
  xlag <- x[1:(n - 1), , drop = FALSE]

  design <- cbind(1.0, xlag)
  coef_mat <- solve(crossprod(design), crossprod(design, y))

  intercept <- as.numeric(coef_mat[1, ])
  a <- t(coef_mat[-1, , drop = FALSE])

  resid <- y - design %*% coef_mat
  sigma <- crossprod(resid) / nrow(resid)

  amat <- diag(p) - a
  if (abs(det(amat)) < 1.0e-12) {
    mu <- rep(NA_real_, p)
  } else {
    mu <- as.numeric(solve(amat, intercept))
  }

  names(intercept) <- paste0("c", seq_len(p))
  names(mu) <- paste0("mu", seq_len(p))
  rownames(a) <- paste0("eq", seq_len(p))
  colnames(a) <- paste0("lag", seq_len(p))
  rownames(sigma) <- colnames(x)
  colnames(sigma) <- colnames(x)

  y <- list(
    intercept = intercept,
    mu = mu,
    a = a,
    sigma = sigma,
    resid = resid,
    fitted = design %*% coef_mat
  )

  return(y)
}

print_var1_fit <- function(fit) {
  cat("intercept:\n")
  print(fit$intercept)

  cat("\nmean:\n")
  print(fit$mu)

  cat("\na matrix:\n")
  print(fit$a)

  cat("\ninnovation covariance matrix:\n")
  print(fit$sigma)

  return(invisible(NULL))
}

set.seed(123)

n <- 1000

mu_true <- c(1.0, -2.0)

a_true <- matrix(c(0.6,  0.2,
                  -0.1,  0.5), nrow = 2, byrow = TRUE)

sigma_true <- matrix(c(1.0, 0.4,
                       0.4, 2.0), nrow = 2, byrow = TRUE)

x <- simulate_var1(n, mu_true, a_true, sigma_true)

fit <- fit_var1(x)

cat("true mean:\n")
print(mu_true)

cat("\ntrue a matrix:\n")
print(a_true)

cat("\ntrue innovation covariance matrix:\n")
print(sigma_true)

cat("\n\nfitted parameters:\n")
print_var1_fit(fit)
