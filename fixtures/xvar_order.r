# var_sim_fit_base_r_no_data_frame.R
# Simulate and fit an arbitrary-order vector autoregression using base R, without data frames.

rmvnorm_chol <- function(n, mu, sigma) {
  p <- length(mu)
  z <- matrix(rnorm(n * p), nrow = n, ncol = p)
  y <- sweep(z %*% chol(sigma), 2, mu, "+")
  return(y)
}

simulate_var <- function(n, mu, a, sigma, x0 = NULL) {
  p <- length(mu)
  order <- length(a)

  if (order < 1) {
    stop("a must contain at least one coefficient matrix")
  }

  if (n <= order) {
    stop("n must be greater than length(a)")
  }

  for (j in 1:order) {
    if (!all(dim(a[[j]]) == c(p, p))) {
      stop("each element of a must be a p by p matrix, where p = length(mu)")
    }
  }

  if (!all(dim(sigma) == c(p, p))) {
    stop("sigma must be a p by p matrix, where p = length(mu)")
  }

  x <- matrix(NA_real_, nrow = n, ncol = p)

  if (is.null(x0)) {
    for (t in 1:order) {
      x[t, ] <- mu
    }
  } else {
    if (!all(dim(x0) == c(order, p))) {
      stop("x0 must be an order by p matrix")
    }
    x[1:order, ] <- x0
  }

  for (t in (order + 1):n) {
    mean_t <- mu

    for (j in 1:order) {
      mean_t <- mean_t + as.numeric(a[[j]] %*% (x[t - j, ] - mu))
    }

    x[t, ] <- rmvnorm_chol(1, mean_t, sigma)
  }

  colnames(x) <- paste0("x", 1:p)

  return(x)
}

make_var_design <- function(x, order) {
  x <- as.matrix(x)
  n <- nrow(x)
  p <- ncol(x)

  if (order < 1) {
    stop("order must be at least 1")
  }

  if (n <= order) {
    stop("nrow(x) must be greater than order")
  }

  y <- x[(order + 1):n, , drop = FALSE]
  xlag <- matrix(NA_real_, nrow = n - order, ncol = p * order)

  for (j in 1:order) {
    col1 <- (j - 1) * p + 1
    col2 <- j * p
    xlag[, col1:col2] <- x[(order + 1 - j):(n - j), , drop = FALSE]
  }

  design <- cbind(1.0, xlag)

  out <- list(y = y, design = design)
  return(out)
}

fit_var <- function(x, order) {
  x <- as.matrix(x)
  n <- nrow(x)
  p <- ncol(x)

  d <- make_var_design(x, order)

  y <- d$y
  design <- d$design

  coef_mat <- solve(crossprod(design), crossprod(design, y))

  intercept <- as.numeric(coef_mat[1, ])
  names(intercept) <- paste0("c", 1:p)

  a <- vector("list", order)

  for (j in 1:order) {
    row1 <- 1 + (j - 1) * p + 1
    row2 <- 1 + j * p

    a[[j]] <- t(coef_mat[row1:row2, , drop = FALSE])
    rownames(a[[j]]) <- paste0("eq", 1:p)
    colnames(a[[j]]) <- paste0("lag", j, "_x", 1:p)
  }

  fitted <- design %*% coef_mat
  resid <- y - fitted

  sigma <- crossprod(resid) / nrow(resid)
  rownames(sigma) <- colnames(x)
  colnames(sigma) <- colnames(x)

  amat <- diag(p)

  for (j in 1:order) {
    amat <- amat - a[[j]]
  }

  if (abs(det(amat)) < 1.0e-12) {
    mu <- rep(NA_real_, p)
  } else {
    mu <- as.numeric(solve(amat, intercept))
  }

  names(mu) <- paste0("mu", 1:p)

  out <- list(
    intercept = intercept,
    mu = mu,
    a = a,
    sigma = sigma,
    fitted = fitted,
    resid = resid,
    coef = coef_mat
  )

  return(out)
}

intercept_from_mu_a <- function(mu, a) {
  p <- length(mu)
  amat <- diag(p)

  for (j in 1:length(a)) {
    amat <- amat - a[[j]]
  }

  intercept <- as.numeric(amat %*% mu)
  names(intercept) <- paste0("c", 1:p)

  return(intercept)
}

print_var_fit <- function(fit) {
  order <- length(fit$a)

  cat("intercept:\n")
  print(fit$intercept)

  cat("\nmean:\n")
  print(fit$mu)

  for (j in 1:order) {
    cat("\na[[", j, "]]:\n", sep = "")
    print(fit$a[[j]])
  }

  cat("\ninnovation covariance matrix:\n")
  print(fit$sigma)

  return(invisible(NULL))
}

print_var_differences <- function(fit, mu_true, a_true, sigma_true) {
  order <- length(a_true)

  intercept_true <- intercept_from_mu_a(mu_true, a_true)

  cat("intercept difference, fitted - true:\n")
  print(fit$intercept - intercept_true)

  cat("\nmean difference, fitted - true:\n")
  print(fit$mu - mu_true)

  for (j in 1:order) {
    cat("\na[[", j, "]] difference, fitted - true:\n", sep = "")
    print(fit$a[[j]] - a_true[[j]])
  }

  cat("\ninnovation covariance difference, fitted - true:\n")
  print(fit$sigma - sigma_true)

  return(invisible(NULL))
}

set.seed(123)

n <- 1000

mu_true <- c(1.0, -2.0)

a_true <- list(
  matrix(c(0.50,  0.20,
          -0.10,  0.40), nrow = 2, byrow = TRUE),

  matrix(c(0.10, -0.05,
           0.05,  0.10), nrow = 2, byrow = TRUE)
)

sigma_true <- matrix(c(1.0, 0.4,
                       0.4, 2.0), nrow = 2, byrow = TRUE)

intercept_true <- intercept_from_mu_a(mu_true, a_true)

x <- simulate_var(n, mu_true, a_true, sigma_true)

fit <- fit_var(x, order = length(a_true))

cat("true intercept:\n")
print(intercept_true)

cat("\ntrue mean:\n")
print(mu_true)

for (j in 1:length(a_true)) {
  cat("\ntrue a[[", j, "]]:\n", sep = "")
  print(a_true[[j]])
}

cat("\ntrue innovation covariance matrix:\n")
print(sigma_true)

cat("\n\nfitted parameters:\n")
print_var_fit(fit)

cat("\n\ndifferences between fitted and true parameters:\n")
print_var_differences(fit, mu_true, a_true, sigma_true)
