# varma_sim_fit_order_select_base_r_no_data_frame.R
# Simulate and fit VARMA(p, q) models using base R, without data frames.

rmvnorm_chol <- function(n, mu, sigma) {
  d <- length(mu)
  z <- matrix(rnorm(n * d), nrow = n, ncol = d)
  y <- sweep(z %*% chol(sigma), 2, mu, "+")
  return(y)
}

simulate_varma <- function(n, mu, a, b, sigma, x0 = NULL) {
  d <- length(mu)
  p <- length(a)
  q <- length(b)
  order <- max(p, q)

  if (n <= order) {
    stop("n must be greater than max(length(a), length(b))")
  }

  if (p > 0) {
    for (i in 1:p) {
      if (!all(dim(a[[i]]) == c(d, d))) {
        stop("each AR matrix must be d by d")
      }
    }
  }

  if (q > 0) {
    for (j in 1:q) {
      if (!all(dim(b[[j]]) == c(d, d))) {
        stop("each MA matrix must be d by d")
      }
    }
  }

  if (!all(dim(sigma) == c(d, d))) {
    stop("sigma must be d by d")
  }

  x <- matrix(NA_real_, nrow = n, ncol = d)
  eps <- matrix(0.0, nrow = n, ncol = d)

  if (is.null(x0)) {
    for (t in 1:order) {
      x[t, ] <- mu
    }
  } else {
    if (!all(dim(x0) == c(order, d))) {
      stop("x0 must be an order by d matrix")
    }
    x[1:order, ] <- x0
  }

  for (t in (order + 1):n) {
    mean_t <- mu

    if (p > 0) {
      for (i in 1:p) {
        mean_t <- mean_t + as.numeric(a[[i]] %*% (x[t - i, ] - mu))
      }
    }

    if (q > 0) {
      for (j in 1:q) {
        mean_t <- mean_t + as.numeric(b[[j]] %*% eps[t - j, ])
      }
    }

    eps[t, ] <- rmvnorm_chol(1, rep(0.0, d), sigma)
    x[t, ] <- mean_t + eps[t, ]
  }

  colnames(x) <- paste0("x", 1:d)
  return(list(x = x, eps = eps))
}

intercept_from_mu_a <- function(mu, a) {
  d <- length(mu)
  p <- length(a)
  amat <- diag(d)

  if (p > 0) {
    for (i in 1:p) {
      amat <- amat - a[[i]]
    }
  }

  intercept <- as.numeric(amat %*% mu)
  names(intercept) <- paste0("c", 1:d)

  return(intercept)
}

pack_varma_par <- function(intercept, a, b) {
  par <- as.numeric(intercept)

  if (length(a) > 0) {
    for (i in 1:length(a)) {
      par <- c(par, as.numeric(t(a[[i]])))
    }
  }

  if (length(b) > 0) {
    for (j in 1:length(b)) {
      par <- c(par, as.numeric(t(b[[j]])))
    }
  }

  return(par)
}

unpack_varma_par <- function(par, d, p, q) {
  idx <- 1

  intercept <- par[idx:(idx + d - 1)]
  idx <- idx + d

  a <- vector("list", p)

  if (p > 0) {
    for (i in 1:p) {
      a[[i]] <- matrix(par[idx:(idx + d * d - 1)], nrow = d, byrow = TRUE)
      idx <- idx + d * d
    }
  }

  b <- vector("list", q)

  if (q > 0) {
    for (j in 1:q) {
      b[[j]] <- matrix(par[idx:(idx + d * d - 1)], nrow = d, byrow = TRUE)
      idx <- idx + d * d
    }
  }

  return(list(intercept = intercept, a = a, b = b))
}

varma_resid <- function(x, par, p, q, start_order) {
  x <- as.matrix(x)
  n <- nrow(x)
  d <- ncol(x)

  obj <- unpack_varma_par(par, d, p, q)

  eps <- matrix(0.0, nrow = n, ncol = d)

  for (t in (start_order + 1):n) {
    mean_t <- obj$intercept

    if (p > 0) {
      for (i in 1:p) {
        mean_t <- mean_t + as.numeric(obj$a[[i]] %*% x[t - i, ])
      }
    }

    if (q > 0) {
      for (j in 1:q) {
        mean_t <- mean_t + as.numeric(obj$b[[j]] %*% eps[t - j, ])
      }
    }

    eps[t, ] <- x[t, ] - mean_t
  }

  return(eps[(start_order + 1):n, , drop = FALSE])
}

varma_neg_loglik_conc <- function(par, x, p, q, start_order, ridge = 1.0e-8) {
  x <- as.matrix(x)
  d <- ncol(x)

  resid <- varma_resid(x, par, p, q, start_order)
  nfit <- nrow(resid)

  sigma <- crossprod(resid) / nfit
  sigma <- sigma + ridge * diag(d)

  r <- try(chol(sigma), silent = TRUE)

  if (inherits(r, "try-error")) {
    return(1.0e100)
  }

  logdet <- 2.0 * sum(log(diag(r)))
  loglik <- -0.5 * nfit * (d * log(2.0 * pi) + logdet + d)

  return(-loglik)
}

initial_varma_par <- function(x, p, q, start_order) {
  x <- as.matrix(x)
  n <- nrow(x)
  d <- ncol(x)
  nfit <- n - start_order

  y <- x[(start_order + 1):n, , drop = FALSE]

  if (p == 0) {
    intercept <- colMeans(y)
    a <- vector("list", 0)
  } else {
    xlag <- matrix(NA_real_, nrow = nfit, ncol = d * p)

    for (i in 1:p) {
      col1 <- (i - 1) * d + 1
      col2 <- i * d
      xlag[, col1:col2] <- x[(start_order + 1 - i):(n - i), , drop = FALSE]
    }

    design <- cbind(1.0, xlag)
    coef_mat <- solve(crossprod(design), crossprod(design, y))

    intercept <- as.numeric(coef_mat[1, ])
    a <- vector("list", p)

    for (i in 1:p) {
      row1 <- 1 + (i - 1) * d + 1
      row2 <- 1 + i * d
      a[[i]] <- t(coef_mat[row1:row2, , drop = FALSE])
    }
  }

  b <- vector("list", q)

  if (q > 0) {
    for (j in 1:q) {
      b[[j]] <- matrix(0.0, nrow = d, ncol = d)
    }
  }

  par <- pack_varma_par(intercept, a, b)
  return(par)
}

fit_varma <- function(x, p, q, start_order = max(p, q),
                      maxit = 1000, ridge = 1.0e-8) {
  x <- as.matrix(x)
  n <- nrow(x)
  d <- ncol(x)

  if (p < 0 || q < 0) {
    stop("p and q must be nonnegative")
  }

  if (n <= start_order) {
    stop("nrow(x) must be greater than start_order")
  }

  par0 <- initial_varma_par(x, p, q, start_order)

  opt <- optim(
    par = par0,
    fn = varma_neg_loglik_conc,
    x = x,
    p = p,
    q = q,
    start_order = start_order,
    ridge = ridge,
    method = "BFGS",
    control = list(maxit = maxit)
  )

  par <- opt$par
  obj <- unpack_varma_par(par, d, p, q)

  resid <- varma_resid(x, par, p, q, start_order)
  nfit <- nrow(resid)
  sigma <- crossprod(resid) / nfit
  rownames(sigma) <- colnames(x)
  colnames(sigma) <- colnames(x)

  r <- chol(sigma + ridge * diag(d))
  logdet <- 2.0 * sum(log(diag(r)))
  loglik <- -0.5 * nfit * (d * log(2.0 * pi) + logdet + d)

  amat <- diag(d)

  if (p > 0) {
    for (i in 1:p) {
      amat <- amat - obj$a[[i]]
    }
  }

  if (abs(det(amat)) < 1.0e-12) {
    mu <- rep(NA_real_, d)
  } else {
    mu <- as.numeric(solve(amat, obj$intercept))
  }

  names(obj$intercept) <- paste0("c", 1:d)
  names(mu) <- paste0("mu", 1:d)

  if (p > 0) {
    for (i in 1:p) {
      rownames(obj$a[[i]]) <- paste0("eq", 1:d)
      colnames(obj$a[[i]]) <- paste0("ar", i, "_x", 1:d)
    }
  }

  if (q > 0) {
    for (j in 1:q) {
      rownames(obj$b[[j]]) <- paste0("eq", 1:d)
      colnames(obj$b[[j]]) <- paste0("ma", j, "_eps", 1:d)
    }
  }

  npar <- d + p * d * d + q * d * d + d * (d + 1) / 2
  aic <- -2.0 * loglik + 2.0 * npar
  bic <- -2.0 * loglik + log(nfit) * npar

  out <- list(
    p = p,
    q = q,
    intercept = obj$intercept,
    mu = mu,
    a = obj$a,
    b = obj$b,
    sigma = sigma,
    resid = resid,
    par = par,
    loglik = loglik,
    aic = aic,
    bic = bic,
    npar = npar,
    nfit = nfit,
    convergence = opt$convergence,
    value = opt$value
  )

  return(out)
}

fit_varma_orders <- function(x, p_max, q_max, maxit = 1000) {
  fits <- vector("list", (p_max + 1) * (q_max + 1))
  tab <- matrix(NA_real_, nrow = length(fits), ncol = 7)

  colnames(tab) <- c("p", "q", "loglik", "aic", "bic", "npar", "convergence")

  start_order <- max(p_max, q_max)

  idx <- 1

  for (p in 0:p_max) {
    for (q in 0:q_max) {
      fit <- fit_varma(x, p, q, start_order = start_order, maxit = maxit)

      fits[[idx]] <- fit

      tab[idx, ] <- c(
        fit$p,
        fit$q,
        fit$loglik,
        fit$aic,
        fit$bic,
        fit$npar,
        fit$convergence
      )

      idx <- idx + 1
    }
  }

  aic_idx <- which.min(tab[, "aic"])
  bic_idx <- which.min(tab[, "bic"])

  out <- list(
    fits = fits,
    table = tab,
    aic_p = tab[aic_idx, "p"],
    aic_q = tab[aic_idx, "q"],
    bic_p = tab[bic_idx, "p"],
    bic_q = tab[bic_idx, "q"]
  )

  return(out)
}

print_varma_fit <- function(fit) {
  cat("intercept:\n")
  print(fit$intercept)

  cat("\nmean:\n")
  print(fit$mu)

  if (fit$p > 0) {
    for (i in 1:fit$p) {
      cat("\na[[", i, "]]:\n", sep = "")
      print(fit$a[[i]])
    }
  }

  if (fit$q > 0) {
    for (j in 1:fit$q) {
      cat("\nb[[", j, "]]:\n", sep = "")
      print(fit$b[[j]])
    }
  }

  cat("\ninnovation covariance matrix:\n")
  print(fit$sigma)

  cat("\nloglik:", fit$loglik, "\n")
  cat("aic   :", fit$aic, "\n")
  cat("bic   :", fit$bic, "\n")
  cat("optim convergence code:", fit$convergence, "\n")

  return(invisible(NULL))
}

print_varma_differences <- function(fit, mu_true, a_true, b_true, sigma_true) {
  intercept_true <- intercept_from_mu_a(mu_true, a_true)

  cat("intercept difference, fitted - true:\n")
  print(fit$intercept - intercept_true)

  cat("\nmean difference, fitted - true:\n")
  print(fit$mu - mu_true)

  if (length(a_true) > 0) {
    for (i in 1:length(a_true)) {
      cat("\na[[", i, "]] difference, fitted - true:\n", sep = "")
      print(fit$a[[i]] - a_true[[i]])
    }
  }

  if (length(b_true) > 0) {
    for (j in 1:length(b_true)) {
      cat("\nb[[", j, "]] difference, fitted - true:\n", sep = "")
      print(fit$b[[j]] - b_true[[j]])
    }
  }

  cat("\ninnovation covariance difference, fitted - true:\n")
  print(fit$sigma - sigma_true)

  return(invisible(NULL))
}

set.seed(123)

n <- 300

mu_true <- c(1.0, -2.0)

a_true <- list(
  matrix(c(0.50,  0.20,
          -0.10,  0.40), nrow = 2, byrow = TRUE),

  matrix(c(0.10, -0.05,
           0.05,  0.10), nrow = 2, byrow = TRUE)
)

b_true <- list(
  matrix(c(0.30,  0.10,
           0.05, -0.20), nrow = 2, byrow = TRUE)
)

sigma_true <- matrix(c(1.0, 0.4,
                       0.4, 2.0), nrow = 2, byrow = TRUE)

p_true <- length(a_true)
q_true <- length(b_true)

extra_p_lags <- 0
extra_q_lags <- 0

p_max <- p_true + extra_p_lags
q_max <- q_true + extra_q_lags

sim <- simulate_varma(n, mu_true, a_true, b_true, sigma_true)
x <- sim$x

fit_true_order <- fit_varma(x, p_true, q_true, start_order = max(p_max, q_max))

cat("true p:", p_true, "\n")
cat("true q:", q_true, "\n")
cat("extra p lags fit:", extra_p_lags, "\n")
cat("extra q lags fit:", extra_q_lags, "\n")
cat("max p fit:", p_max, "\n")
cat("max q fit:", q_max, "\n\n")

cat("true mean:\n")
print(mu_true)

if (p_true > 0) {
  for (i in 1:p_true) {
    cat("\ntrue a[[", i, "]]:\n", sep = "")
    print(a_true[[i]])
  }
}

if (q_true > 0) {
  for (j in 1:q_true) {
    cat("\ntrue b[[", j, "]]:\n", sep = "")
    print(b_true[[j]])
  }
}

cat("\ntrue innovation covariance matrix:\n")
print(sigma_true)

cat("\n\nfitted parameters at true order:\n")
print_varma_fit(fit_true_order)

cat("\n\ndifferences between fitted and true parameters at true order:\n")
print_varma_differences(fit_true_order, mu_true, a_true, b_true, sigma_true)

order_select <- fit_varma_orders(x, p_max, q_max, maxit = 1000)

cat("\n\norder selection table:\n")
print(round(order_select$table, 4))

cat("\ntrue p:", p_true, "\n")
cat("true q:", q_true, "\n")
cat("extra p lags fit:", extra_p_lags, "\n")
cat("extra q lags fit:", extra_q_lags, "\n")
cat("max p fit:", p_max, "\n")
cat("max q fit:", q_max, "\n")
cat("order chosen by AIC: p =", order_select$aic_p, "q =", order_select$aic_q, "\n")
cat("order chosen by BIC: p =", order_select$bic_p, "q =", order_select$bic_q, "\n")
