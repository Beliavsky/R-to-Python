# var_sim_fit_order_select_base_r_no_data_frame.R
# Simulate and fit arbitrary-order vector autoregressions using base R, without data frames.

rmvnorm_chol <- function(n, mu, sigma) {
  p <- length(mu)
  z <- matrix(rnorm(n * p), nrow = n, ncol = p)
  y <- sweep(z %*% chol(sigma), 2, mu, "+")
  return(y)
}

simulate_var <- function(n, mu, a, sigma, x0 = NULL) {
  p <- length(mu)
  order <- length(a)

  if (n <= order) {
    stop("n must be greater than length(a)")
  }

  for (j in 1:order) {
    if (!all(dim(a[[j]]) == c(p, p))) {
      stop("each element of a must be a p by p matrix")
    }
  }

  if (!all(dim(sigma) == c(p, p))) {
    stop("sigma must be a p by p matrix")
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

make_var_design <- function(x, order, start_order = order) {
  x <- as.matrix(x)
  n <- nrow(x)
  p <- ncol(x)

  if (order < 0) {
    stop("order must be nonnegative")
  }

  if (start_order < order) {
    stop("start_order must be at least order")
  }

  if (n <= start_order) {
    stop("nrow(x) must be greater than start_order")
  }

  y <- x[(start_order + 1):n, , drop = FALSE]
  nfit <- n - start_order

  if (order == 0) {
    design <- matrix(1.0, nrow = nfit, ncol = 1)
  } else {
    xlag <- matrix(NA_real_, nrow = nfit, ncol = p * order)

    for (j in 1:order) {
      col1 <- (j - 1) * p + 1
      col2 <- j * p
      xlag[, col1:col2] <- x[(start_order + 1 - j):(n - j), , drop = FALSE]
    }

    design <- cbind(1.0, xlag)
  }

  out <- list(y = y, design = design)
  return(out)
}

fit_var <- function(x, order, start_order = order) {
  x <- as.matrix(x)
  p <- ncol(x)

  d <- make_var_design(x, order, start_order)

  y <- d$y
  design <- d$design

  coef_mat <- solve(crossprod(design), crossprod(design, y))

  intercept <- as.numeric(coef_mat[1, ])
  names(intercept) <- paste0("c", 1:p)

  a <- vector("list", order)

  if (order > 0) {
    for (j in 1:order) {
      row1 <- 1 + (j - 1) * p + 1
      row2 <- 1 + j * p

      a[[j]] <- t(coef_mat[row1:row2, , drop = FALSE])
      rownames(a[[j]]) <- paste0("eq", 1:p)
      colnames(a[[j]]) <- paste0("lag", j, "_x", 1:p)
    }
  }

  fitted <- design %*% coef_mat
  resid <- y - fitted

  nfit <- nrow(resid)
  sigma <- crossprod(resid) / nfit
  rownames(sigma) <- colnames(x)
  colnames(sigma) <- colnames(x)

  if (order == 0) {
    mu <- intercept
  } else {
    amat <- diag(p)

    for (j in 1:order) {
      amat <- amat - a[[j]]
    }

    if (abs(det(amat)) < 1.0e-12) {
      mu <- rep(NA_real_, p)
    } else {
      mu <- as.numeric(solve(amat, intercept))
    }
  }

  names(mu) <- paste0("mu", 1:p)

  logdet <- 2.0 * sum(log(diag(chol(sigma))))
  loglik <- -0.5 * nfit * (p * log(2.0 * pi) + logdet + p)

  npar <- p + order * p * p + p * (p + 1) / 2
  aic <- -2.0 * loglik + 2.0 * npar
  bic <- -2.0 * loglik + log(nfit) * npar

  out <- list(
    order = order,
    intercept = intercept,
    mu = mu,
    a = a,
    sigma = sigma,
    fitted = fitted,
    resid = resid,
    coef = coef_mat,
    loglik = loglik,
    aic = aic,
    bic = bic,
    npar = npar,
    nfit = nfit
  )

  return(out)
}

intercept_from_mu_a <- function(mu, a) {
  p <- length(mu)
  order <- length(a)

  amat <- diag(p)

  if (order > 0) {
    for (j in 1:order) {
      amat <- amat - a[[j]]
    }
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

  if (order > 0) {
    for (j in 1:order) {
      cat("\na[[", j, "]]:\n", sep = "")
      print(fit$a[[j]])
    }
  }

  cat("\ninnovation covariance matrix:\n")
  print(fit$sigma)

  cat("\nloglik:", fit$loglik, "\n")
  cat("aic   :", fit$aic, "\n")
  cat("bic   :", fit$bic, "\n")

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

fit_var_orders <- function(x, max_order) {
  if (max_order < 0) {
    stop("max_order must be nonnegative")
  }

  fits <- vector("list", max_order + 1)
  table <- matrix(NA_real_, nrow = max_order + 1, ncol = 5)

  colnames(table) <- c("order", "loglik", "aic", "bic", "npar")

  for (order in 0:max_order) {
    fit <- fit_var(x, order, start_order = max_order)

    fits[[order + 1]] <- fit

    table[order + 1, ] <- c(
      fit$order,
      fit$loglik,
      fit$aic,
      fit$bic,
      fit$npar
    )
  }

  aic_order <- table[which.min(table[, "aic"]), "order"]
  bic_order <- table[which.min(table[, "bic"]), "order"]

  out <- list(
    fits = fits,
    table = table,
    aic_order = aic_order,
    bic_order = bic_order
  )

  return(out)
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

k_true <- length(a_true)
max_order <- k_true + 2

intercept_true <- intercept_from_mu_a(mu_true, a_true)

x <- simulate_var(n, mu_true, a_true, sigma_true)

fit_true_order <- fit_var(x, order = k_true, start_order = max_order)

cat("true order:", k_true, "\n\n")

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

cat("\n\nfitted parameters at true order:\n")
print_var_fit(fit_true_order)

cat("\n\ndifferences between fitted and true parameters at true order:\n")
print_var_differences(fit_true_order, mu_true, a_true, sigma_true)

order_select <- fit_var_orders(x, max_order)

cat("\n\norder selection table:\n")
print(round(order_select$table, 4))

cat("\ntrue, AIC, BIC orders:", k_true, order_select$aic_order, order_select$bic_order)
