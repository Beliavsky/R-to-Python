# test103_manual_kalman_filter.R
set.seed(123)

n <- 50
q <- 0.1
r <- 0.5

x_true <- numeric(n)
y <- numeric(n)

x_true[1] <- rnorm(1)
y[1] <- x_true[1] + rnorm(1, sd = sqrt(r))

for (i in 2:n) {
  x_true[i] <- x_true[i - 1] + rnorm(1, sd = sqrt(q))
  y[i] <- x_true[i] + rnorm(1, sd = sqrt(r))
}

x_pred <- 0
p_pred <- 1

x_filt <- numeric(n)
p_filt <- numeric(n)

for (i in 1:n) {
  k <- p_pred / (p_pred + r)
  x_filt[i] <- x_pred + k * (y[i] - x_pred)
  p_filt[i] <- (1 - k) * p_pred

  x_pred <- x_filt[i]
  p_pred <- p_filt[i] + q
}

print(head(cbind(x_true, y, x_filt), 3))
