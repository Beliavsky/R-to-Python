# test42_arbitrary_order_ar.R
set.seed(123)

simulate_ar <- function(phi, n, sigma) {
  p <- length(phi)
  x <- numeric(n)
  e <- rnorm(n, sd = sigma)

  for (i in 1:n) {
    x[i] <- e[i]
    for (j in 1:p) {
      if (i - j >= 1) {
        x[i] <- x[i] + phi[j] * x[i - j]
      }
    }
  }

  return(x)
}

x <- simulate_ar(c(0.7, -0.2), 100000, 1.0)
print(head(x))
print(mean(x))
print(sd(x))
