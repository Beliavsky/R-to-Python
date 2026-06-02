# test74_monte_carlo_pi.R
set.seed(123)

n <- 10000
x <- runif(n, -1, 1)
y <- runif(n, -1, 1)

inside <- x^2 + y^2 <= 1
pi_hat <- 4 * mean(inside)

print(pi_hat)
