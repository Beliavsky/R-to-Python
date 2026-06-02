# test104_simple_em_two_normals.R
set.seed(123)

n <- 200
z <- rbinom(n, 1, 0.4)
x <- ifelse(z == 1, rnorm(n, -2, 1), rnorm(n, 2, 1.5))

pi1 <- 0.5
mu1 <- -1
mu2 <- 1
sd1 <- 1
sd2 <- 1

for (iter in 1:20) {
  d1 <- pi1 * dnorm(x, mu1, sd1)
  d2 <- (1 - pi1) * dnorm(x, mu2, sd2)
  w <- d1 / (d1 + d2)

  pi1 <- mean(w)
  mu1 <- sum(w * x) / sum(w)
  mu2 <- sum((1 - w) * x) / sum(1 - w)
  sd1 <- sqrt(sum(w * (x - mu1)^2) / sum(w))
  sd2 <- sqrt(sum((1 - w) * (x - mu2)^2) / sum(1 - w))
}

print(c(pi1 = pi1, mu1 = mu1, mu2 = mu2, sd1 = sd1, sd2 = sd2))
