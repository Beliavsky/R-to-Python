# test45_markov_switching_mean.R
set.seed(123)

n <- 100
P <- matrix(c(0.95, 0.05,
              0.10, 0.90), nrow = 2, byrow = TRUE)
mu <- c(-1, 1)
sigma <- c(0.5, 2.0)

state <- integer(n)
y <- numeric(n)

state[1] <- 1
y[1] <- rnorm(1, mean = mu[state[1]], sd = sigma[state[1]])

 for (i in 2:n) {
  u <- runif(1)
  old <- state[i - 1]

  if (u <= P[old, 1]) {
    state[i] <- 1
  } else {
    state[i] <- 2
  }

  y[i] <- rnorm(1, mean = mu[state[i]], sd = sigma[state[i]])
}

print(head(data.frame(state = state, y = y), 20))
print(tapply(y, state, mean))
print(tapply(y, state, sd))
