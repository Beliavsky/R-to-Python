# test44_markov_chain.R
set.seed(123)

n <- 50
P <- matrix(c(0.9, 0.1,
              0.2, 0.8), nrow = 2, byrow = TRUE)

state <- integer(n)
state[1] <- 1

for (i in 2:n) {
  u <- runif(1)
  if (u <= P[state[i - 1], 1]) {
    state[i] <- 1
  } else {
    state[i] <- 2
  }
}

print(state)
print(table(state))
