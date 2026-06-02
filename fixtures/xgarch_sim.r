# test105_simple_garch_simulation.R
set.seed(123)

n <- 500
omega <- 0.1
alpha <- 0.1
beta <- 0.8

e <- numeric(n)
h <- numeric(n)
z <- rnorm(n)

h[1] <- omega / (1 - alpha - beta)
e[1] <- sqrt(h[1]) * z[1]

for (i in 2:n) {
  h[i] <- omega + alpha * e[i - 1]^2 + beta * h[i - 1]
  e[i] <- sqrt(h[i]) * z[i]
}

print(head(cbind(e, h), 10))
print(mean(e))
print(var(e))
print(mean(h))
