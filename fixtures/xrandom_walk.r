# test36_simulate_random_walk.R
set.seed(123)

n <- 20
eps <- rnorm(n)
x <- numeric(n)

x[1] <- eps[1]
for (i in 2:n) {
  x[i] <- x[i - 1] + eps[i]
}

print(x)
print(mean(x))
print(sd(x))
