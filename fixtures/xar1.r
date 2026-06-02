# test41_ar1_simulation.R
set.seed(123)

n <- 100
phi <- 0.8
sigma <- 1.0

x <- numeric(n)
e <- rnorm(n, sd = sigma)

x[1] <- e[1]
for (i in 2:n) {
  x[i] <- phi * x[i - 1] + e[i]
}

print(head(x))
print(mean(x))
print(sd(x))
print(acf(x, plot = FALSE)$acf[1:5])
