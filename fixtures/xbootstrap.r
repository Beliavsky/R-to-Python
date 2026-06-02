# test73_simple_bootstrap.R
set.seed(123)

x <- rnorm(20)
B <- 100
boot_mean <- numeric(B)

for (b in 1:B) {
  xb <- sample(x, size = length(x), replace = TRUE)
  boot_mean[b] <- mean(xb)
}

print(mean(x))
print(mean(boot_mean))
print(sd(boot_mean))
print(quantile(boot_mean, c(0.025, 0.975)))
