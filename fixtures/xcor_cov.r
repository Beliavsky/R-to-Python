# test51_cor_cov.R
set.seed(123)

x <- matrix(rnorm(100), nrow = 25, ncol = 4)
x[, 2] <- x[, 1] + rnorm(25, sd = 0.2)

print(cor(x))
print(cov(x))
