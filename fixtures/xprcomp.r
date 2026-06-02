# test77_prcomp.R
set.seed(123)

x <- matrix(rnorm(100), nrow = 25, ncol = 4)
x[, 4] <- x[, 1] + x[, 2] + rnorm(25, sd = 0.1)

fit <- prcomp(x, center = TRUE, scale. = TRUE)

print(fit$sdev)
print(fit$rotation)
print(head(fit$x))
