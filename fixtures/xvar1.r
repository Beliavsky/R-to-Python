# test43_var1_simulation.R
set.seed(123)

n <- 10000
k <- 2

a <- c(0.1, -0.2)
B <- matrix(c(0.7, 0.1,
              0.2, 0.5), nrow = k, byrow = TRUE)
cat("\na =",a)
cat("\nB =\n",B,"\n")
x <- matrix(0, nrow = n, ncol = k)
e <- matrix(rnorm(n * k), nrow = n, ncol = k)

x[1, ] <- e[1, ]
for (i in 2:n) {
  x[i, ] <- a + B %*% x[i - 1, ] + e[i, ]
}

print(head(x))
cat("\ncolmeans(x)\n")
print(colMeans(x))
cat("\ncov(x)\n")
print(cov(x))
