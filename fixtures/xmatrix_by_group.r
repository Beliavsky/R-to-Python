# test100_matrix_by_group_manual.R
set.seed(123)

n <- 12
g <- rep(1:3, each = 4)
x <- matrix(rnorm(n * 2), nrow = n, ncol = 2)

out <- matrix(0, nrow = 3, ncol = 2)

for (j in 1:3) {
  out[j, ] <- colMeans(x[g == j, , drop = FALSE])
}

print(x)
print(g)
print(out)
