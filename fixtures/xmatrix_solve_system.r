# test115_matrix_solve_system.R
A <- matrix(c(3, 1,
              1, 2), nrow = 2, byrow = TRUE)
b <- c(9, 8)

x <- solve(A, b)

print(x)
print(A %*% x)