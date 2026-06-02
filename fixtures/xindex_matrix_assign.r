# test118_index_matrix_assignment.R
x <- matrix(0, nrow = 4, ncol = 4)

idx <- cbind(c(1, 2, 3), c(4, 3, 2))
x[idx] <- c(10, 20, 30)

print(x)
print(x[idx])