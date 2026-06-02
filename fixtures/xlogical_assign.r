# test67_matrix_logical_assignment.R
x <- matrix(1:12, nrow = 3, ncol = 4)

x[x %% 2 == 0] <- 0

print(x)
