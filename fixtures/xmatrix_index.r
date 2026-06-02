# test14_matrix_indexing.R
x <- matrix(1:12, nrow = 3, ncol = 4)

print(x[1, 1])
print(x[2, ])
print(x[, 3])
print(x[1:2, 2:4])
print(x[-1, ])
print(x[, -2])

x[1, 1] <- 99
x[2, ] <- c(10, 20, 30, 40)
print(x)
