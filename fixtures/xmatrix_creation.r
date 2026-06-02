# test13_matrix_creation.R
x <- matrix(1:12, nrow = 3, ncol = 4)
print(x)

y <- matrix(1:12, nrow = 3, ncol = 4, byrow = TRUE)
print(y)

print(dim(x))
print(nrow(x))
print(ncol(x))
print(length(x))
