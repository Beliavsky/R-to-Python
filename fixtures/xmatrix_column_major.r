# test112_matrix_column_major_order.R
x <- matrix(1:12, nrow = 3, ncol = 4)

print(x)
print(as.vector(x))
print(matrix(as.vector(x), nrow = 3, ncol = 4))
