# test16_matrix_multiplication.R
a <- matrix(1:6, nrow = 2, ncol = 3)
b <- matrix(1:6, nrow = 3, ncol = 2)

print(a)
print(b)
print(a %*% b)
print(crossprod(a))
print(tcrossprod(a))
