# test15_matrix_arithmetic.R
a <- matrix(1:6, nrow = 2, ncol = 3)
b <- matrix(10:15, nrow = 2, ncol = 3)

print(a + b)
print(b - a)
print(a * b)
print(b / a)
print(a + 100)
print(t(a))
