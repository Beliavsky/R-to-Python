# test108_drop_false_dimension.R
x <- matrix(1:6, nrow = 2, ncol = 3)

print(x[1, ])
print(x[1, , drop = FALSE])
print(x[, 1])
print(x[, 1, drop = FALSE])
