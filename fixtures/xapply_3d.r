# test18_arrays.R
x <- array(1:24, dim = c(2, 3, 4))

print(x)
print(dim(x))
print(x[1, 2, 3])
print(x[, , 1])
print(x[1, , ])
print(sum(x))
print(apply(x, 3, sum))
