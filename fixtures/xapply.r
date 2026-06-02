# test17_apply.R
x <- matrix(1:12, nrow = 3, ncol = 4)

print(apply(x, 1, sum))
print(apply(x, 2, sum))
print(apply(x, 1, mean))
print(apply(x, 2, mean))
print(apply(x, 2, median))
print(apply(x, 2, var))
print(rowSums(x))
print(colSums(x))
print(rowMeans(x))
print(colMeans(x))
