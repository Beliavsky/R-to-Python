# test03_logical_indexing.R
x <- c(3, -1, 0, 7, -5, 2)

print(x > 0)
print(x[x > 0])
print(x[x <= 0])
x[x < 0] <- 0
print(x)

y <- c(10, 20, 30, 40, 50, 60)
print(y[x == 0])
