# test110_inf_nan.R
x <- c(1, 0, -1)

y <- x / 0
z <- sqrt(-1)

print(y)
print(z)
print(is.infinite(y))
print(is.nan(z))
print(is.finite(c(y, z, 1)))
