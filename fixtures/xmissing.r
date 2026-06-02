# test25_missing_values.R
x <- c(1, NA, 3, NA, 5)

print(is.na(x))
print(!is.na(x))
print(sum(is.na(x)))
print(mean(x))
print(mean(x, na.rm = TRUE))
print(sum(x, na.rm = TRUE))

x[is.na(x)] <- 0
print(x)
