# test68_diag_lower_upper.R
x <- matrix(1:16, nrow = 4, ncol = 4)

print(diag(x))
print(lower.tri(x))
print(upper.tri(x))

x[lower.tri(x)] <- 0
print(x)

diag(x) <- 99
print(x)
