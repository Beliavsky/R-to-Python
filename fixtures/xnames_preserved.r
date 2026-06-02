# test111_names_preserved.R
x <- c(a = 1, b = 2, c = 3)
y <- x^2

print(y)
print(names(y))

z <- x[c("c", "a")]
print(z)
