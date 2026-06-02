# test19_lists.R
x <- list(a = 1:5, b = "hello", c = matrix(1:4, 2, 2))

print(x)
print(x$a)
print(x[["b"]])
print(x[[3]])

x$d <- TRUE
print(names(x))
print(length(x))
print(x)
