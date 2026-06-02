# test113_append_replace_length.R
x <- c(1, 2, 3)

print(append(x, 99))
print(append(x, 99, after = 1))

length(x) <- 5
print(x)

x[is.na(x)] <- 0
print(x)
