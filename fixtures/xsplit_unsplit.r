# test31_split_unsplit.R
x <- c(1, 2, 10, 20, 100, 200)
g <- c("a", "a", "b", "b", "c", "c")

sx <- split(x, g)
print(sx)

means <- sapply(sx, mean)
print(means)
