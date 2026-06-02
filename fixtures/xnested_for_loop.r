# test117_nested_for_loops.R
x <- matrix(0, nrow = 4, ncol = 5)

for (i in 1:nrow(x)) {
  for (j in 1:ncol(x)) {
    x[i, j] <- i + 10 * j
  }
}

print(x)
