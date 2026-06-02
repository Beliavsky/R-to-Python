# test06_if_else.R
x <- 7

if (x > 10) {
  print("large")
} else if (x > 5) {
  print("medium")
} else {
  print("small")
}

y <- ifelse(c(-1, 0, 2, 5) > 0, 1, 0)
print(y)
