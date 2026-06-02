# test10_functions_basic.R
square <- function(x) {
  return(x^2)
}

center <- function(x) {
  y <- x - mean(x)
  return(y)
}

x <- c(1, 2, 3, 4, 5)

print(square(x))
print(center(x))
