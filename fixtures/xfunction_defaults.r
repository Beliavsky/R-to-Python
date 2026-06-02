# test11_functions_defaults.R
scale_shift <- function(x, a = 1, b = 0) {
  y <- a * x + b
  return(y)
}

x <- c(1, 2, 3)

print(scale_shift(x))
print(scale_shift(x, a = 10))
print(scale_shift(x, b = 5))
print(scale_shift(x, a = 10, b = 5))
