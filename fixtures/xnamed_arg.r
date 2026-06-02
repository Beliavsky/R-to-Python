# test80_named_function_arguments.R
f <- function(a, b, c) {
  return(100 * a + 10 * b + c)
}

print(f(1, 2, 3))
print(f(a = 1, b = 2, c = 3))
print(f(c = 3, a = 1, b = 2))
print(f(1, c = 3, b = 2))
