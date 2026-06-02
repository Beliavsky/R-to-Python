# test63_recursive_function.R
factorial2 <- function(n) {
  if (n <= 1) {
    return(1)
  } else {
    return(n * factorial2(n - 1))
  }
}

print(factorial2(5))
print(sapply(1:6, factorial2))
