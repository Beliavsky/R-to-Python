# test82_dots_arguments.R
f <- function(...) {
  x <- list(...)
  print(x)
  return(length(x))
}

print(f(1, 2, a = 3, b = "hello"))
