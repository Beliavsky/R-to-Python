# test83_do_call.R
f <- function(a, b, c) {
  return(a + 10 * b + 100 * c)
}

args <- list(a = 1, b = 2, c = 3)

print(do.call(f, args))
print(do.call("sum", list(1:5)))
