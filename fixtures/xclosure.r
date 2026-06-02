# test64_closure.R
make_adder <- function(a) {
  f <- function(x) {
    return(x + a)
  }
  return(f)
}

add10 <- make_adder(10)

print(add10(1))
print(add10(c(1, 2, 3)))
