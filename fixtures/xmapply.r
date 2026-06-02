# test30_mapply.R
x <- c(1, 2, 3)
y <- c(10, 20, 30)

f <- function(a, b) {
  return(a + 2 * b)
}

print(mapply(f, x, y))
