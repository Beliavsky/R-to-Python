# test29_lapply_sapply.R
x <- list(a = 1:3, b = 4:6, c = 7:10)

print(lapply(x, sum))
print(sapply(x, sum))
print(sapply(x, mean))

f <- function(v) {
  return(sum(v^2))
}

print(sapply(x, f))
