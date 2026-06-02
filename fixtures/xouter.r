# test61_outer.R
x <- 1:4
y <- 1:3

print(outer(x, y, "+"))
print(outer(x, y, "*"))

f <- function(a, b) {
  return(a^2 + b^2)
}

print(outer(x, y, f))
