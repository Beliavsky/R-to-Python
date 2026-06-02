# test65_environment_assignment.R
counter <- function() {
  x <- 0

  f <- function() {
    x <<- x + 1
    return(x)
  }

  return(f)
}

c1 <- counter()

print(c1())
print(c1())
print(c1())
