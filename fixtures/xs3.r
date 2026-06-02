# test92_class_s3_simple.R
make_point <- function(x, y) {
  p <- list(x = x, y = y)
  class(p) <- "point"
  return(p)
}

print.point <- function(p) {
  cat("point(", p$x, ",", p$y, ")\n", sep = "")
}

norm.point <- function(p) {
  return(sqrt(p$x^2 + p$y^2))
}

p <- make_point(3, 4)

print(p)
print(norm.point(p))
