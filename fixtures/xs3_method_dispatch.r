# test93_s3_method_dispatch.R
area <- function(x) {
  UseMethod("area")
}

area.circle <- function(x) {
  return(pi * x$radius^2)
}

area.rectangle <- function(x) {
  return(x$width * x$height)
}

circle <- list(radius = 2)
class(circle) <- "circle"

rectangle <- list(width = 3, height = 4)
class(rectangle) <- "rectangle"

print(area(circle))
print(area(rectangle))
