# test120_stop_warning_message.R
f <- function(x) {
  if (x < 0) {
    stop("negative x")
  }

  if (x == 0) {
    warning("x is zero")
  }

  message("computing square")
  return(x^2)
}

print(f(2))
print(f(0))
