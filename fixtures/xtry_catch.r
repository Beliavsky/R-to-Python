# test54_try_catch.R
safe_log <- function(x) {
  ans <- tryCatch(
    {
      if (x <= 0) {
        stop("x must be positive")
      }
      log(x)
    },
    error = function(e) {
      NA
    }
  )

  return(ans)
}

print(safe_log(10))
print(safe_log(-1))
