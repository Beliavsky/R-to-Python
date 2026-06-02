# test62_vectorized_custom_function.R
f_scalar <- function(x) {
  if (x < 0) {
    return(-x)
  } else {
    return(x^2)
  }
}

f_vec <- Vectorize(f_scalar)

x <- c(-3, -2, -1, 0, 1, 2, 3)
print(f_vec(x))
