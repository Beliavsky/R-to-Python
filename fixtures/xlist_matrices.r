# test106_list_of_matrices.R
make_mats <- function(n) {
  out <- vector("list", n)

  for (i in 1:n) {
    out[[i]] <- matrix(i, nrow = 2, ncol = 2)
  }

  return(out)
}

mats <- make_mats(4)

print(mats)
print(Reduce("+", mats))
