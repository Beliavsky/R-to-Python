# test53_cholesky.R
A <- matrix(c(4, 2,
              2, 3), nrow = 2, byrow = TRUE)

R <- chol(A)

print(R)
print(t(R) %*% R)
