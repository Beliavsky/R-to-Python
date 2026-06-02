# test52_eigen_svd_qr.R
A <- matrix(c(2, 1,
              1, 2), nrow = 2, byrow = TRUE)

print(eigen(A))
print(svd(A))
print(qr(A))
print(det(A))
print(solve(A))
