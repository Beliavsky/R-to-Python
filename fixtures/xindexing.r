# test109_negative_zero_one_based_indexing.R
x <- c(10, 20, 30, 40, 50)

print(x[1])
print(x[0])
print(x[-1])
print(x[c(1, 3, 5)])
cat("\nhere\n")
print(x[c(TRUE, FALSE, TRUE, FALSE, TRUE)])
