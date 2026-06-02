# test98_raw_vectors.R
x <- charToRaw("hello")
cat("\nx:\n")
print(x)
print(rawToChar(x))
print(as.integer(x))
