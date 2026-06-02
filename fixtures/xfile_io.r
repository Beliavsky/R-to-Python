# test56_file_io_text.R
x <- c("alpha", "beta", "gamma")

file <- tempfile()
writeLines(x, con = file)

y <- readLines(file)
print(y)

unlink(file)
