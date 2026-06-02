# test89_save_load_rds.R
x <- list(a = 1:5, b = matrix(1:4, 2, 2))

file <- tempfile(fileext = ".rds")

saveRDS(x, file)
y <- readRDS(file)

print(y)

unlink(file)
