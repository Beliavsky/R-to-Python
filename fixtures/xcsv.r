# test55_file_io_csv.R
df <- data.frame(
  x = 1:5,
  y = c(10, 20, 30, 40, 50)
)

file <- tempfile(fileext = ".csv")

write.csv(df, file = file, row.names = FALSE)
df2 <- read.csv(file)

print(df2)

unlink(file)
