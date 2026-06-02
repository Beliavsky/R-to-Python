# test34_dates.R
d <- as.Date(c("2024-01-01", "2024-01-05", "2024-02-01"))

print(d)
print(d + 1)
print(diff(d))
print(format(d, "%Y"))
print(format(d, "%m"))
print(format(d, "%d"))

df <- data.frame(date = d, x = c(10, 20, 30))
print(df)
