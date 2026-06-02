# test90_scan_text_connection.R
txt <- "1 2 3 4 5"
con <- textConnection(txt)

x <- scan(con, quiet = TRUE)
close(con)

print(x)
print(sum(x))
