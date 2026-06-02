# test26_strings.R
s <- c("alpha", "beta", "gamma", "delta")

print(nchar(s))
print(toupper(s))
print(tolower(s))
print(substr(s, 1, 2))
print(paste(s, collapse = ","))
print(paste("id", 1:4, sep = "_"))
print(grepl("a", s))
print(s[grepl("mm", s)])
