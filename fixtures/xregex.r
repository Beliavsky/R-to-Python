# test119_regular_expressions.R
s <- c("abc123", "def45", "xyz", "a9b8")

cat("\ngrep\n")
print(grep("[0-9]+", s))
cat("\ngrepl\n")
print(grepl("[0-9]+", s))
cat("\nsub\n")
print(sub("[0-9]+", "NUM", s))
cat("\ngsub\n")
print(gsub("[0-9]", "#", s))
cat("\nregexpr\n")
print(regexpr("[0-9]+", s))
