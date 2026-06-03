# tibble_demo.r
# Demonstrate common tibble features.

library(tibble)

# create a tibble
df <- tibble(
  name = c("ann", "bob", "cara", "dan"),
  age = c(21, 25, 23, 28),
  score = c(88.5, 91.0, 79.5, 95.0),
  passed = score >= 80
)

print(df)

# tibbles print compactly and show column types
cat("\nstructure:\n")
str(df)

# create a tibble row by row
df2 <- tribble(
  ~name, ~age, ~score,
  "erin", 22, 84.0,
  "fred", 24, 76.5,
  "gina", 27, 90.5
)

cat("\ntribble result:\n")
print(df2)

# add a row
df3 <- add_row(df, name = "hank", age = 30, score = 82.0, passed = TRUE)

cat("\nafter add_row:\n")
print(df3)

# add a column
df4 <- add_column(df3, group = c("a", "a", "b", "b", "a"), .after = "name")

cat("\nafter add_column:\n")
print(df4)

# subset columns
cat("\nselecting one column with $:\n")
print(df4$name)

cat("\nselecting one column as a tibble:\n")
print(df4["name"])

cat("\nselecting rows and columns:\n")
print(df4[1:3, c("name", "score")])

# tibbles do not simplify as aggressively as data frames
cat("\nclass of df4['score']:\n")
print(class(df4["score"]))

cat("\nclass of df4$score:\n")
print(class(df4$score))

# convert between data.frame and tibble
base_df <- data.frame(x = 1:3, y = c("a", "b", "c"))
tbl <- as_tibble(base_df)

cat("\nconverted data.frame to tibble:\n")
print(tbl)

cat("\nconverted tibble back to data.frame:\n")
print(as.data.frame(tbl))

# column names can be nonstandard if backticks are used
weird <- tibble(
  `first name` = c("ann", "bob"),
  `test score` = c(88, 91)
)

cat("\ntibble with nonstandard column names:\n")
print(weird)

cat("\naccess nonstandard name:\n")
print(weird$`first name`)

# create a one-row tibble
one <- tibble_row(name = "ivy", age = 26, score = 89.5)

cat("\none-row tibble:\n")
print(one)

# list columns are allowed
nested <- tibble(
  id = 1:3,
  values = list(1:3, c(10, 20), numeric(0))
)

cat("\ntibble with a list column:\n")
print(nested)

cat("\nsecond list-column element:\n")
print(nested$values[[2]])
