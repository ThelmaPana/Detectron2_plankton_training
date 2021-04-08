library(googlesheets4)

gs4_auth(
  use_oob = TRUE,
  cache = ".secrets",
  email = "<my_address>@gmail.com"
)
