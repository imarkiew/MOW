# ustawienie domyślnej ścieżki do pliku
path = "/home/igor/Pulpit/GitHub/MOW"
setwd(path)

# ładowanie bibliotek
library(recommenderlab)
library(caret)
library(dismo)
library(caTools)
library(Matrix)
library(purrr)
library(dplyr)

# minimalna liczba ocen jaką musi mieć użytkownik żeby był uwzględniony w danych
min_nr_of_ratings <- 5

# liczba użytkowników przeznaczonych do badań
nr_of_users_to_choose <- 1000

# ustawienie ziarna generatora liczb pseudolosowych
seed_ <- 1648
set.seed(seed_)

# wczytanie danych
ratings_raw <- read.csv(paste(path, "BX-CSV-Dump/BX-Book-Ratings.csv", sep = "/"), sep = ";", header = TRUE)

# usunięcie rekordów zawierających oceny (binarne) zebrane w sposób pośredni
ratings_raw <- ratings_raw[(ratings_raw$Book.Rating != 0), ]

# filtracja rekordów w których użytkownicy mają ocenione co najmniej min_nr_of_ratings książek
unique_user_ids <- as.data.frame(table(ratings_raw$User.ID)) %>% filter(as.numeric(Freq) >= min_nr_of_ratings)
unique_user_ids <- unique_user_ids[, 1]
unique_user_ids_length <- length(unique_user_ids) # unique_user_ids zawiera wartości unikalne

# losowanie bez zwracania nr_of_users_to_choose użytkowników
samples <- sample(1:unique_user_ids_length, nr_of_users_to_choose, replace = FALSE) # samples zawiera indeksy
unique_choosed_user_ids <- unique_user_ids[samples] # choosed_user_ids zawiera wybrane wartości unikalne ID użytkowników
ratings <- ratings_raw[ratings_raw$User.ID %in% unique_choosed_user_ids, ]

# przypisanie nowych ID użytkowników i książek do ratings
ratings$New.User.ID <- group_indices(ratings, User.ID)
ratings$New.Book.ID <- group_indices(ratings, ISBN)

# utworzenie macierzy żadkiej
ratings_sparse = sparseMatrix(as.integer(ratings$New.User.ID), as.integer(ratings$New.Book.ID), x = ratings$Book.Rating)
colnames(ratings_sparse) = levels(factor(ratings$ISBN))
rownames(ratings_sparse) = levels(factor(ratings$User.ID))

# utworzenie obiektu realRatingMatrix
ratings_object <- new("realRatingMatrix", data = ratings_sparse)

# rozkład rozkład ocen w zależności on ID użytkownika i ISBN książki
image(ratings_object, main = "Rozkład ocen")

# histogram liczba ocen - liczba użytkowników którzy wystawili taką liczbę ocen
row_counts <- rowCounts(ratings_object)
hist(row_counts, 
          main = "Histogram liczba ocen - liczba użytkwoników",
          col = "darkslategray4", 
          border = "red", 
          xlab = "liczba ocen",
          ylab = "liczba użytkowników",
          breaks = seq(min_nr_of_ratings - 1, max(row_counts) + 1, by = 10)
)

# histogram liczba ocen - liczba książek, które mają wystawioną taką liczbę ocen
column_counts <- colCounts(ratings_object)
hist(column_counts, 
          main = "Histogram liczba ocen - liczba książek",
          col = "darkslategray4", 
          border = "red", 
          xlab = "liczba ocen",
          ylab = "liczba książek",
          breaks = seq(1, max(column_counts) + 1, by = 1)
)

# histogram rozkładu ocen
histogram(getRatings(ratings_object), 
          type = "percent",
          main = paste("Histogram ocen - liczba ocen: ", length(getRatings(ratings_object)), ""), 
          col = "darkslategray4", 
          border = "red", 
          xlab = "ocena", 
          ylab = "% całości", 
          xlim = c(0.5, 10.5), 
          ylim = c(0, 30), 
          breaks = -0.5:(max(getRatings(ratings_object) + 1) - 0.5), 
          scales = list(x = list(at = 1:10)), 
          panel = function(...){
  panel.grid(h = 5, v = 10)
  panel.histogram(...)
})

# podstawowe statystyki
summary(getRatings(ratings_object))

# normalizacja danych - "center", "Z-score"
normalized_ratings_object <- normalize(ratings_object, method = "center", row = TRUE)
r <- getRatings(normalized_ratings_object)
x <- seq(min(r), max(r), length = 50) 
y <- 100*dnorm(x, mean = mean(r), sd = sd(r))

# histogram znormalizowanych ocen
histogram(r,
          type = "percent",
          main = "Znormalizowany histogram ocen", 
          col = "darkslategray4", 
          border = "red", 
          xlab = "ocena przeskalowana", 
          ylab = "% całości",
          scales = list(x = list(at = floor(min(r)):ceiling(max(r)))), 
          breaks = 10,
          ylim = c(0, 35),
          key = list(
          corner = c(0, 0.95), 
          lines = list(col = c("darkslategray4", "green"), lty = 1, lwd = 2), 
          text = list(c("histogram", "rozkład normalny"))), 
          panel = function(...){
  panel.grid(h = 5, v = 10)
  panel.histogram(...)
  panel.lines(x, y, col = "green", lwd = 2)
})

# i tak normalizujemy dla każdego użytkownika osobno (wierszami), więc możemy podziała na zbiory test i train przesunąć po wspólnej normalizacji
sample <- sample.split(1:nrow(normalized_ratings_object), SplitRatio = 0.8)
train <- subset(normalized_ratings_object, sample == TRUE)
test <- subset(normalized_ratings_object, sample == FALSE)

# schemat ewaluacji
# metody - "UBCF", "IBCF", "SVDF", "POPULAR", ....
# "IBCF" - bardzo długie obliczenia, wartości NaN, dla większej ilości danych bardzo zasobożerne
# metryki podobieństwa - "Cosine", "pearson"
# method = "split", train = 0.9
# method = "cross", k = 3
scheme <- evaluationScheme(train, method = "cross", k = 5, given = -1)
algorithms <- list(
  "UBCF" = list(name = "UBCF", param = list(method = "Cosine", nn = 25, normalize = NULL)),
  # "IBCF" = list(name = "IBCF", param = list(method = "Cosine", k = 25, normalize = NULL)),
  "SVDF" = list(name = "SVDF", param = list(k = 10, gamma = 0.015, lambda = 0.001,  max_epoc = 1000, normalize = NULL))
)
history <- recommenderlab::evaluate(scheme, algorithms, type = "ratings")
# history$UBCF@results[[1]]@cm[[1]] - RMSE