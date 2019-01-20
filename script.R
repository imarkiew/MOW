# ustawienie domyslnej sciezki do pliku
path = "/home/igor/Pulpit/GitHub/MOW"
setwd(path)

# ladowanie bibliotek
library(recommenderlab)
library(caret)
library(dismo)
library(caTools)
library(Matrix)
library(purrr)
library(dplyr)

# -----------------------------------------------------------------------------------------------------
# wczytywanie i wstepne przetwarzanie danych
# -----------------------------------------------------------------------------------------------------

# minimalna liczba ocen jaka musi miec użytkownik żeby byl uwzgledniony w danych
min_nr_of_ratings <- 5

# liczba uzytkownikow przeznaczonych do badań
nr_of_users_to_choose <- 500

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

# -----------------------------------------------------------------------------------------------------
# statystyki danych
# -----------------------------------------------------------------------------------------------------

# histogram liczba ocen - liczba użytkowników którzy wystawili taką liczbę ocen
row_counts <- rowCounts(ratings_object)
hist(row_counts, 
     main = "Histogram liczba ocen - liczba użytkwoników",
     col = "darkslategray4", 
     border = "red", 
     xlab = "liczba ocen",
     ylab = "liczba użytkowników",
     breaks = seq(min_nr_of_ratings - 1, max(row_counts) + 1, by = 10),
     xlim = c(min(column_counts) - 0.5, 300)
)

# histogram liczba ocen - liczba książek, które mają wystawioną taką liczbę ocen
column_counts <- colCounts(ratings_object)
hist(column_counts, 
     main = "Histogram liczba ocen - liczba książek",
     col = "darkslategray4", 
     border = "red", 
     xlab = "liczba ocen",
     ylab = "liczba książek",
     breaks = seq(min(column_counts) - 0.5, max(column_counts) + 0.5, by = 1),
     xlim = c(min(column_counts) - 0.5, 10.5),
     xaxt = "n"
)
axis(side = 1, at = seq(1, 10, 1))

# histogram rozkładu ocen
histogram(getRatings(ratings_object), 
          type = "percent",
          main = "Histogram ocen", 
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
normalized_ratings_object_center <- normalize(ratings_object, method = "center", row = TRUE)
r <- getRatings(normalized_ratings_object_center)
x <- seq(min(r), max(r), length = 50) 
y <- 100*dnorm(x, mean = mean(r), sd = sd(r))

histogram(r,
          type = "percent",
          main = "Znormalizowany histogram ocen dla metody center", 
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

normalized_ratings_object_zscore <- normalize(ratings_object, method = "Z-score", row = TRUE)
r <- getRatings(normalized_ratings_object_zscore)
x <- seq(min(r), max(r), length = 50) 
y <- 100*dnorm(x, mean = mean(r), sd = sd(r))

histogram(r,
          type = "percent",
          main = "Znormalizowany histogram ocen dla metody Z-score", 
          col = "darkslategray4", 
          border = "red", 
          xlab = "ocena przeskalowana", 
          ylab = "% całości",
          scales = list(x = list(at = floor(min(r)):ceiling(max(r)))), 
          breaks = 10,
          ylim = c(0, 45),
          key = list(
          corner = c(0, 0.95), 
          lines = list(col = c("darkslategray4", "green"), lty = 1, lwd = 2), 
          text = list(c("histogram", "rozkład normalny"))), 
          panel = function(...){
          panel.grid(h = 5, v = 10)
          panel.histogram(...)
          panel.lines(x, y, col = "green", lwd = 2)
          })

# -----------------------------------------------------------------------------------------------------
# testy parametrow 
# -----------------------------------------------------------------------------------------------------

normalization_types <- c("center", "Z-score")
kfold <- 5
examples_for_test <- 1

# -------------------------------------------------- UBCF ---------------------------------------------

# testowanie normalizacji "center", "Z-score"
ubcf_normzlization_history <- list()
for(type in normalization_types){
  scheme <- evaluationScheme(ratings_object, method = "cross", k = kfold, given = -examples_for_test)
  algorithm <- list("UBCF" = list(name = "UBCF", param = list(normalize = type)))
  ubcf_normzlization_history[type] <- recommenderlab::evaluate(scheme, algorithm, type = "ratings")
}

ubcf_normzlization_rmse_center <- vector()
ubcf_normzlization_rmse_zscore <- vector()
ubcf_normzlization_mae_center <- vector()
ubcf_normzlization_mae_zscore <- vector()

for(i in 1:kfold){
  ubcf_normzlization_rmse_center[i] <- ubcf_normzlization_history["center"]$center@results[[i]]@cm[1]
  ubcf_normzlization_rmse_zscore[i] <- ubcf_normzlization_history["Z-score"]$`Z-score`@results[[i]]@cm[1]
  ubcf_normzlization_mae_center[i] <- ubcf_normzlization_history["center"]$center@results[[i]]@cm[3]
  ubcf_normzlization_mae_zscore[i] <- ubcf_normzlization_history["Z-score"]$`Z-score`@results[[i]]@cm[3]
}

# wykresy pudelkowe dla UBCF dla normalizacji danych
boxplot(ubcf_normzlization_rmse_center, ubcf_normzlization_rmse_zscore, at = c(1, 2), names = c("center", "Z-score"), ylab = "RMSE", col = c("blue", "red"), main = "Porównanie wartości RMSE dla normalizacji")
boxplot(ubcf_normzlization_mae_center, ubcf_normzlization_mae_zscore, at = c(1, 2), names = c("center", "Z-score"), ylab = "MAE", col = c("blue", "red"), main = "Porównanie wartości MAE dla normalizacji")

# normalizacja - "Z-score"
# testowanie miary podobienstwa "Cosine", "pearson"
similarity_measures <- c("Cosine", "pearson")
ubcf_similarities_history <- list()
for(measure in similarity_measures){
  scheme <- evaluationScheme(ratings_object, method = "cross", k = kfold, given = -examples_for_test)
  algorithm <- list("UBCF" = list(name = "UBCF", param = list(method = measure, normalize = "Z-score")))
  ubcf_similarities_history[measure] <- recommenderlab::evaluate(scheme, algorithm, type = "ratings")
}

ubcf_similarity_rmse_cosine <- vector()
ubcf_similarity_rmse_pearson <- vector()
ubcf_similarity_mae_cosine <- vector()
ubcf_similarity_mae_pearson <- vector()

for(i in 1:kfold){
  ubcf_similarity_rmse_cosine[i] <- ubcf_similarities_history["Cosine"]$Cosine@results[[i]]@cm[1]
  ubcf_similarity_rmse_pearson[i] <- ubcf_similarities_history["pearson"]$pearson@results[[i]]@cm[1]
  ubcf_similarity_mae_cosine[i] <- ubcf_similarities_history["Cosine"]$Cosine@results[[i]]@cm[3]
  ubcf_similarity_mae_pearson[i] <- ubcf_similarities_history["pearson"]$pearson@results[[i]]@cm[3]
}

# wykresy pudelkowe dla UBCF dla miar podobieństwa
boxplot(ubcf_similarity_rmse_cosine, ubcf_similarity_rmse_pearson, at = c(1, 2), names = c("cosine", "pearson"), ylab = "RMSE", col = c("blue", "red"), main = "Porównanie wartości RMSE dla podobieństwa")
boxplot(ubcf_similarity_mae_cosine, ubcf_similarity_mae_pearson, at = c(1, 2), names = c("cosine", "pearson"), ylab = "MAE", col = c("blue", "red"), main = "Porównanie wartości MAE dla podobieństwa")

# normalizacja - "center", podobieństwo - "Pearson"
# testy dla liczby najbliższych sąsiadów nn
nn <- seq(2, 20, 2)
ubcf_nn_history <- list()
for(n in nn){
  scheme <- evaluationScheme(ratings_object, method = "cross", k = kfold, given = -examples_for_test)
  algorithm <- list("UBCF" = list(name = "UBCF", param = list(method = "pearson", normalize = "center")))
  ubcf_nn_history[n] <- recommenderlab::evaluate(scheme, algorithm, type = "ratings")
}

ubcf_nn_rmse <- matrix(0, nrow = kfold, ncol = length(nn))
ubcf_nn_mae <- matrix(0, nrow = kfold, ncol = length(nn))

for(i in 1:length(nn)){
  for(j in 1:kfold){
    ubcf_nn_rmse[j, i] <- ubcf_nn_history[[nn[i]]]@results[[j]]@cm[1]
    ubcf_nn_mae[j, i] <- ubcf_nn_history[[nn[i]]]@results[[j]]@cm[3]
  }
}

# wykresy pudelkowe dla UBCF dla różnej liczby najbliższych sąsiadów
boxplot(ubcf_nn_rmse, xlab = "ilość najbliższych sąsiadów", ylab = "RMSE", names = nn, col = c("blue"), main = "Porównanie wartości RMSE dla najbliższych sąsiadów")
boxplot(ubcf_nn_mae, xlab = "ilość najbliższych sąsiadów", ylab = "MAE", names = nn, col = c("blue"), main = "Porównanie wartości MAE dla najbliższych sąsiadów")


# podsumowanie - normalizacja - "center", podobieństwo - "Pearson", nn = 8


# -------------------------------------------------- SVDF ---------------------------------------------

# testowanie normalizacji "center", "Z-score"
svdf_normzlization_history <- list()
for(type in normalization_types){
  scheme <- evaluationScheme(ratings_object, method = "cross", k = kfold, given = -examples_for_test)
  algorithm <- list("SVDF" = list(name = "SVDF", param = list(normalize = type)))
  svdf_normzlization_history[type] <- recommenderlab::evaluate(scheme, algorithm, type = "ratings")
}

svdf_normzlization_rmse_center <- vector()
svdf_normzlization_rmse_zscore <- vector()
svdf_normzlization_mae_center <- vector()
svdf_normzlization_mae_zscore <- vector()

for(i in 1:kfold){
  svdf_normzlization_rmse_center[i] <- svdf_normzlization_history["center"]$center@results[[i]]@cm[1]
  svdf_normzlization_rmse_zscore[i] <- svdf_normzlization_history["Z-score"]$`Z-score`@results[[i]]@cm[1]
  svdf_normzlization_mae_center[i] <- svdf_normzlization_history["center"]$center@results[[i]]@cm[3]
  svdf_normzlization_mae_zscore[i] <- svdf_normzlization_history["Z-score"]$`Z-score`@results[[i]]@cm[3]
}

# wykresy pudelkowe dla SVDF dla normalizacji danych
boxplot(svdf_normzlization_rmse_center, svdf_normzlization_rmse_zscore, at = c(1, 2), names = c("center", "Z-score"), ylab = "RMSE", col = c("blue", "red"), main = "Porównanie wartości RMSE dla normalizacji")
boxplot(svdf_normzlization_mae_center, svdf_normzlization_mae_zscore, at = c(1, 2), names = c("center", "Z-score"), ylab = "MAE", col = c("blue", "red"), main = "Porównanie wartości MAE dla normalizacji")

# normalizacja - "center"
# testowanie roznych wartosci latent_factor
svd_iters <- 5
k <- seq(6, 6 + 2*(svd_iters - 1), 2)
svdf_k_history <- list()
for(i in 1:length(k)){
  scheme <- evaluationScheme(ratings_object, method = "cross", k = kfold, given = -examples_for_test)
  algorithm <- list("SVDF" = list(name = "SVDF", param = list(normalize = "center", k = k[i])))
  svdf_k_history[i] <- recommenderlab::evaluate(scheme, algorithm, type = "ratings")
}

svdf_k_rmse <- matrix(0, nrow = kfold, ncol = length(k))
svdf_k_mae <- matrix(0, nrow = kfold, ncol = length(k))

for(i in 1:length(k)){
  for(j in 1:kfold){
    svdf_k_rmse[j, i] <- svdf_k_history[[i]]@results[[j]]@cm[1]
    svdf_k_mae[j, i] <- svdf_k_history[[i]]@results[[j]]@cm[3]
  }
}

# wykresy pudelkowe dla SVDF dla różnej liczby latent_factor
boxplot(svdf_k_rmse, xlab = "liczba składowych utajonych", ylab = "RMSE", names = k, col = c("blue"), main = "Porównanie wartości RMSE dla różnej liczby składowych utajonych")
boxplot(svdf_k_mae, xlab = "liczba składowych utajonych", ylab = "MAE", names = k, col = c("blue"), main = "Porównanie wartości MAE dla różnej liczby składowych utajonych")

# normalizacja "Z-score", k = 10
# testowanie współczynnika szybkości uczenia
lambda <- c(5e-4, seq(1e-3, 1e-3*(svd_iters - 1), 1e-3))
svdf_lambda_history <- list()
for(i in 1:length(lambda)){
  scheme <- evaluationScheme(ratings_object, method = "cross", k = kfold, given = -examples_for_test)
  algorithm <- list("SVDF" = list(name = "SVDF", param = list(normalize = "Z-score", k = 10, lambda = lambda[i])))
  svdf_lambda_history[i] <- recommenderlab::evaluate(scheme, algorithm, type = "ratings")
}

svdf_lambda_rmse <- matrix(0, nrow = kfold, ncol = length(lambda))
svdf_lambda_mae <- matrix(0, nrow = kfold, ncol = length(lambda))

for(i in 1:length(lambda)){
  for(j in 1:kfold){
    svdf_lambda_rmse[j, i] <- svdf_lambda_history[[i]]@results[[j]]@cm[1]
    svdf_lambda_mae[j, i] <- svdf_lambda_history[[i]]@results[[j]]@cm[3]
  }
}

# wykresy pudelkowe dla SVDF dla różnych wartości współczynnika szybkości uczenia
boxplot(svdf_lambda_rmse, xlab = "współczynnik szybkości uczenia", ylab = "RMSE", names = lambda, col = c("blue"), main = "Porównanie wartości RMSE dla różnych współczynników szybkości uczenia")
boxplot(svdf_lambda_mae, xlab = "współczynnik szybkości uczenia", ylab = "MAE", names = lambda, col = c("blue"), main = "Porównanie wartości MAE dla różnych współczynników szybkości uczenia")

# podsumowanie - normalizacja "Z-score", k = 10, lambda = 0.001


# -------------------------------------------------- porównanie najlepszych modeli ---------------------------------------------

boxplot(ubcf_nn_rmse[, 4], svdf_lambda_rmse[, 4], at = c(1, 2), names = c("UBCF", "SVDF"), ylab = "RMSE", col = c("blue", "red"), main = "Porównanie wartości RMSE dla najlepszych modeli")
boxplot(ubcf_nn_mae[, 2], svdf_lambda_mae[, 2], at = c(1, 2), names = c("UBCF", "SVDF"), ylab = "MAE", col = c("blue", "red"), main = "Porównanie wartości MAE dla najlepszych modeli")

# -----------------------------------------------------------------------------------------------------
# binaryzacja problemu i ROC 
# -----------------------------------------------------------------------------------------------------