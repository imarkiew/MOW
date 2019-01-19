# ustawienie domyolnej ocie??ki do pliku
#path = "/home/igor/Pulpit/GitHub/MOW"
path = "C:/Users/Aleksander/Desktop/Sem 9/MOW/Projekt"
setwd(path)

# 3adowanie bibliotek
library(recommenderlab)
library(caret)
library(dismo)
library(caTools)
library(Matrix)
library(purrr)
library(dplyr)
library(graphics)

# minimalna liczba ocen jak1 musi miea u??ytkownik ??eby by3 uwzgledniony w danych
min_nr_of_ratings <- 5

# liczba u??ytkowników przeznaczonych do badan
nr_of_users_to_choose <- 1000

# ustawienie ziarna generatora liczb pseudolosowych
seed_ <- 1648
set.seed(seed_)

# wczytanie danych
ratings_raw <- read.csv(paste( "BX-CSV-Dump/BX-Book-Ratings.csv", sep = "/"), sep = ";", header = TRUE)

# usuniecie rekordów zawieraj1cych oceny (binarne) zebrane w sposób pooredni
ratings_raw <- ratings_raw[(ratings_raw$Book.Rating != 0), ]

# filtracja rekordów w których u??ytkownicy maj1 ocenione co najmniej min_nr_of_ratings ksi1??ek
unique_user_ids <- as.data.frame(table(ratings_raw$User.ID)) %>% filter(as.numeric(Freq) >= min_nr_of_ratings)
unique_user_ids <- unique_user_ids[, 1]
unique_user_ids_length <- length(unique_user_ids) # unique_user_ids zawiera wartooci unikalne

# losowanie bez zwracania nr_of_users_to_choose u??ytkowników
samples <- sample(1:unique_user_ids_length, nr_of_users_to_choose, replace = FALSE) # samples zawiera indeksy
unique_choosed_user_ids <- unique_user_ids[samples] # choosed_user_ids zawiera wybrane wartooci unikalne ID u??ytkowników
ratings <- ratings_raw[ratings_raw$User.ID %in% unique_choosed_user_ids, ]

# przypisanie nowych ID u??ytkowników i ksi1??ek do ratings
ratings$New.User.ID <- group_indices(ratings, User.ID)
ratings$New.Book.ID <- group_indices(ratings, ISBN)

# utworzenie macierzy ??adkiej
ratings_sparse = sparseMatrix(as.integer(ratings$New.User.ID), as.integer(ratings$New.Book.ID), x = ratings$Book.Rating)
colnames(ratings_sparse) = levels(factor(ratings$ISBN))
rownames(ratings_sparse) = levels(factor(ratings$User.ID))

# utworzenie obiektu realRatingMatrix
ratings_object <- new("realRatingMatrix", data = ratings_sparse)

# rozk3ad rozk3ad ocen w zale??nooci on ID u??ytkownika i ISBN ksi1??ki
image(ratings_object, main = "Rozk3ad ocen")

# histogram liczba ocen - liczba u??ytkowników którzy wystawili tak1 liczbe ocen
row_counts <- rowCounts(ratings_object)
hist(row_counts, 
     main = "Histogram liczba ocen - liczba u??ytkwoników",
     col = "darkslategray4", 
     border = "red", 
     xlab = "liczba ocen",
     ylab = "liczba u??ytkowników",
     breaks = seq(min_nr_of_ratings - 1, max(row_counts) + 1, by = 10)
)

# histogram liczba ocen - liczba ksi1??ek, które maj1 wystawion1 tak1 liczbe ocen
column_counts <- colCounts(ratings_object)
hist(column_counts, 
     main = "Histogram liczba ocen - liczba ksi1??ek",
     col = "darkslategray4", 
     border = "red", 
     xlab = "liczba ocen",
     ylab = "liczba ksi1??ek",
     breaks = seq(1, max(column_counts) + 1, by = 1)
)

# histogram rozk3adu ocen - NIE RYSOWALO MI PRZY "HISTOGRAM"
hist(getRatings(ratings_object), 
     type = "percent",
     main = paste("Histogram ocen - liczba ocen: ", length(getRatings(ratings_object)), ""), 
     col = "darkslategray4", 
     border = "red", 
     xlab = "ocena", 
     ylab = "% ca3ooci", 
     xlim = c(0.5, 10.5), 
     #ylim = c(0, 30), 
     breaks = -0.5:(max(getRatings(ratings_object) + 1) - 0.5), 
     scales = list(x = list(at = 1:10)), 
     panel = function(...){
       panel.grid(h = 5, v = 10)
       panel.histogram(...)
     })

# podstawowe statystyki
summary(getRatings(ratings_object))

# wektory zawieraj1ce kolejne badane wartoœci parametrów
normalizations_vector <- c("center", "Z-score")
# UBCF
methods_vector <- c("Cosine", "pearson")
delta_nn <- 10
nn_vector <- seq(2,10,2)
# SVDF
SVD_iters <- 2
k_vector <- seq(6, 6 + 2*(SVD_iters-1),2)
gamma_vector <- seq(0.05, 0.05 + 0.05*(SVD_iters-1), 0.05)
lambda_vector <- c(5e-4, seq(1e-3, 1e-3*(SVD_iters-1), 1e-3))

# PÊTLA DO OPTYMALIZACJI PARAMETRÓW ALGORYMTÓW
# parameter_vector1 = {1, normalizations_vector}
UBCF_results = 0
SVDF_results = 0
parameter_vector1 = normalizations_vector
K = 10 # liczba podzbiorów walidacji
for (i in 1:length(parameter_vector1)) {
  
  # normalizacja danych -  "center", "Z-score"
  normalized_ratings_object <- normalize(ratings_object, method = parameter_vector1[i], row = TRUE)

  # schemat ewaluacji, metody - "UBCF","SVDF", method = "cross", k = 10
  scheme <- evaluationScheme(normalized_ratings_object, method = "cross", k = K, given = -1)
  
  # UBCF - NALE¯Y OKREŒLIÆ WARTOŒÆ WEKTORA "parameter_vector2" W ZALE¯NOŒCI KTÓRY PARAMETR CHCEMY OPTYMALIZOWAÆ - parameter_vector2 = {1, methods_vector, nn_vector}
  # ZOPTYMALIZOWANE DANE NALE¯Y WPISAÆ NA LISTÊ "param = list()" I DOPISAÆ NOWY DO ZOPTYMALIZOWANIA JAKO "parameter_vector[j]"
  parameter_vector2 = 1
  for (j in 1:length(parameter_vector2)) {
      "UBCF" = list(name = "UBCF", param = list(normalize = NULL) )
      UBCF_result <- recommenderlab::evaluate(scheme, "UBCF", type = "ratings")
      UBCF_results = c(UBCF_results,UBCF_result)
  }
  
  # SVDF - NALE¯Y OKREŒLIÆ WARTOŒÆ WEKTORA "parameter_vector2" W ZALE¯NOŒCI KTÓRY PARAMETR CHCEMY OPTYMALIZOWAÆ - parameter_vector2 = {1, k_vector, gamma_vector lambda_vector}
  # ZOPTYMALIZOWANE DANE NALE¯Y WPISAÆ NA LISTÊ "param = list()" I DOPISAÆ NOWY DO ZOPTYMALIZOWANIA JAKO "parameter_vector[j]"
#  parameter_vector2 = 1
  for (j in 1:length(parameter_vector2)) {
    algorithm <- list(SVDF = list(name = "SVDF", param = list(normalize = NULL)))
    SVDF_result <- recommenderlab::evaluate(scheme, algorithm, type = "ratings")
    SVDF_results = c(SVDF_results,SVDF_result)
  }
}

UBCF_results[[1]] = NULL
N_UBCF = length(UBCF_results)
UBCF_RMSE_matrix = matrix(nrow = K, ncol = N_UBCF)
UBCF_MAE_matrix = matrix(nrow = K, ncol = N_UBCF)
for (i in 1:K) {
  for (j in 1:N_UBCF) {
    UBCF_RMSE_matrix[i,j] = UBCF_results[[j]]@results[[i]]@cm[[1]] 
    UBCF_MAE_matrix[i,j] = UBCF_results[[j]]@results[[i]]@cm[[3]] 
  }
}
boxplot(UBCF_RMSE_matrix)
boxplot(UBCF_MAE_matrix)

SVDF_results[[1]] = NULL
N_SVDF = length(SVDF_results)
SVDF_RMSE_matrix = matrix(nrow = K, ncol = N_UBCF)
SVDF_MAE_matrix = matrix(nrow = K, ncol = N_UBCF)
for (i in 1:K) {
  for (j in 1:N_SVDF) {
    SVDF_RMSE_matrix[i,j] = SVDF_results[[j]]@results[[i]]@cm[[1]] 
    SVDF_MAE_matrix[i,j] = SVDF_results[[j]]@results[[i]]@cm[[3]]
  }
}
boxplot(SVDF_RMSE_matrix)
boxplot(SVDF_MAE_matrix)







# -------------------------------------------------------------
# KOD DO URUCHOMIENIA PO OPTYMALIZACJI PARAMETRÓW
# -------------------------------------------------------------

#Testowanie z optymalnymi parametrami
#scheme <- evaluationScheme(test, method="split",  k = 1, given = -2, goodRating = 0)
#algorithms <- list( RANDOM = list(name = "RANDOM", param = list(normalize = NULL)), 
#                    POPULAR = list(name = "POPULAR", param = list(normalize = NULL)),
#                    UBCF = list(name="UBCF", param = list(method = "Cosine", nn = 6, normalize = NULL)))#, 
#SVDF = list(name="SVDF", param = list(k = 8, gamma = 0.1, lambda = 1e-4, max_epoc = 20, normalize = NULL))) 
#results <- recommenderlab::evaluate(scheme, algorithms, type = "topNList")
#recommenderlab::plot(results, legend = "topright") 
#recommenderlab::plot(results, "ROC", annotate = 3, legend="topright") # NIE WIEM CZEMU TO RYSUJE TO SAMO CO POPRZEDNI PLOT, NIE WIEM CZEMU OBA RYSUJY ROC (ALE CHUJOWE) GDY TYPE = 'TOPNLIST'


#r <- getRatings(normalized_ratings_object)
#x <- seq(min(r), max(r), length = 50) 
#y <- 100*dnorm(x, mean = mean(r), sd = sd(r))
# "IBCF" - bardzo d3ugie obliczenia, wartooci NaN, dla wiekszej ilooci danych bardzo zasobo??erne
# "IBCF" = list(name = "IBCF", param = list(method = "Cosine", k = 25, normalize = NULL)),
#POPULAR = list(name = "POPULAR", param = NULL, normalize = NULL),
# histogram znormalizowanych ocen
#histogram(r,
#          type = "percent",
#          main = "Znormalizowany histogram ocen", 
#          col = "darkslategray4", 
#          border = "red", 
#          xlab = "ocena przeskalowana", 
#          ylab = "% ca3ooci",
#          scales = list(x = list(at = floor(min(r)):ceiling(max(r)))), 
#          breaks = 10,
#         ylim = c(0, 35),
#          key = list(
#          corner = c(0, 0.95), 
#          lines = list(col = c("darkslategray4", "green"), lty = 1, lwd = 2), 
#          text = list(c("histogram", "rozk3ad normalny"))), 
#          panel = function(...){
#  panel.grid(h = 5, v = 10)
#  panel.histogram(...)
#  panel.lines(x, y, col = "green", lwd = 2)
#})