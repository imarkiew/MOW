# ustawienie domyœlnej œcie¿ki do pliku
#path = "/home/igor/Pulpit/GitHub/MOW"
#setwd(path)

# ³adowanie bibliotek
library(recommenderlab)
library(caret)
library(dismo)
library(caTools)
library(Matrix)
library(purrr)
library(dplyr)
library(graphics)

# minimalna liczba ocen jak¹ musi mieæ u¿ytkownik ¿eby by³ uwzglêdniony w danych
min_nr_of_ratings <- 5

# liczba u¿ytkowników przeznaczonych do badañ
nr_of_users_to_choose <- 1000

# ustawienie ziarna generatora liczb pseudolosowych
seed_ <- 1648
set.seed(seed_)

# wczytanie danych
ratings_raw <- read.csv(paste( "BX-CSV-Dump/BX-Book-Ratings.csv", sep = "/"), sep = ";", header = TRUE)

# usuniêcie rekordów zawieraj¹cych oceny (binarne) zebrane w sposób poœredni
ratings_raw <- ratings_raw[(ratings_raw$Book.Rating != 0), ]

# filtracja rekordów w których u¿ytkownicy maj¹ ocenione co najmniej min_nr_of_ratings ksi¹¿ek
unique_user_ids <- as.data.frame(table(ratings_raw$User.ID)) %>% filter(as.numeric(Freq) >= min_nr_of_ratings)
unique_user_ids <- unique_user_ids[, 1]
unique_user_ids_length <- length(unique_user_ids) # unique_user_ids zawiera wartoœci unikalne

# losowanie bez zwracania nr_of_users_to_choose u¿ytkowników
samples <- sample(1:unique_user_ids_length, nr_of_users_to_choose, replace = FALSE) # samples zawiera indeksy
unique_choosed_user_ids <- unique_user_ids[samples] # choosed_user_ids zawiera wybrane wartoœci unikalne ID u¿ytkowników
ratings <- ratings_raw[ratings_raw$User.ID %in% unique_choosed_user_ids, ]

# przypisanie nowych ID u¿ytkowników i ksi¹¿ek do ratings
ratings$New.User.ID <- group_indices(ratings, User.ID)
ratings$New.Book.ID <- group_indices(ratings, ISBN)

# utworzenie macierzy ¿adkiej
ratings_sparse = sparseMatrix(as.integer(ratings$New.User.ID), as.integer(ratings$New.Book.ID), x = ratings$Book.Rating)
colnames(ratings_sparse) = levels(factor(ratings$ISBN))
rownames(ratings_sparse) = levels(factor(ratings$User.ID))

# utworzenie obiektu realRatingMatrix
ratings_object <- new("realRatingMatrix", data = ratings_sparse)

# rozk³ad rozk³ad ocen w zale¿noœci on ID u¿ytkownika i ISBN ksi¹¿ki
image(ratings_object, main = "Rozk³ad ocen")

# histogram liczba ocen - liczba u¿ytkowników którzy wystawili tak¹ liczbê ocen
row_counts <- rowCounts(ratings_object)
hist(row_counts, 
     main = "Histogram liczba ocen - liczba u¿ytkwoników",
     col = "darkslategray4", 
     border = "red", 
     xlab = "liczba ocen",
     ylab = "liczba u¿ytkowników",
     breaks = seq(min_nr_of_ratings - 1, max(row_counts) + 1, by = 10)
)

# histogram liczba ocen - liczba ksi¹¿ek, które maj¹ wystawion¹ tak¹ liczbê ocen
column_counts <- colCounts(ratings_object)
hist(column_counts, 
     main = "Histogram liczba ocen - liczba ksi¹¿ek",
     col = "darkslategray4", 
     border = "red", 
     xlab = "liczba ocen",
     ylab = "liczba ksi¹¿ek",
     breaks = seq(1, max(column_counts) + 1, by = 1)
)

# histogram rozk³adu ocen - NIE RYSOWA£O MI PRZY "HISTOGRAM"
hist(getRatings(ratings_object), 
          type = "percent",
          main = paste("Histogram ocen - liczba ocen: ", length(getRatings(ratings_object)), ""), 
          col = "darkslategray4", 
          border = "red", 
          xlab = "ocena", 
          ylab = "% ca³oœci", 
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

# normalizacja danych - "center", "Z-score"
normalized_ratings_object <- normalize(ratings_object, method = "center", row = TRUE)
r <- getRatings(normalized_ratings_object)
x <- seq(min(r), max(r), length = 50) 
y <- 100*dnorm(x, mean = mean(r), sd = sd(r))

# wektory zawieraj¹ce kolejne badane wartoœci parametrów
normalizations_vector <- c("center", "Z-score")
methods_vector <- c("Cosine", "pearson")
delta_nn <- 10
nn_vector <- 2:10
SVD_iters <- 2
k_vector <- seq(6, 6 + 2*(SVD_iters-1),2)
gamma_vector <- seq(0.05, 0.05 + 0.05*(SVD_iters-1), 0.05)
lambda_vector <- c(5e-4, seq(1e-3, 1e-3*(SVD_iters-1), 1e-3))
max_epoc_vector <- seq(10, 10 + 10*(SVD_iters-1), 10)

best_UBCF_err = list(RMSE = Inf, MSE = Inf, MAE = Inf)
best_UBCF_par = list(RMSE = NULL, MSE = NULL, MAE = NULL)
best_SVDF_err = list(RMSE = Inf, MSE = Inf, MAE = Inf)
best_SVDF_par = list(RMSE = NULL, MSE = NULL, MAE = NULL)

for (i_norm in 1:length(normalizations_vector)) {
  
  # normalizacja danych -  "center", "Z-score"
  normalized_ratings_object <- normalize(ratings_object, method = normalizations_vector[i_norm], row = TRUE)
  r <- getRatings(normalized_ratings_object)
  x <- seq(min(r), max(r), length = 50) 
  y <- 100*dnorm(x, mean = mean(r), sd = sd(r))
  
  # histogram znormalizowanych ocen
  #histogram(r,
  #          type = "percent",
  #          main = "Znormalizowany histogram ocen", 
  #          col = "darkslategray4", 
  #          border = "red", 
  #          xlab = "ocena przeskalowana", 
  #          ylab = "% ca³oœci",
  #          scales = list(x = list(at = floor(min(r)):ceiling(max(r)))), 
  #          breaks = 10,
  #         ylim = c(0, 35),
  #          key = list(
  #          corner = c(0, 0.95), 
  #          lines = list(col = c("darkslategray4", "green"), lty = 1, lwd = 2), 
  #          text = list(c("histogram", "rozk³ad normalny"))), 
  #          panel = function(...){
  #  panel.grid(h = 5, v = 10)
  #  panel.histogram(...)
  #  panel.lines(x, y, col = "green", lwd = 2)
  #})
  
  # i tak normalizujemy dla ka¿dego u¿ytkownika osobno (wierszami), wiêc mo¿emy podzia³a na zbiory test i train przesun¹Ä‡ po wspólnej normalizacji
  sample <- sample.split(1:nrow(normalized_ratings_object), SplitRatio = 0.8)
  train <- subset(normalized_ratings_object, sample == TRUE)
  test <- subset(normalized_ratings_object, sample == FALSE)
  
  # schemat ewaluacji
  # metody - "UBCF", "IBCF", "SVDF", "POPULAR", ....
  # method = "cross", k = 5
  scheme <- evaluationScheme(train, method = "cross", k = 5, given = -1)
  
  # UBCF - metryki podobieñstwa - "Cosine", "pearson"
  for (i_mv in 1:length(methods_vector)) {
    for (i_nn in 1:length(nn_vector)) {
      algorithm <- list(UBCF = list(name = "UBCF", param = list(method = methods_vector[i_mv], nn = nn_vector[i_nn], normalize = NULL)))
      UBCF_result_temp <- recommenderlab::evaluate(scheme, algorithm, type = "ratings")
      avg_UBCF_result_temp <- avg(UBCF_result_temp)
      list[best_UBCF_err, best_UBCF_par] = checking_for_opt_params(best_UBCF_err, best_UBCF_par, avg_UBCF_result_temp$UBCF, algorithm, normalizations_vector[i_norm])
    }
  }
  
  #SVDF
  for (i_kv in 1:SVD_iters) {
    for(i_gv in 1:SVD_iters) {
      for(i_lv in 1:SVD_iters) {
        for(i_mev in 1:SVD_iters) {
          algorithm <- list(SVDF = list(name = "SVDF", param = list(k = k_vector[i_kv], gamma = gamma_vector[i_gv], lambda = lambda_vector[i_lv],
                                                                    max_epoc = max_epoc_vector[i_mev], normalize = NULL)))
          SVDF_result_temp <- recommenderlab::evaluate(scheme, algorithm, type = "ratings")
          avg_SVDF_result_temp <- avg(SVDF_result_temp)
          list[best_SVDF_err, best_SVDF_par] = checking_for_opt_params(best_SVDF_err, best_SVDF_par, avg_SVDF_result_temp$SVDF, algorithm, normalizations_vector[i_norm])
        }
      }
    }
  }
}

# Testowanie z optymalnymi parametrami
scheme <- evaluationScheme(test, method="split",  k = 1, given = -1, goodRating = 0)
algorithms <- list( RANDOM = list(name = "RANDOM", param = list(normalize = NULL)), 
                    POPULAR = list(name = "POPULAR", param = list(normalize = NULL)),
                    UBCF = list(name="UBCF", param = list(method = "Cosine", nn = 6, normalize = NULL)))#, 
                    #SVDF = list(name="SVDF", param = list(k = 8, gamma = 0.1, lambda = 1e-4, max_epoc = 20, normalize = NULL))) 
results <- recommenderlab::evaluate(scheme, algorithms, type = "ratings")
recommenderlab::plot(results, legend = "topright") 
recommenderlab::plot(results, "ROC", annotate = 3, legend="topright") # NIE WIEM CZEMU TO RYSUJE TO SAMO CO POPRZEDNI PLOT, NIE WIEM CZEMU OBA RYSUJ¥ ROC (ALE CHUJOWE) GDY TYPE = 'TOPNLIST'


# "IBCF" - bardzo d³ugie obliczenia, wartoœci NaN, dla wiêkszej iloœci danych bardzo zasobo¿erne
# "IBCF" = list(name = "IBCF", param = list(method = "Cosine", k = 25, normalize = NULL)),
#POPULAR = list(name = "POPULAR", param = NULL, normalize = NULL),