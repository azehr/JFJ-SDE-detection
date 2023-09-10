# Author: Andrew Zehr
# Title: TRex_select
#
#  This script uses the T-Rex variable selection framework to perform variable
#  selection on the JFJ dataset

library(TRexSelector)

data <- read.csv("~/ETH Zurich/Masters Thesis/main/data/processed/Data_cleaned_Rob/aerosol_data_JFJ_2015_to_2021_CLEANED_May2023.csv", row.names="X")
data2020 <- subset(data, row.names(data) > as.POSIXct("2020-01-01 00:00:00"))


data2020 <- subset(data2020, select = -c(block_nr))



X <- as.matrix(data2020[complete.cases(data2020),])



labels <- read.csv("~/ETH Zurich/Masters Thesis/main/data/raw/Jungfraujoch/dust_event_info.csv", row.names="DateTimeUTC")
Y <- labels[row.names(X),]$sde_event

# Seed
set.seed(1234)

# Numerical zero
eps <- .Machine$double.eps

# Make a list of FDR target-thresholds to investigate
fdr_targets <- c(0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8)


cols <- colnames(data2020)

results <- data.frame(matrix(ncol = length(cols), nrow = length(fdr_targets)))
colnames(results) <- cols
row.names(results) <- fdr_targets


for (fdr in fdr_targets) {
  res <- trex(X = X, y = Y, tFDR = fdr, verbose = FALSE)
  selected_var <- which(res$selected_var > eps)
  selected_features <- colnames(data2020)[selected_var]
  out <- rep(0, length(cols))
  out[selected_var] <- 1
  
  results[toString(fdr), ] <- out
}

write.csv(results, "trex_selected_features.csv")

  
res <- trex(X = X, y = Y, tFDR = 0.01, verbose = FALSE)
selected_var <- which(res$selected_var > eps)
selected_features <- colnames(data2020)[selected_var]
