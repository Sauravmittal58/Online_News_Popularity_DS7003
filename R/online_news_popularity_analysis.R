rm(list = ls())         
graphics.off()         
cat("\014")

# Load and install the required packages
if (!require("pacman")) install.packages("pacman")
pacman::p_load(caret, dplyr, rpart, rpart.plot, e1071, ggplot2, corrplot, gridExtra)

# Load the dataset
OnlineNewsPopularityData <- read.csv("OnlineNewsPopularity.csv")

# Check column names
print(colnames(OnlineNewsPopularityData))

# Display some basic information about the dataset
print(paste("Dataset Dimensions:", dim(OnlineNewsPopularityData)[1], "rows and", dim(OnlineNewsPopularityData)[2], "columns"))

# Print the first few rows of the dataset
head(OnlineNewsPopularityData)

# Check for missing values
colSums(is.na(OnlineNewsPopularityData))

# Get summary statistics for numeric columns
print(summary(OnlineNewsPopularityData))

# Remove non-predictive attributes: 'url' and 'timedelta' 
OnlineNewsPopularityData <- OnlineNewsPopularityData %>% select(-url, -timedelta)

# Convert 'shares' column into a binary target variable
threshold <- quantile(OnlineNewsPopularityData$shares, 0.5)
OnlineNewsPopularityData$Popularity <- ifelse(OnlineNewsPopularityData$shares >= threshold, "Popular", "Unpopular")
OnlineNewsPopularityData$Popularity <- factor(OnlineNewsPopularityData$Popularity, levels = c("Unpopular","Popular"))

# Drop the original 'shares' column to prevent data leakage
OnlineNewsPopularityData <- OnlineNewsPopularityData %>% select(-shares)

# Clean column names
colnames(OnlineNewsPopularityData) <- gsub("\\.", "_", colnames(OnlineNewsPopularityData))

# Creating ratio between content and title length
OnlineNewsPopularityData$content_title_ratio <- 
  OnlineNewsPopularityData$n_tokens_content / (OnlineNewsPopularityData$n_tokens_title + 1)

# Creating media engagement feature
OnlineNewsPopularityData$media_score <- 
  OnlineNewsPopularityData$num_imgs + OnlineNewsPopularityData$num_videos

# Plot the Distribution of Popular vs Unpopular Articles
ggplot(OnlineNewsPopularityData, aes(x = Popularity, fill = Popularity)) +
  geom_bar(color = "black") +
  theme_minimal() +
  scale_fill_manual(values = c("Popular" = "lightgreen", "Unpopular" = "salmon")) +
  labs(title = "Distribution of News Popularity", x = "Class", y = "Number of Articles") +
  theme(legend.position = "none")

# Plot the Number of Words in Content vs Popularity
ggplot(OnlineNewsPopularityData, aes(x = Popularity, y = n_tokens_content + 1, fill = Popularity)) +
  geom_boxplot(color = "black") +
  scale_y_log10() +
  theme_minimal() +
  scale_fill_manual(values = c("Popular" = "lightgreen", "Unpopular" = "salmon")) +
  labs(title = "Content Length vs Popularity", x = "Class", y = "Word Count (Log Scale + 1)") +
  theme(legend.position = "none")

# Plot the Number of Images vs Popularity
ggplot(OnlineNewsPopularityData, aes(x = Popularity, y = num_imgs, fill = Popularity)) +
  geom_boxplot(color = "black", outlier.shape = NA) +
  coord_cartesian(ylim = c(0, 20)) +
  theme_minimal() +
  scale_fill_manual(values = c("Popular" = "lightgreen", "Unpopular" = "salmon")) +
  labs(title = "Number of Images vs Popularity", x = "Class", y = "Image Count") +
  theme(legend.position = "none")

# Q-Q plots to check normality of the raw target-driving variable ('shares')
# and a couple of key numeric predictors. These help justify transformations
# (e.g. log-scaling) used elsewhere in the EDA.
shares_raw <- read.csv("OnlineNewsPopularity.csv")$shares

par(mfrow = c(1, 2))
qqnorm(shares_raw, main = "Q-Q Plot: Raw shares")
qqline(shares_raw, col = "red")

qqnorm(log1p(shares_raw), main = "Q-Q Plot: log(1 + shares)")
qqline(log1p(shares_raw), col = "red")
par(mfrow = c(1, 1))

qqnorm(OnlineNewsPopularityData$n_tokens_content, main = "Q-Q Plot: n_tokens_content")
qqline(OnlineNewsPopularityData$n_tokens_content, col = "red")

# convert 'Popularity' back to numeric for correlation
popularity_numeric <- ifelse(OnlineNewsPopularityData$Popularity == "Popular", 1, 0)

# Select numeric columns for correlation analysis
numeric_cols_full <- sapply(OnlineNewsPopularityData, is.numeric)

# Calculate correlations between numeric columns and the numeric Popularity
cor_values <- cor(OnlineNewsPopularityData[numeric_cols_full], popularity_numeric)

# Select features with correlation > 0.05 
highly_correlated_features <- names(OnlineNewsPopularityData)[numeric_cols_full][abs(cor_values) > 0.05]

# Create the subset correlation matrix
cor_matrix <- cor(OnlineNewsPopularityData[highly_correlated_features])

# Plot the correlation matrix
corrplot(cor_matrix, method = "circle", type = "upper", 
         tl.col = "black", tl.srt = 90, tl.cex = 0.7, 
         mar = c(1, 1, 1, 1), 
         title = "Correlation Matrix of Highly Correlated Features")

# Bar chart of each feature's correlation with the target (easier to read at a glance)
cor_df <- data.frame(
  feature = highly_correlated_features,
  correlation = as.numeric(cor_values[highly_correlated_features, ])
)
cor_df <- cor_df[order(cor_df$correlation), ]
cor_df$feature <- factor(cor_df$feature, levels = cor_df$feature)

ggplot(cor_df, aes(x = feature, y = correlation, fill = correlation > 0)) +
  geom_col(color = "black") +
  coord_flip() +
  theme_minimal() +
  scale_fill_manual(values = c("TRUE" = "lightgreen", "FALSE" = "salmon")) +
  labs(title = "Feature Correlation with Popularity (|r| > 0.05)",
       x = NULL, y = "Correlation coefficient") +
  theme(legend.position = "none")

# STRATIFIED SAMPLING 
set.seed(123)
sample_index <- createDataPartition(OnlineNewsPopularityData$Popularity, p = 0.5, list = FALSE)
OnlineNewsPopularityData <- OnlineNewsPopularityData[sample_index, ]

# Ensure both classes exist after sampling
if(length(unique(OnlineNewsPopularityData$Popularity)) < 2){
  stop("Sampling removed one class. Increase sampling size.")
}

# Identify numeric columns for standardization
numeric_cols <- setdiff(names(OnlineNewsPopularityData)[sapply(OnlineNewsPopularityData, is.numeric)], "Popularity")

# Remove near-zero variance predictors 
nzv_cols <- nearZeroVar(OnlineNewsPopularityData[numeric_cols], saveMetrics = FALSE)

if(length(nzv_cols) > 0) {
  OnlineNewsPopularityData <- OnlineNewsPopularityData[, -nzv_cols]
  
  numeric_cols <- setdiff(names(OnlineNewsPopularityData)[sapply(OnlineNewsPopularityData, is.numeric)], "Popularity")
}

# Standardize numeric columns
OnlineNewsPopularityData[numeric_cols] <- scale(OnlineNewsPopularityData[numeric_cols])

# Split the sampled data into 80% Train and 20% Test
set.seed(123)
train_index <- createDataPartition(OnlineNewsPopularityData$Popularity, p = 0.8, list = FALSE)

train_data <- OnlineNewsPopularityData[train_index, ]
test_data  <- OnlineNewsPopularityData[-train_index, ]

# Define training control
train_control <- trainControl(method = "cv", number = 3, sampling = "up")

# Implementing the Decision Tree model
dt_model <- train(
  Popularity ~ ., 
  data = train_data,
  method = "rpart",
  trControl = train_control,
  tuneLength = 5)

# Predicting and Evaluate Decision Tree
dt_pred <- predict(dt_model, test_data)
print(confusionMatrix(dt_pred, test_data$Popularity, positive = "Popular"))

# Implementing the SVM model 
svm_model <- train(
  Popularity ~ ., 
  data = train_data,
  method = "svmRadial",
  trControl = train_control,
  tuneLength = 5)

# Predicting and Evaluate SVM
svm_pred <- predict(svm_model, test_data)

print(confusionMatrix(svm_pred, test_data$Popularity, positive = "Popular"))

# Implementing the Random Forest model
rf_model <- train(
  Popularity ~ ., 
  data = train_data,
  method = "rf",
  trControl = train_control,
  tuneGrid = data.frame(mtry = c(5, 10, 15)))

# Predicting and Evaluate Random Forest
rf_pred <- predict(rf_model, test_data)

print(confusionMatrix(rf_pred, test_data$Popularity, positive = "Popular"))
