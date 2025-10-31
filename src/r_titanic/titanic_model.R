# Titanic Survival Prediction - R Version

library(readr)
library(dplyr)
library(caret)

cat("Libraries loaded successfully.\n")

# Load data
train <- read_csv("src/data/train.csv")
test  <- read_csv("src/data/test.csv")
cat("Data loaded successfully.\n")

# Handle missing values
train$Age[is.na(train$Age)] <- median(train$Age, na.rm = TRUE)
test$Age[is.na(test$Age)]   <- median(test$Age, na.rm = TRUE)
test$Fare[is.na(test$Fare)] <- median(test$Fare, na.rm = TRUE)
train$Embarked[is.na(train$Embarked)] <- "S"
test$Embarked[is.na(test$Embarked)]   <- "S"
cat("Missing values handled.\n")

# Convert categorical variables
train$Sex <- as.factor(train$Sex)
test$Sex  <- as.factor(test$Sex)
train$Embarked <- as.factor(train$Embarked)
test$Embarked  <- as.factor(test$Embarked)
cat("Categorical variables converted.\n")

# Logistic regression model
model <- glm(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked,
             data = train, family = binomial)
cat("Model trained successfully.\n")

# Training accuracy
train_pred <- ifelse(predict(model, type = "response") > 0.5, 1, 0)
train_acc <- mean(train_pred == train$Survived)
cat("Training accuracy:", round(train_acc, 4), "\n")

# Predict test set
test_pred <- ifelse(predict(model, newdata = test, type = "response") > 0.5, 1, 0)

# Save results
pred <- data.frame(PassengerId = test$PassengerId, Survived = test_pred)
write_csv(pred, "src/data/predictions_r.csv")
cat("Predictions saved to src/data/predictions_r.csv\n")
