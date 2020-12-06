
# To load the required library tools and if not previously installed, install the relevant packages.
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
library(tidyverse)
library(caret)
library(data.table)



#Load the data file, and mutate the class column from string type to factor type.
data <- read.csv("Kaggle-source-data.csv") %>% mutate(class = as.factor(class))

#Examine the first few rows of the data set.
head(data)

#Check how many rows there are in the data set.
number_of_rows <- nrow(data)
number_of_rows



#Calculate the percentage of normal diagnosis in the data set.
percentage_normal <- mean(data$class == "Normal")
percentage_normal



# Split the data set into the test (15%) and train (85%) sets.
set.seed(1, sample.kind="Rounding")
test_index <- createDataPartition(y = data$class, times = 1, p = 0.15, list = FALSE)
train_set <- data[-test_index,]
test_set <- data[test_index,]



#Model 1: Logistic Regression Model
  
# Train the logistic model on the train set.
glm_fit <- train(class ~ ., method = "glm", data = train_set)

# Make predictions on the test set using the trained model.
glm_predict <- predict(glm_fit, test_set, type = "raw")

# Calculate the prediction accuracy on the test set.
glm_accuracy <- confusionMatrix(glm_predict, test_set$class)$overall[["Accuracy"]]
glm_accuracy



#Model 2: K Nearest Neighbours Model  
  
# Set the cross validation to be 10-fold. 
control <- trainControl(method = "cv", number = 10, p = 0.9)

# Train the k nearest neighbours model on the train set 
# with a tuning grid of k ranging from 3 to 39.
knn_fit <- train(class ~ ., method = "knn", tuneGrid = data.frame(k = seq(3, 39, 2)), 
                 data = train_set, trControl = control)

#Visualize the tuning results of k.
ggplot(knn_fit, highlight = TRUE)

#Show the trained model.
knn_fit$finalModel
optimal_k_on_train_set <- knn_fit$finalModel$k
optimal_k_on_train_set

# Make predictions on the test set using the trained model.
knn_predict <- predict(knn_fit, test_set, type = "raw")

# Calculate the prediction accuracy on the test set.
knn_accuracy <- confusionMatrix(knn_predict, test_set$class)$overall[["Accuracy"]]
knn_accuracy



#The final prediction model: K Nearest Neighbours Model with 10-fold cross validation
  
# Set the cross validation to be 10-fold. 
control <- trainControl(method = "cv", number = 10, p = 0.9)

# Train the k nearest neighbours model on the entire data set 
# with a tuning grid of k ranging from 3 to 39.
final_knn_fit <- train(class ~ ., method = "knn", tuneGrid = data.frame(k = seq(3, 39, 2)), 
                       data = data, trControl = control)

# Visualize the tuning results of k.
ggplot(final_knn_fit, highlight = TRUE)


#Show the trained final prediction model.
final_knn_fit$finalModel
optimal_k_on_entire_data_set <- final_knn_fit$finalModel$k
optimal_k_on_entire_data_set
