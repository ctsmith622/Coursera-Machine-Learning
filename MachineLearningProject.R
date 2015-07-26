# read in the data
train <- read.csv("~/Coursera/Machine Learning/training.csv", stringsAsFactors = F)

# check for NA values and anything else weird
summary(train)

# looks like some values are divided by zero and some are empty
# define vectors with the indices of these values so we can remove them.
div.zero <-  apply(train, 2, function(x) which(x == "#DIV/0!"))
empty <- apply(train, 2, function(x) which(x == ""))

# for each column, remove the aforementioned indices
cols <- 1:ncol(train)
for(i in cols){
  train[,i][div.zero[[i]]] <- NA
  train[,i][empty[[i]]] <- NA
}

# look at the data again
summary(train)

# find the percent of NA values in each column
num.NA.by.column <- apply(train, MARGIN = 2, function(x) length(which(is.na(x))))
percent.NA.by.column <- sapply(num.NA.by.column, function(x) x / nrow(train))

# remove all the columns that have more than 50% NA values
good.cols <- which(percent.NA.by.column < 0.5)
train <- train[,good.cols]

#find how many instances that leaves us with
length(which(complete.cases(train)))

#all remaining instances are complete
train <- train[which(complete.cases(train)), ]

str(train)
train$classe <- as.factor(train$classe)

#remove the columns that are text because they have too many levels to be treated as factors in a random forest
summary(train)
bad.cols <- which(colnames(train) %in% c("user_name", "new_window", "cvtd_timestamp"))
train <- train[,-bad.cols]

#generate 10 folds for cross validation
library(randomForest)
library(caret)
cv.folds <- createFolds(y = 1:nrow(train))
resp.col <- which(colnames(train) == "classe")

for (i in 1:length(cv.folds)){
  this.test <- train[cv.folds[[i]], ]
  this.train <- train[-cv.folds[[i]], ]
  this.randomforest <- randomForest(x = this.train[,-resp.col], y = this.train[,resp.col], 
                                    xtest = this.test[,-resp.col], ytest = this.test[,resp.col], keep.forest = T)
  
  assign(x = paste("rf" , i , sep = ""), value = this.randomforest, envir = .GlobalEnv)
  print(i)
}

#build a list of all 9 randomforests
rf.list <- list(rf1, rf2, rf3, rf4, rf5, rf6, rf7, rf8, rf9)

# for each, find the test set error for each class
sapply(rf.list, function(x) x$test$confusion[,6])

#since all 9 fold had 0% error for all of the classes, the predicted misclassification rate for new data is also 0%.
