# Practical Machine Learning: Course Project
# ===========================================

# The goal of your project is to predict the manner in which they did 
# the exercise. This is the "classe" variable in the training set. 
# You may use any of the other variables to predict with. You should create 
# a report describing how you built your model, how you used cross validation, 
# what you think the expected out of sample error is, and why you made the 
# choices you did. You will also use your prediction model to predict 
# 20 different test cases.

# ---------- SYSTEM SETUP AND LIBRARIES ----------

# -- SVH --
setwd("C:/Users/OKufirin/OneDrive - St Vincent's Health Australia/Practical Machine Learning")

# -- HOME --
setwd("D:/Data Science Specialisation/Practical Machine Learning/4/Course Project/")

library(caret)
library(xlsx)

# ------------------------------------------------

# downloading the trainig data
download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", 
              "training.csv")
# downloading the test data
download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", 
              "test.csv")

# getting the data
training_org <- read.csv("training.csv", header = T, na.strings = c("", "NA"))
test_org<- read.csv("test.csv", header = T, na.strings = c("", "NA"))
# checking dimensions
dim(training_org) # 19622x160
dim(test_org) # 20x160
# variables' names
names(training_org)

# checking if dataset is balanced
round(table(training_org$classe) / dim(training_org)[1] * 100, 2)
barplot(round(table(training_org$classe) / dim(training_org)[1] * 100, 2))
# the dataset is balanced

# checking the number of NAs for each column
colSums(is.na(training_org))
table(colSums(is.na(training_org)))
# all have either 0 or 19216 NAs

# extract column names with NAs
NA_cols <- colnames(training_org)[colSums(is.na(training_org)) > 0]
# 100 columns have large number NAs and should be excluded

# exclude NA columns
training <- training_org[, !(names(training_org) %in% NA_cols)]
test <- test_org[, !(names(test_org) %in% NA_cols)]

# exclude other non-sensor variables
colnames(training)
redundant_cols <- colnames(training)[1:7]
training <- training[, !(names(training) %in% redundant_cols)]
test <- test[, !(names(test) %in% redundant_cols)]

# convert outcome to a factor varaible
training$classe <- as.factor(training$classe)
str(training)

# checking correlation
correlationMatrix <- cor(training[,1:length(training)-1])
# write.xlsx(correlationMatrix, "correlationMatrix.xlsx", sheetName = "Corr",
#               col.names = T, row.names = T, append = F)

set.seed(7)

# ================================
# Penalized Multinomial Regression 
trControl <- trainControl(method = "cv", number = 5)
system.time(multinomReg_model <- train(classe ~ .,
                                       trControl = trControl,
                                       preProcess = c("center", "scale", "nzv"),
                                       method = "multinom",
                                       data = training, trace = FALSE))
# no PCA, no Corr,                      Accuracy 0.73,  Time 170.19 *
# PCA_thresh=0.9, Corr_cutoff=0.75,     Accuracy 0.5004 Time 114.90
# PCA_thresh=0.95,                      Accuracy 0.5329 Time 132.92
# PCA_thresh=0.8,                       Accuracy 0.4762 Time 86.72
# Corr_cutoff=0.9,                      Accuracy 0.7016 Time 149.28
# Corr_cutoff=0.75,                     Accuracy 0.5908 Time 127.81
# Corr_cutoff=0.5,                      Accuracy 0.4999 Time 110.90


# ================================
# kNN
trControl <- trainControl(method = "cv", number = 5)
system.time(knn_model <- train(classe ~ .,
                               trControl = trControl,
                               preProcess = c("center", "scale", "nzv"),
                               method = "knn",
                               tuneGrid = expand.grid(k = 1:1),
                               data = training))
# no PCA, no Corr,                      Accuracy 0.9925 Time 49.30 *
# PCA_thresh=0.9, Corr_cutoff=0.75,     Accuracy 0.9864 Time 19.19
# PCA_thresh=0.95,                      Accuracy 0.9892 Time 24.17
# PCA_thresh=0.8,                       Accuracy 0.981  Time 17.4
# Corr_cutoff=0.9,                      Accuracy 0.9912 Time 41.16
# Corr_cutoff=0.75,                     Accuracy 0.9886 Time 27.45
# Corr_cutoff=0.5,                      Accuracy 0.9835 Time 19.83

prediction <- data.frame(test$problem_id, predict(knn_model, 
                                                  test[, -ncol(test)]))
names(prediction) <- c("ID", "knn_Prd")
prediction



# ================================
# CART
trControl <- trainControl(method = "cv", number = 5,
                          preProcOptions = list(cutoff = 0.5, thresh = 0.8))
system.time(cart_model <- train(classe ~ .,
                                trControl = trControl,
                                preProcess = c("center", "scale", "nzv"),
                                method = "rpart",
                                data = training))
# no PCA, no Corr,                      Accuracy 0.5042 Time 14.98 *
# PCA_thresh=0.9, Corr_cutoff=0.75,     Accuracy 0.3344 Time 17.61
# PCA_thresh=0.95,                      Accuracy 0.3933 Time 20.04
# PCA_thresh=0.8,                       Accuracy 0.3632 Time 18.55
# Corr_cutoff=0.9,                      Accuracy 0.4732 Time 19.22
# Corr_cutoff=0.75,                     Accuracy 0.5033 Time 17.20
# Corr_cutoff=0.5,                      Accuracy 0.4638 Time 19.17



# ================================
# Bagged CART
trControl <- trainControl(method = "cv", number = 5,
                          preProcOptions = list(cutoff = 0.9))
system.time(baggedCART_model <- train(classe ~ .,
                                      trControl = trControl,
                                      preProcess = c("center", "scale", "nzv", "corr"),
                                      method = "treebag",
                                      data = training))
# no PCA, no Corr,                      Accuracy 0.9871 Time 99.34 
# PCA_thresh=0.9, Corr_cutoff=0.75,     Accuracy 0.9509 Time 59.69
# PCA_thresh=0.95,                      Accuracy 0.9633 Time 73.15
# PCA_thresh=0.8,                       Accuracy 0.9459 Time 46.94
# Corr_cutoff=0.9,                      Accuracy 0.9881 Time 92.61 *
# Corr_cutoff=0.75,                     Accuracy 0.9800 Time 65.89
# Corr_cutoff=0.5,                      Accuracy 0.9604 Time 27.84



# ================================
# Random Forest
trControl <- trainControl(method = "cv", number = 5)
system.time(rf_model <- train(classe ~ .,
                              trControl = trControl,
                              preProcess = c("center", "scale", "nzv"),
                              method = "rf",
                              tuneGrid = expand.grid(mtry = 1:1),
                              data = training))
# no PCA, no Corr,                      Accuracy 0.9909 Time 306.75 *
# PCA_thresh=0.9, Corr_cutoff=0.75,     Accuracy 0.9728 Time 132.52
# PCA_thresh=0.95,                      Accuracy 0.9772 Time 183.72
# PCA_thresh=0.8,                       Accuracy 0.9635 Time 93.36
# Corr_cutoff=0.9,                      Accuracy 0.9900 Time 265.90
# Corr_cutoff=0.75,                     Accuracy 0.9891 Time 191.56
# Corr_cutoff=0.5,                      Accuracy 0.9789 Time 139.48



# ================================
# Parallel Random Forest
trControl <- trainControl(method = "cv", number = 5)
system.time(parRF_model <- train(classe ~ .,
                                 trControl = trControl,
                                 preProcess = c("center", "scale", "nzv"),
                                 method = "parRF",
                                 data = training))
# no PCA, no Corr,                      Accuracy 0.9944 Time 1126.83 *
# PCA_thresh=0.8,                       Accuracy 0.9666 Time 321.48
# Corr_cutoff=0.9,                      Accuracy 0.9932 Time 1101.12
# Corr_cutoff=0.75,                     Accuracy 0.9930 Time 658.35
# Corr_cutoff=0.2,                      Accuracy 0.9387 Time 224.54



# ================================
# LVQ - Learning Vector Quantization
trControl <- trainControl(method = "cv", number = 5,
                          preProcOptions = list(cutoff = 0.5, thresh = 0.95))
system.time(lvq_model <- train(classe ~ .,
                               trControl = trControl,
                               preProcess = c("center", "scale", "nzv",
                                              "corr"),
                               method = "lvq",
                               data = training))
# Corr_cutoff=0.5,                      Accuracy 0.6286 Time 474.34 *
# Corr_cutoff=0.2,                      Accuracy 0.5523 Time 166.58



# ================================
# HHDA - High Dimensional Discriminant Analysis
trControl <- trainControl(method = "cv", number = 5,
                          preProcOptions = list(cutoff = 0.85,))
system.time(hdda_model <- train(classe ~ .,
                                trControl = trControl,
                                preProcess = c("center", "scale", "nzv", "corr"),
                                method = "hdda",
                                data = training))
# no PCA, no Corr,                      Accuracy 0.7264 Time 30.06
# PCA_thresh=0.9, Corr_cutoff=0.75,     Accuracy 0.6659 Time 29.56
# PCA_thresh=0.95,                      Accuracy 0.6825 Time 33.14
# PCA_thresh=0.8,                       Accuracy 0.5704 Time 31.52
# Corr_cutoff=0.95,                     Accuracy 0.7572 Time 31.69
# Corr_cutoff=0.9,                      Accuracy 0.7829 Time 30.33
# Corr_cutoff=0.85,                     Accuracy 0.7872 Time 29.96 *
# Corr_cutoff=0.8,                      Accuracy 0.7824 Time 30.19
# Corr_cutoff=0.75,                     Accuracy 0.7377 Time 27.85
# Corr_cutoff=0.5,                      Accuracy 0.6886 Time 26.75



# ================================
# Naive Bayes
trControl <- trainControl(method = "cv", number = 5,
                          preProcOptions = list(cutoff = 0.9))
system.time(nb_model <- train(classe ~ .,
                              trControl = trControl,
                              preProcess = c("center", "scale", "nzv", "corr"),
                              method = "nb",
                              data = training))
# no PCA, no Corr,                      Accuracy 0.7441 Time 205,85
# PCA_thresh=0.9, Corr_cutoff=0.75,     Accuracy 0.6079 Time 92.22
# PCA_thresh=0.95,                      Accuracy 0.6442 Time 114.59
# PCA_thresh=0.8,                       Accuracy 0.559  Time 72.03
# Corr_cutoff=0.95,                     Accuracy 0.7515 Time 209.41
# Corr_cutoff=0.9,                      Accuracy 0.7569 Time 173.68 *
# Corr_cutoff=0.75,                     Accuracy 0.7389 Time 135.23
# Corr_cutoff=0.5,                      Accuracy 0.6694 Time 101.38
# Corr_cutoff=0.2,                      Accuracy 0.5234 Time 49.91



# ================================
# LDA - Linear Discriminant Analysis
trControl <- trainControl(method = "cv", number = 5)
system.time(lda_model <- train(classe ~ .,
                               trControl = trControl,
                               preProcess = c("center", "scale", "nzv"),
                               method = "lda",
                               data = training))
# no PCA, no Corr,                      Accuracy 0.7018 Time 11.78 *
# PCA_thresh=0.9, Corr_cutoff=0.75,     Accuracy 0.5188 Time 11.59
# PCA_thresh=0.95,                      Accuracy 0.5272 Time 11.81
# PCA_thresh=0.8,                       Accuracy 0.4673 Time 12.11
# Corr_cutoff=0.95,                     Accuracy 0.6852 Time 12.77
# Corr_cutoff=0.9,                      Accuracy 0.6776 Time 11.96 
# Corr_cutoff=0.75,                     Accuracy 0.5838 Time 11.25
# Corr_cutoff=0.5,                      Accuracy 0.4976 Time 10.61



# ================================
# Example with Correlation and PCA: LDA - Linear Discriminant Analysis
# trControl <- trainControl(method = "cv", number = 5,
#                           preProcOptions = list(cutoff = 0.75, thresh = 0.95))
# system.time(lvq_model <- train(classe ~ .,
#                                trControl = trControl,
#                                preProcess = c("center", "scale", "nzv",
#                                               "corr", "pca"),
#                                method = "lvq",
#                                data = training))
