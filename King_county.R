library(pacman)
library(readxl)
library(dplyr)
library(ggplot2)
library(ggiraph)
library(ggiraphExtra)
library(plyr)
library(caret)
#install.packages("move")
#library(move)

#Fetching the Data File
data_raw <- read.csv(file.choose())

#Data Exploration
class(data_raw)
head(data_raw)
str(data_raw)
glimpse(data_raw)
data_clean <- data_raw

#Summary of Data
summary(data_clean)

#Changing Date to yymmdd
data_clean$date <- substr(data_clean$date, 1, 8)
data_clean$date <- as.numeric(as.character(data_clean$date))
head(data_clean)
str(data_clean)

#Checking NA Values
length(which(is.na(data_clean)))

#Removing ID column
data_clean$id <- NULL
data_clean$date <- NULL
#data Visualization
ggplot(data = data_clean, aes(x = sqft_living, y = price)) + geom_point() +ggtitle("Prices According to Square feet")
ggplot(data = data_clean, aes(x = bathrooms, y = price)) + geom_point() +ggtitle("Prices According to Bathrooms")

ggplot(data = data_clean, aes(x=waterfront, y = price,fill=waterfront)) + geom_point()+ggtitle("Prices According to WaterFront")

#checking skewness in our variables and adjusting those which add value to the prediction
#install.packages("moments")
library(moments)
apply(data_clean[,1:19], 2, skewness, na.rm =TRUE)

data_clean$price <- log(data_clean$price)
data_clean$sqft_lot <- log(data_clean$sqft_lot)
data_clean$sqft_lot15 <- log(data_clean$sqft_lot15)

#finding correlation and checking which variables have positive and negative impact on Price
library(corrplot)
library(GGally)
library(ggcorrplot)
library(corrr)
correlationplot <- ggcorr(data_clean[, 1:19], geom = "blank", label = TRUE, hjust = 0.75) +
  geom_point(size = 10, aes(color = coefficient > 0, alpha = abs(coefficient) > 0.5)) +
  scale_alpha_manual(values = c("TRUE" = 0.25, "FALSE" = 0)) +
  guides(color = FALSE, alpha = FALSE)
correlationplot

CorrelationResults = cor(data_clean)
corrplot(CorrelationResults)

#Taking data in train and test sets
set.seed(1234)
samp <- sample(nrow(data_clean),0.75*nrow(data_clean))
train <- data_clean[samp,]
test <- data_clean[-samp,]

#Applying linear regression model on all variables to check significance of each variable
model <- lm(data = train, price ~ .)
summary(model)

#predicting prices for reduced model
pred_log_prob_full<-predict(model, newdata = test, type = 'response')

#finding RMSE(root mean square error) less the value more better the model and R2 to check accuracy
RMSE(pred_log_prob_full,test$price)
R2(pred_log_prob_full,test$price)

#forward selection method
frwd_model<-step(model,direction = 'forward')
Null_to_full<-lm(price ~ bedrooms + bathrooms + sqft_living + sqft_lot + floors + 
                      waterfront + view + condition + grade + sqft_above + sqft_basement + 
                      yr_built + yr_renovated + zipcode + lat + long + sqft_living15 + 
                      sqft_lot15, data=train)
summary(Null_to_full)

#backward selection method
bckd_model<-step(model,direction = 'backward')
reduced_model<-lm(price ~ bedrooms + bathrooms + sqft_living + sqft_lot + floors + 
                    waterfront + view + condition + grade + yr_built + yr_renovated + 
                    zipcode + lat + long + sqft_living15 + sqft_lot15, data=train)
summary(reduced_model)

#plotting the reduced model to check normality and homoscidastisity
par(mfrow=c(2,2))
plot(reduced_model)

#predicting prices for reduced model
pred_log_prob<-predict(reduced_model, newdata = test, type = 'response')

#finding RMSE(root mean square error) less the value more better the model and R2 to check accuracy
RMSE(pred_log_prob,test$price)
R2(pred_log_prob,test$price)


#decision tree
library(rpart)
library(rpart.plot)

reg<-rpart(price ~ bedrooms + bathrooms + sqft_living + sqft_lot + floors + 
             waterfront + view + condition + grade + yr_built + yr_renovated + 
             zipcode + lat + long + sqft_living15 + sqft_lot15, data=train)

summary(reg)

#Predicting prices of decision tree
pred_tree<-predict(reg,newdata = test)

#finding RMSE(root mean square error) less the value more better the model and R2 to check accuracy
RMSE(pred_tree,test$price)
R2(pred_tree,test$price)

rpart.plot(reg, box.palette="RdBu", shadow.col="gray", nn=TRUE)


#using random forest
library(randomForest)
set.seed(123)

#var.predict<-paste(names(train)[-19],collapse="+")
#rf.form <- as.formula(paste(names(train)[19], var.predict, sep = " ~ "))

rndm_frst<-randomForest(price ~ bedrooms + bathrooms + sqft_living + sqft_lot + floors + 
                          waterfront + view + condition + grade + yr_built + yr_renovated + 
                          zipcode + lat + long + sqft_living15 + sqft_lot15, data=train)
print(rndm_frst)
#summary(rndm_frst)

#finding importance of each variable in the model
imp<-importance(rndm_frst)
varImpPlot(rndm_frst)

#Predicting values of applied Random Forest model
pred_rndm<-predict(rndm_frst,newdata = test)

#finding RMSE(root mean square error) less the value more better the model and R2 to check accuracy
RMSE(pred_rndm,test$price)
R2(pred_rndm,test$price)


#using gradient boosting
library(caret)
#install.packages("gbm")
require(gbm)
require(MASS)
set.seed(123)

fitControl <- trainControl(method = "cv", number = 50)
tune_Grid <-  expand.grid(interaction.depth = 2, n.trees = 500, shrinkage = 0.1, n.minobsinnode = 10)
grdnt_bstng<-train(price ~ bedrooms + bathrooms + sqft_living + sqft_lot + floors + 
                     waterfront + view + condition + grade + yr_built + yr_renovated + 
                     zipcode + lat + long + sqft_living15 + sqft_lot15, data=train,method='gbm',trControl = fitControl, verbose = FALSE)
print(grdnt_bstng)
#grdnt_bstng<-gbm(price ~ bedrooms + bathrooms + sqft_living + sqft_lot + waterfront + 
#      view + condition + grade + sqft_above + yr_built + yr_renovated + 
#      zipcode + lat + long + sqft_living15 + sqft_lot15,data = train,distribution = "gaussian",n.trees = 10000,
#    shrinkage = 0.01, interaction.depth = 4)
#summary(grdnt_bstng)

#Predicting values of applied Gradient Boosting model
pred_grd<-predict(grdnt_bstng,newdata = test)

#finding RMSE(root mean square error) less the value more better the model and R2 to check accuracy
RMSE(pred_grd,test$price)
R2(pred_grd,test$price)
