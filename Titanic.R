rm(list=ls(all=TRUE))


# define functions
f.usePackage = function(p) {
    if (!is.element(p, installed.packages()[,1])) {
        install.packages(p, dep = TRUE);
    }
    require(p, character.only = TRUE);
}

f.getTitle = function(data) {
    title.start = regexpr("\\,[A-Z ]{1,20}\\.", data, TRUE)
    title.end = title.start + attr(title.start, "match.length") - 1
    Title = substr(data, title.start+2, title.end-1)
    return (Title)
}

`%ni%` = Negate(`%in%`)

f.cutoff.optimizer = function(pred, y) {
    output.auc = vector()
    grid = seq(0.1, 0.99, by=0.01)
    for (cut.i in 1:length(grid)) {
        yhat = ifelse(pred >= grid[cut.i], 1, 0)
        result = prediction(yhat, y)
        perf = performance(result,"tpr","fpr")
        auc = performance(result,"auc")
        auc = unlist(slot(auc, "y.values"))
        output.auc = rbind(output.auc, auc)
    }   
    output = cbind(grid, output.auc)
    return(output)
}

f.logloss = function(actual, pred) {
    eps = 1e-15
    nr = nrow(pred)
    pred = matrix(sapply(pred, function(x) max(eps,x)), nrow = nr)      
    pred = matrix(sapply(pred, function(x) min(1-eps,x)), nrow = nr)
    ll = sum(actual*log(pred) + (1-actual)*log(1-pred))
    ll = -ll/nr      
    return(ll);
}

f.cutoff.optimizer = function(pred, y) {
    output.auc <- vector()
    grid = seq(0.1, 0.99, by=0.01)
    for (cut.i in 1:length(grid)) {
        yhat = ifelse(pred >= grid[cut.i], 1, 0)
        result = prediction(yhat, y)
        perf = performance(result,"tpr","fpr")
        auc = performance(result,"auc")
        auc = unlist(slot(auc, "y.values"))
        output.auc = rbind(output.auc, auc)
    }   
    output = cbind(grid=grid, auc=output.auc)
    return(output)
}

f.lasso.modeller = function(x.train, y.train, x.test, y.test){
    # Use cv.glmnet() to determine the best lambda for the lasso:
    lasso.cv = cv.glmnet(x=x.train, y=y.train, alpha=1, family='binomial', type.measure='auc', nfolds=10)
    bestlam.lasso = lasso.cv$lambda.min
    
    # Calculate the predicted outcome of the training and test sets:
    fitted = predict(lasso.cv, s=bestlam.lasso, newx=x.train, type='response')
    pred = predict(lasso.cv, s=bestlam.lasso, newx=x.test, type='response')
    
    y.train = as.numeric(as.character(y.train))
    y.test = as.numeric(as.character(y.test))
    
    # Calculate the log-loss of the test
    logloss = f.logloss(actual=y.test, pred=pred)
    
    # Find the optimal probability cutoff with highest accuracy
    output = f.cutoff.optimizer(fitted, y.train) 
    output = data.frame(output)
    OptimalCutoff = output[which.max(output[,2]),c("grid")]
    
    # Use the optimal probability cutoff to calcuate the accuracy of test data
    pred.binary = ifelse(pred>=OptimalCutoff, 1, 0)
    conf.table = confusionMatrix(table(pred=pred.binary, actual=y.test))
    pred.accuracy = as.numeric(conf.table$overall[1])
    #sens = as.numeric(conf.table$byClass[2])
    #spec = as.numeric(conf.table$byClass[1])
    #kappa = as.numeric(conf.table$overall[2])
    
    rm(pred, fitted, lasso.cv, bestlam.lasso, pred.binary, output)
    
    return(c(logloss, OptimalCutoff, pred.accuracy))
}
    
    
# load library
f.usePackage("data.table")
f.usePackage("glmnet")
f.usePackage("dplyr")
f.usePackage("pROC")
f.usePackage("ROCR")
f.usePackage("caret")

setwd("C:/Users/chang_000/Dropbox/Pinsker/CV_Resume/InterviewQuestions/Avant/avant-analytics-interview/")
#dir()

#============================================================
# Read data sets
data1 = fread("TitanicData.csv")
data2 = fread("TitanicData2.csv")
data1 = data.frame(data1)
data2 = data.frame(data2)

str(data1)
str(data2)
unique(data1$PassengerId)
unique(data2$PassengerId)

#============================================================
# Data manupulation
# data1
data1$Survived = as.factor(data1$Survived)
data1$Sex = as.factor(ifelse(data1$Sex=="male", 0, 1))
data1$Pclass = as.factor(data1$Pclass)

# data2
data2$Survived = as.factor(data2$Survived)
data2$Sex = as.factor(data2$Sex)
data2$Pclass = as.factor(data2$Pclass)
data2$Fare = as.numeric(gsub(",","",data2$Fare))

# merge the data sets together
variables_list = names(data1)[order(names(data1))]
data = rbind(data1[,variables_list], data2[,variables_list])
dim(data)
summary(data)

# remove the duplicates if there are any in the merged data set
data = data[!duplicated(data), ]
dim(data)


##################################################################
# extract some additional information from the `Name` variable
##################################################################
# data$Title
data$Title = f.getTitle(data$Name)
table(data$Title)

# reduce the levels of the "Title" variable
data$Title[data$Title %in% c("Mme","Mlle","Lady","Ms","the Countess", "Jonkheer")] = "OtherFemale"
data$Title[data$Title %in% c("Capt", "Don", "Major", "Sir","Col","Dr","Rev")] = "OtherMale"
data$Title = as.factor(data$Title)


#######################################
# detect and impute the missing values
#######################################
apply(is.na(data), 2, sum)


#######
# Age
#######
# impute the missing values of "Age" variable
for (i in 1:nrow(data)) {
    if (is.na(data$Age[i])) {
        data$Age[i] = median(data$Age[data$Title==data$Title[i] & data$Pclass==data$Pclass[i]], na.rm=TRUE) 
    } 
}

# Bin Age
obj = cut(data$Age, c(seq(0, 60, 10),Inf), labels=c(0:6))
data$Age_bin = as.factor(obj)

#######
# Fare
#######
data[data$Fare > 1000,]

# Adjust 2 outliers
for (i in 1:nrow(data)) {
    if (data$Fare[i] > 1000) {
        data$Fare[i] = median(data$Fare[data$Title==data$Title[i] & data$Pclass==data$Pclass[i] & data$Embarked==data$Embarked[i]], na.rm=TRUE) 
    } 
}

# log-transformation of Fare: log(Fare)
data$logFare = ifelse(data$Fare > 0, log(data$Fare), log(0.001))

###########
# Embarked
###########
table(data$Embarked) # S is the majority
data$Embarked[data$Embarked == ""] = "S" # impute missing value of Embarked
data$Embarked = as.factor(data$Embarked)


# save the new data set
write.csv(data,"mergeddata.csv", row.names=FALSE)


#####################################################################################
# Produce a glmnet model predicting the chance that a Titanic passanger survived.
#####################################################################################

# set up of K-fold CV for validation
K = 10
block = sample(1:K, nrow(data), replace=TRUE)

# Lasso models:
yvariable = c("Survived")
mod1.xvariables = c("Age","Fare","Embarked","Pclass","Sex","Title")
mod2.xvariables = c("Age_bin","Fare","Embarked","Pclass","Sex","Title")
mod3.xvariables = c("Age","logFare","Embarked","Pclass","Sex","Title")
mod4.xvariables = c("Age_bin","logFare","Embarked","Pclass","Sex","Title")

# initiate tables for the outputs
cv.logloss <- cv.OptimalCutoff <- cv.accuracy <- matrix(0, K, 4)


for (i in 1:K) {
    train.data = data[block!=i,] 
    test.data = data[block==i,] 
    
    #=========================================================
    # Model 1: Age + Embarked + Fare + Pclass + Sex + Title
    
    train = train.data[,c(yvariable,mod1.xvariables)]
    test = test.data[,c(yvariable,mod1.xvariables)]
    
    x.train = model.matrix(Survived~., train)[,-1]
    y.train = train$Survived
    
    x.test = model.matrix(Survived~., test)[,-1]
    y.test = test$Survived
    
    temp = f.lasso.modeller(x.train, y.train, x.test, y.test)
    
    cv.logloss[i,1] = temp[1]
    cv.OptimalCutoff[i,1] = temp[2]
    cv.accuracy[i,1] = temp[3]
    
    rm(train, test, x.train, y.train, x.test, y.test, temp)
    
    #=========================================================
    # Model 2: 
    train = train.data[,c(yvariable,mod2.xvariables)]
    test = test.data[,c(yvariable,mod2.xvariables)]
    
    x.train = model.matrix(Survived~., train)[,-1]
    y.train = train$Survived
    
    x.test = model.matrix(Survived~., test)[,-1]
    y.test = test$Survived
    
    temp = f.lasso.modeller(x.train, y.train, x.test, y.test)
    
    cv.logloss[i,2] = temp[1]
    cv.OptimalCutoff[i,2] = temp[2]
    cv.accuracy[i,2] = temp[3]
    
    rm(train, test, x.train, y.train, x.test, y.test, temp)
    
    #=========================================================
    # Model 3: 
    train = train.data[,c(yvariable,mod3.xvariables)]
    test = test.data[,c(yvariable,mod3.xvariables)]
    
    x.train = model.matrix(Survived~., train)[,-1]
    y.train = train$Survived
    
    x.test = model.matrix(Survived~., test)[,-1]
    y.test = test$Survived
    
    temp = f.lasso.modeller(x.train, y.train, x.test, y.test)
    
    cv.logloss[i,3] = temp[1]
    cv.OptimalCutoff[i,3] = temp[2]
    cv.accuracy[i,3] = temp[3]
    
    rm(train, test, x.train, y.train, x.test, y.test, temp)

    #=========================================================
    # Model 4: 
    train = train.data[,c(yvariable,mod4.xvariables)]
    test = test.data[,c(yvariable,mod4.xvariables)]
    
    x.train = model.matrix(Survived~., train)[,-1]
    y.train = train$Survived
    
    x.test = model.matrix(Survived~., test)[,-1]
    y.test = test$Survived
    
    temp = f.lasso.modeller(x.train, y.train, x.test, y.test)
    
    cv.logloss[i,4] = temp[1]
    cv.OptimalCutoff[i,4] = temp[2]
    cv.accuracy[i,4] = temp[3]
    
    rm(train, test, x.train, y.train, x.test, y.test, temp)

}

# average log-loss of the models
cv.logloss = data.frame(cv.logloss)
colnames(cv.logloss) = c("Model 1", "Model 2", "Model 3", "Model 4")
rownames(cv.logloss) = 1:K
cv.logloss

apply(cv.logloss, 2, mean)

# average test accuracy of the models
cv.accuracy = data.frame(cv.accuracy)
colnames(cv.accuracy) = c("Model 1", "Model 2", "Model 3", "Model 4")
rownames(cv.accuracy) = 1:K
cv.accuracy

apply(cv.accuracy, 2, mean)

# 
cv.OptimalCutoff = data.frame(cv.OptimalCutoff)
colnames(cv.OptimalCutoff) = c("Model 1", "Model 2", "Model 3", "Model 4")
rownames(cv.OptimalCutoff) = 1:K
cv.OptimalCutoff

apply(cv.OptimalCutoff, 2, mean)


######################################################################
# Model 3 has the smallest log-loss and largest accuracy on test set
# Build Model 3 using Full data
######################################################################
dataset = data[,c(yvariable,mod3.xvariables)]

x = model.matrix(Survived~., dataset)[,-1]
y = dataset$Survived

# Use cv.glmnet() to determine the best lambda for the lasso:
mod.lasso = cv.glmnet(x=x, y=y, alpha=1, family='binomial', type.measure='auc', nfolds=10)
(bestlam.lasso = mod.lasso$lambda.min)

plot(mod.lasso, main="lasso")

# Accuracy, Sensitivity, Specificity, and Kappa of the new model using the Full dataset
pred = predict(mod.lasso, s=bestlam.lasso, newx=x, type='response')
optimal.cutoff = apply(cv.OptimalCutoff, 2, mean)[3]
pred.binary = ifelse(pred>=optimal.cutoff, 1, 0)
conf.table = confusionMatrix(table(pred=pred.binary, actual=y))


result = cbind(accuracy = as.numeric(conf.table$overall[1]), 
               sensitivity = as.numeric(conf.table$byClass[2]),
               specificity = as.numeric(conf.table$byClass[1]),
               kappa = as.numeric(conf.table$overall[2]))

result = data.frame(result)
rownames(result) = c("Model 1")
result


# Grep the variables which have non-zero coefficients in the lasso
c = coef(mod.lasso, s=bestlam.lasso, exact=TRUE)
inds = which(c!=0)
(variables = row.names(c)[inds])

result.coef = data.frame(cbind(var=variables, coef=c@x))
result.coef = cbind(result.coef, exp(c@x)-1)
colnames(result.coef) = c("var","coef","exp(coef) - 1")
result.coef


#######################################################################
# Adding Gender:Pclass terms to Model 3
#######################################################################

dataNew = data
dataNew$Male_Pclass1 = as.factor(ifelse((data$Sex==0 & data$Pclass==1),1,0))
dataNew$Male_Pclass2 = as.factor(ifelse((data$Sex==0 & data$Pclass==2),1,0))
dataNew$Male_Pclass3 = as.factor(ifelse((data$Sex==0 & data$Pclass==3),1,0))
dataNew$Female_Pclass1 = as.factor(ifelse((data$Sex==1 & data$Pclass==1),1,0))
dataNew$Female_Pclass2 = as.factor(ifelse((data$Sex==1 & data$Pclass==2),1,0))
dataNew$Female_Pclass3 = as.factor(ifelse((data$Sex==1 & data$Pclass==3),1,0))

mod3a.xvariables = c("Age","logFare","Embarked","Pclass","Sex","Title","Male_Pclass1",
                     "Male_Pclass2","Male_Pclass3","Female_Pclass1","Female_Pclass2",
                     "Female_Pclass3")
yvariable = c("Survived")

dataset = dataNew[,c(yvariable,mod3a.xvariables)]

# set up of K-fold CV for validation
K = 10
block = sample(1:K, nrow(dataset), replace=TRUE)

# initiate tables for the outputs
cv.result <- matrix(0, K, 3)

for (i in 1:K) {
    train.data = dataset[block!=i,] 
    test.data = dataset[block==i,] 
    
    #=========================================================
    # Model 3a: Age+logFare+Embarked+Pclass+Sex+Title+Male_Pclass1+Male_Pclass2+Male_Pclass3
    #           +Female_Pclass1+Female_Pclass2+Female_Pclass3
    
    train = train.data[,c(yvariable,mod3a.xvariables)]
    test = test.data[,c(yvariable,mod3a.xvariables)]
    
    x.train = model.matrix(Survived~., train)[,-1]
    y.train = train$Survived
    
    x.test = model.matrix(Survived~., test)[,-1]
    y.test = test$Survived
    
    temp = f.lasso.modeller(x.train, y.train, x.test, y.test)
    
    cv.result[i,1] = temp[1]
    cv.result[i,2] = temp[2]
    cv.result[i,3] = temp[3]
    
    rm(train.data, test.data, x.train, y.train, x.test, y.test, temp)
}

# output
cv.result = data.frame(cv.result)
colnames(cv.result) = c("log-loss", "optimal.cutoff", "test.accuracy")
rownames(cv.result) = 1:K
cv.result

apply(cv.result, 2, mean)


# Build Model 3a using the Full data
x = model.matrix(Survived~., dataset)[,-1]
y = dataset$Survived

mod.lasso = cv.glmnet(x=x, y=y, alpha=1, family='binomial', type.measure='auc', nfolds=10)
(bestlam.lasso = mod.lasso$lambda.min)

# Predict using Model 3a
pred = predict(mod.lasso, s=bestlam.lasso, newx=x, type='response')

# Calculate the log-loss of Model 3a
f.logloss(as.numeric(as.character(y)), pred) 

# Calcuate the accuracy of Model 3a
optimal.cutoff = apply(cv.result, 2, mean)[2]
pred.binary = ifelse(pred>=optimal.cutoff, 1, 0)
conf.table = confusionMatrix(table(pred=pred.binary, actual=y))

result.3a = cbind(accuracy = as.numeric(conf.table$overall[1]), 
                  sensitivity = as.numeric(conf.table$byClass[2]),
                  specificity = as.numeric(conf.table$byClass[1]),
                  kappa = as.numeric(conf.table$overall[2]))
result.3a = data.frame(result.3a)
rownames(result.3a) = c("Model_3a")
result.3a = rbind(result, result.3a)
result.3a
