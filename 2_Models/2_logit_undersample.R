library(caret)
library(mlbench)
library(randomForest)
library(e1071)
library(doParallel)
library(MLeval)
library(MLmetrics)


#Find out how many cores are available (if you don't already know)
cores<-detectCores()
#Create cluster with desired number of cores, leave one open for the machine         
#core processes
cl <- makeCluster(cores[1]-1)
#Register cluster
registerDoParallel(cl)

#Read the data set
data <- readRDS("~/masteroppgave2020/data16052020.rds")


sum(is.na(data$originality))

data <- subset(data, !is.na(data$claims))
data <- subset(data, !is.na(data$originality))

sum(is.na(data))

levels(data$Litigation) <- c(levels(data$Litigation), "Yes")
data$Litigation[data$Litigation == 1] <- "Yes"


levels(data$Litigation) <- c(levels(data$Litigation), "No")
data$Litigation[data$Litigation == 0] <- "No"


data$Litigation <- droplevels(data$Litigation)


data$tech_field <- as.factor(data$tech_field) 

sum(is.na(data))


# split training set off
split1 <- createDataPartition(data$Litigation, p = .7)[[1]]
test <- data[-split1,]
training <- data[split1,]

down_train <- downSample(x = training[, -11],
                         y = training$Litigation)
summary(down_train$Class)


control <- trainControl(method = "cv",
                        number = 5,
                        summaryFunction = twoClassSummary,
                        classProbs = TRUE,
                        savePredictions = TRUE
)

metric <- "ROC"

set.seed(123)


names(down_train)

glm_fit <- train(Class ~ tech_field+many_field+patent_scope+family_size +
                   grant_lag+bwd_cits+npl_cits+claims+
                   originality+radicalness+assignee_PV+
                   foreign_individual+foreign_corp+JPN_corp+JPN_individual+
                   US_individual+US_corp+foreign_priority_binary+PCT_binary+
                   lawyer_binary,
                 data = down_train, 
                 method = "glm",
                 family = "binomial",
                 metric = "ROC",
                 trControl = control
)

saveRDS(glm.fit, "./glm_fit_25052020")
print(glm.fit)
summary(glm.fit)

results.predicted <- predict(glm_fit, test)

results.predicted.probs <- predict(glm_fit, newdata = test, type = "prob")

confusionMatrix(data =results.predicted, test$Litigation)

plot(varImp(glm_fit))

# AUC and ROC
library(ROCR)


predictions_glm <- as.vector(results.predicted.probs$Yes)
pred_glm <- prediction(predictions_glm, test$Litigation)

perf_AUC=performance(pred_glm,"auc")
auc=perf_AUC@y.values[[1]]
auc
auc_glm <- auc


perf_ROC_glm=performance(pred_glm,"tpr","fpr") #plot the actual ROC curve
plot(perf_ROC, main="ROC plot") +
  
  text(0.5,0.5,paste("AUC = ",format(auc, digits=5, scientific=FALSE))) 