library(caret)
library(ranger)

set.seed(123)

##### LOAD data
data <- readRDS("data16052020.rds")

data <- na.omit(data)
sum(is.na(data))

levels(data$Litigation) <- c(levels(data$Litigation), "Yes")
data$Litigation[data$Litigation == 1] <- "Yes"

levels(data$Litigation) <- c(levels(data$Litigation), "No")
data$Litigation[data$Litigation == 0] <- "No"

data$Litigation <- droplevels(data$Litigation)

data$many_field <- as.factor(data$many_field)
data$tech_field <- as.factor(data$tech_field)
data$assignee_PV <- as.factor(data$assignee_PV)
data$foreign_corp <- as.factor(data$foreign_corp)
data$foreign_individual <- as.factor(data$foreign_individual)
data$JPN_corp <- as.factor(data$JPN_corp)             
data$JPN_individual <- as.factor(data$JPN_individual)
data$US_individual <- as.factor(data$US_individual) 
data$US_corp <- as.factor(data$US_corp)
data$foreign_priority_binary<- as.factor(data$foreign_priority_binary)
data$PCT_binary <- as.factor(data$PCT_binary)
data$lawyer_binary <- as.factor(data$lawyer_binary)

summary(data$Litigation)
str(data)

# split training set off
split1 <- createDataPartition(data$Litigation, p = .7)[[1]]
test <- data[-split1,]
training <- as.data.frame(data[split1,])

# Downsample amount of non-litigated companies
training <- downSample(x = training[, -11],
                       y = training$Litigation)
summary(training$Class)


fit_control <- trainControl(
  method = 'repeatedcv',                   # k-fold cross validation
  number = 5,                      # number of folds
  repeats = 5,
  savePredictions = 'final',       # saves predictions for optimal tuning parameter
  classProbs = T,                  # should class probabilities be returned
  summaryFunction=twoClassSummary, #  R= ROC
) 

# define a grid of parameter options to try
rf_grid <- expand.grid(mtry = c(4, 5, 6, 7),
                       splitrule = c("gini", "extratrees"),
                       min.node.size = c(1, 3, 5, 7, 10))

#Best from the grid above
rf_grid <- expand.grid(mtry = c(5),
                       splitrule = c("gini"),
                       min.node.size = c(7))

rf_grid

library(tictoc)

tic()
# fit a random forest model (using ranger)
rf_fit.4 <- train(Class ~ tech_field+many_field+patent_scope+family_size+grant_lag+bwd_cits+npl_cits+claims
                  +originality+radicalness+assignee_PV+num_assignees_PV+foreign_individual+foreign_corp+JPN_corp+JPN_individual         
                  +US_individual+US_corp+foreign_priority_binary+PCT_binary+lawyer_binary, 
                  data = training, 
                  method = "ranger",
                  trControl = fit_control,
                  tuneGrid = rf_grid,
                  num.trees = 200,
                  importance = "permutation",
                  metric = "ROC"
)

toc()

rf_fit$bestTune

# predict the outcome on a test set
rf_pred <- predict(rf_fit, test)

library(ggplot2)

# compare predicted outcome and true outcome
confusionMatrix(rf_pred, test$Litigation, positive='Yes')

varimpRF <- ggplot(varImp(rf_fit)) + theme_half_open() + background_grid()

varimpRF

ggplot(varImp(rf_fit, type = 2))


rf_pred.probs <- predict(rf_fit, newdata = test, type = "prob")


### FUNGERENDE AUC OG ROC ###
library(ROCR)

predictions_rf <- as.vector(rf_pred.probs$Yes)
pred_rf <- prediction(predictions_rf, test$Litigation)

perf_AUC=performance(pred_rf,"auc")
auc=perf_AUC@y.values[[1]]
auc
auc_rf <-auc 


perf_ROC_RF=performance(pred_rf,"tpr","fpr") #plot the actual ROC curve
plot(perf_ROC, main="ROC plot") +
  
  text(0.5,0.5,paste("AUC = ",format(auc, digits=5, scientific=FALSE))) 
#  abline(a = 0, b = 1)


#GGplot ROC
library(ggplot2)
library(cowplot)



df <- data.frame(Curve=as.factor(rep(c(1), each=length(perf_ROC@x.values[[1]]))), 
                 FalsePositive=c(perf_ROC@x.values[[1]],perf_ROC@x.values[[1]]),
                 TruePositive=c(perf_ROC@y.values[[1]],perf_ROC@y.values[[1]]))
plt_rf <- ggplot(df, aes(x=FalsePositive, y=TruePositive, color=Curve)) + 
  geom_line() + 
  labs(title= "ROC curve", 
       x = "False Positive Rate (1-Specificity)", 
       y = "True Positive Rate (Sensitivity)") + #label_value(auc_rf) + 
  theme_minimal_grid(12)

print(plt_rf)


stopCluster(cl)


# Load required packages
library(pdp)

p1 <- partial(rf_fit, pred.var = "bwd_cits", plot = TRUE, rug = TRUE, type = "classification", 
              plot.engine = "ggplot2")

p2 <- partial(rf_fit, pred.var = "tech_field", plot = TRUE, rug = TRUE, type = "classification",
              plot.engine = "ggplot2")

p3 <- partial(rf_fit, pred.var = "foreign_priority_binary", plot = TRUE, rug = TRUE, type = "classification",
              plot.engine = "ggplot2")

p4 <- partial(rf_fit, pred.var = "claims", plot = TRUE, rug = TRUE, type = "classification",
              plot.engine = "ggplot2")

p5 <- partial(rf_fit, pred.var = "US_corp", plot = TRUE, rug = TRUE, type = "classification",
              plot.engine = "ggplot2")

p6 <- partial(rf_fit, pred.var = "npl_cits", plot = TRUE, rug = TRUE, type = "classification",
              plot.engine = "ggplot2")

p7 <- partial(rf_fit, pred.var = "JPN_corp", plot = TRUE, rug = TRUE, type = "classification",
              plot.engine = "ggplot2")

p8 <- partial(rf_fit, pred.var = "originality", plot = TRUE, rug = TRUE, type = "classification",
              plot.engine = "ggplot2")

p9 <- partial(rf_fit, pred.var = "foreign_corp", plot = TRUE, rug = TRUE, type = "classification",
              plot.engine = "ggplot2")

p10 <- partial(rf_fit, pred.var = "US_individual", plot = TRUE, rug = TRUE, type = "classification",
               plot.engine = "ggplot2")

p11 <- partial(rf_fit, pred.var = "many_field", plot = TRUE, rug = TRUE, type = "classification",
               plot.engine = "ggplot2")

p12 <- partial(rf_fit, pred.var = "patent_scope", plot = TRUE, rug = TRUE, type = "classification",
               plot.engine = "ggplot2")

p13 <- partial(rf_fit, pred.var = "family_size", plot = TRUE, rug = TRUE, type = "classification",
               plot.engine = "ggplot2")

p14 <- partial(rf_fit, pred.var = "grant_lag", plot = TRUE, rug = TRUE, type = "classification",
               plot.engine = "ggplot2")

p15 <- partial(rf_fit, pred.var = "radicalness", plot = TRUE, rug = TRUE, type = "classification",
               plot.engine = "ggplot2")

p16 <- partial(rf_fit, pred.var = "assignee_PV", plot = TRUE, rug = TRUE, type = "classification",
               plot.engine = "ggplot2")

p17 <- partial(rf_fit, pred.var = "num_assignees_PV", plot = TRUE, rug = TRUE, type = "classification",
               plot.engine = "ggplot2")

p18 <- partial(rf_fit, pred.var = "foreign_individual", plot = TRUE, rug = TRUE, type = "classification",
               plot.engine = "ggplot2")

p19 <- partial(rf_fit, pred.var = "JPN_individual", plot = TRUE, rug = TRUE, type = "classification",
               plot.engine = "ggplot2")

p20 <- partial(rf_fit, pred.var = "PCT_binary", plot = TRUE, rug = TRUE, type = "classification",
               plot.engine = "ggplot2")

p21 <- partial(rf_fit, pred.var = "lawyer_binary", plot = TRUE, rug = TRUE, type = "classification",
               plot.engine = "ggplot2", color = "yhat", palette = "jco")


# Combination plots
library(ggsci)
grid1 <- grid.arrange(p13 + theme_half_open() + background_grid(), p5 + theme_half_open() + background_grid(), p1 + theme_half_open()+ background_grid(), p3 + theme_half_open() + background_grid(), ncol = 2)
grid2 <- grid.arrange(p7+ theme_half_open() + background_grid(), p4+ theme_half_open() + background_grid(), p6+ theme_half_open() + background_grid(), p10+ theme_half_open() + background_grid(), ncol = 2)  

grid3 <- grid.arrange(p5+ theme_half_open() + background_grid(), 
                      p8+ theme_half_open() + background_grid(), 
                      p9+ theme_half_open() + background_grid(), 
                      p11+ theme_half_open() + background_grid(), 
                      p12+ theme_half_open() + background_grid(), 
                      p14+ theme_half_open() + background_grid(), 
                      p15+ theme_half_open() + background_grid(), 
                      p16+ theme_half_open() + background_grid(), 
                      p17+ theme_half_open() + background_grid(), 
                      p18+ theme_half_open() + background_grid(), 
                      p19+ theme_half_open() + background_grid(), 
                      p20+ theme_half_open() + background_grid(), 
                      p21+ theme_half_open() + background_grid(), ncol = 2)  



p13 +  scale_fill_jco() + theme_classic()
p13 + theme_cowplot()
p21 + theme_bw() + scale_fill_jco()
p13 + theme_minimal()
p13 + theme_minimal_grid()

p2 + theme_half_open() + background_grid()


grid3 <- grid.aranger


save_plot("PDPgrid1.pdf", grid1)
save_plot("PDPgrid2.pdf", grid2)
save_plot("PDPtechfield.pdf", p2)


