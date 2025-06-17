#### CASE STUDY 1

churn= read.csv("/Users/andresperez/Desktop/R Files/churn_project/data/ChurnData.csv", skipNul = TRUE)
churn$ID = NULL
str(churn)
summary(churn)

sum(churn$Churn == 1)
sum(churn$Churn == 0)

churn_1 = subset(churn, Churn == 1)
churn_0 = subset(churn, Churn == 0)

### UNDERSTANDING THE DATA
# Customer Months
plot(churn$Customer.Months, churn$Churn)
hist(churn_0$Customer.Months, col="lightblue")
hist(churn_1$Customer.Months, col="lightcoral")

summary(churn$Customer.Months[churn$Churn == 1])
summary(churn$Customer.Months[churn$Churn == 0])

# CHI.Score.Mon0
plot(churn$CHI.Score.Mon0, churn$Churn)
hist(churn_0$CHI.Score.Mon0,col="lightblue")
hist(churn_1$CHI.Score.Mon0, col="lightcoral")

summary(churn$CHI.Score.Mon0[churn$Churn == 1])
summary(churn$CHI.Score.Mon0[churn$Churn == 0])

# CHI.Score
hist(churn_0$CHI.Score,col="lightblue")
hist(churn_1$CHI.Score, col="lightcoral")
summary(churn$CHI.Score[churn$Churn == 1])
summary(churn$CHI.Score[churn$Churn == 0])

# Support.Cases.Mon0 
plot(churn$Support.Cases.Mon0, churn$Churn)
hist(churn_0$Support.Cases.Mon0, col="lightblue")
hist(churn_1$Support.Cases.Mon0, col="lightcoral")
summary(churn$Support.Cases.Mon0[churn$Churn == 1])
summary(churn$Support.Cases.Mon0[churn$Churn == 0])

# Support.Cases
plot(churn$Support.Cases, churn$Churn)
hist(churn_0$Support.Cases, col="lightblue")
hist(churn_1$Support.Cases, col="lightcoral")
summary(churn$Support.Cases[churn$Churn == 1])
summary(churn$Support.Cases[churn$Churn == 0])

# SP. Mon0 (Support Priority - each support case is given a support priority (SP) value)
plot(churn$SP.Mon0, churn$Churn)
hist(churn_0$SP.Mon0, col="lightblue")
hist(churn_1$SP.Mon0, col="lightcoral")
summary(churn$SP.Mon0[churn$Churn == 1])
summary(churn$SP.Mon0[churn$Churn == 0])

# SP
plot(churn$SP, churn$Churn)
hist(churn_0$SP, col="lightblue")
hist(churn_1$SP, col="lightcoral")
summary(churn$SP[churn$Churn == 1])
summary(churn$SP[churn$Churn == 0])

# Logins 
plot(churn$Logins, churn$Churn)
hist(churn_0$Logins, col="lightblue")
hist(churn_1$Logins, col="lightcoral")
summary(churn$Logins[churn$Churn == 1])
summary(churn$Logins[churn$Churn == 0])

# Blog Articles
plot(churn$Blog.Articles, churn$Churn)
hist(churn_0$Blog.Articles, col="lightblue")
hist(churn_1$Blog.Articles, col="lightcoral")
summary(churn$Blog.Articles[churn$Churn == 1])
summary(churn$Blog.Articles[churn$Churn == 0])

# Views
plot(churn$Views, churn$Churn)
filtered_views = churn[churn$Views != 230414.0, ]
plot(filtered_views$Views, filtered_views$Churn)

hist(churn_0$Views, col="lightblue")
hist(churn_1$Views, col="lightcoral")
summary(churn$Views[churn$Churn == 1])
summary(churn$Views[churn$Churn == 0])

# Days Since Last Login
plot(churn$Days.Since.Last.Login, churn$Churn)
filtered_days = churn[churn$Days.Since.Last.Login !=-648.000,]
plot(filtered_days$Days.Since.Last.Login, filtered_days$Churn)

hist(churn_0$Days.Since.Last.Login, col="lightblue")
hist(churn_1$Days.Since.Last.Login, col="lightcoral")
summary(churn$Days.Since.Last.Login[churn$Churn == 1])
summary(churn$Days.Since.Last.Login[churn$Churn == 0])


###############################################################################

### PREDICTIVE MODELS
library(caTools)
library(rpart)
library(rpart.plot)
library(randomForest)
library(caret)
library(e1071)
library(ROCR)

# BASELINE ACCURACY
table(churn$Churn)
baseline_accuracy = 5422/(nrow(churn))
baseline_accuracy

#### CLUSTERS TO UNDERSTAND CHARACTERISTICS OF CHURN VS NO CHURN
preproc = preProcess(churn)
churnNorm = predict(preproc,churn)
summary(churnNorm)
k=4
set.seed(123)
churn.KMC = kmeans(churnNorm,centers=k,iter.max=1000)
table(churn.KMC$cluster)
churn.KMC$centers

churnKMC.1 = subset(churn, churn.KMC$cluster==1)
churnKMC.2 = subset(churn, churn.KMC$cluster==2)
churnKMC.3 = subset(churn, churn.KMC$cluster==3)
churnKMC.4 = subset(churn, churn.KMC$cluster==4)

table(churn.KMC$cluster)
churn.KMC$centers

lapply(split(churn, churn.KMC$cluster),colMeans)
churnDF.KMC = as.data.frame(lapply(split(churn, churn.KMC$cluster),colMeans))
clusterNames = seq(1,4,1)
clusterNames = paste("Cluster",clusterNames,"")
names(churnDF.KMC) = clusterNames
setwd("/Users/andresperez/Desktop/churn_project")
write.csv(churnDF.KMC, "churn_4Clusters.csv")


### LOGISTIC RREGRESSION MODEL
set.seed(110)
split = sample.split(churn$Churn, SplitRatio=0.6)
churnTrain = subset(churn, split==TRUE)
churnTest = subset(churn, split==FALSE)

churn.glm1 = glm(Churn~CHI.Score + Logins + Views + Days.Since.Last.Login,
                 family=binomial, data=churnTrain)
summary(churn.glm1)

pred.glm1 = predict(churn.glm1, newdata=churnTest, type="response")
summary(pred.glm1)
table(churnTest$Churn, as.numeric(pred.glm1)>=0.07)
accuracy=(1841+50)/(nrow(churnTest))
accuracy
sensitivity=(50)/(66+50)
sensitivity
specificity=(1841)/(1841+328)
specificity

ROCRchurnPred1 = prediction(pred.glm1, churnTest$Churn)
ROCRperf1 = performance(ROCRchurnPred1, "tpr", "fpr")
plot(ROCRperf1, colorize=TRUE, print.cutoffs.at=seq(0,0.1,by=0.01), text.adj=c(-1,0.5))
AUC1 = as.numeric(performance(ROCRchurnPred1, "auc")@y.values)
AUC1

table(churnTrain$Churn, as.numeric(pred.glm.Train)>=0.07)
acc = (2744+72)/(nrow(churnTrain))
acc
sens = (72)/(103+72)
sens 
spec = (2744)/(2744+509)
spec


### CLASSIFICATION TREE
set.seed(110)
churn$Churn = factor(churn$Churn)
split = sample.split(churn$Churn, SplitRatio = 0.6)
TreeTrain = subset(churn, split == TRUE)
table(TreeTrain$Churn)
TreeTest = subset(churn, split == FALSE)

churn.tree = rpart(Churn~., method="class", data=TreeTrain)
prp(churn.tree)

pred.tree = predict(churn.tree, newdata=churnTest, type="class")
table(churnTest$Churn, pred.tree)
accuracy2 = (2159+9)/(nrow(churnTest))
accuracy2
sensitivity2 = (9)/(107+9)
sensitivity2
specificity2 = (2159)/(2159+10)
specificity2

pred.tree = predict(churn.tree, newdata=churnTest)
ROCRpred.tree = prediction(pred.tree[,2],churnTest$Churn)
ROCRperf.tree = performance(ROCRpred.tree,"tpr","fpr")
plot(ROCRperf.tree, colorize=TRUE, print.cutoffs.at=seq(0,0.11,by=0.01), text.adj=c(-1,0.5))
AUC2 =  as.numeric(performance(ROCRpred.tree, "auc")@y.values)
AUC2


#### CLASSSIFCATION TREE 2
set.seed(110)
churn$Churn = factor(churn$Churn)

churn.tree2 = rpart(Churn~CHI.Score + Logins + Views + Days.Since.Last.Login+Customer.Months, method="class", data=TreeTrain)
prp(churn.tree2)

pred.tree = predict(churn.tree, newdata=churnTest, type="class")
table(churnTest$Churn, pred.tree)
accuracy2 = (2159+10)/(nrow(churnTest))
accuracy2
sensitivity2 = (9)/(107+9)
sensitivity2
specificity2 = (2159)/(2159+10)
specificity2

pred.tree = predict(churn.tree, newdata=churnTest)
ROCRpred.tree = prediction(pred.tree[,2],churnTest$Churn)
ROCRperf.tree = performance(ROCRpred.tree,"tpr","fpr")
plot(ROCRperf.tree, colorize=TRUE, print.cutoffs.at=seq(0,0.11,by=0.01), text.adj=c(-1,0.5))
AUC2 =  as.numeric(performance(ROCRpred.tree, "auc")@y.values)
AUC2


# RANDOM FOREST 
set.seed(110)
churn.forest = randomForest(Churn~CHI.Score + Logins + Views + Days.Since.Last.Login,data=TreeTrain,ntree=200)
churn.forest.pred = predict(churn.forest, newdata=TreeTest,type="class")
table(TreeTest$Churn,churn.forest.pred)

tr.control = trainControl(method = "cv", number = 15)

cp.grid = expand.grid( .cp = (0:100)*0.001)
set.seed(110)
train(Churn ~., data=TreeTrain, method="rpart",trControl=tr.control, tuneGrid = cp.grid)

set.seed(110)
churn.tree.cp=train(Churn ~., data=TreeTrain, method="rpart",trControl=tr.control, tuneGrid = cp.grid)

best.tree1 = churn.tree.cp$finalModel
prp(best.tree1)


####### LOGISITIC REGRESSION WITH CLUSTERS

subTrain = churnTrain
subTrain$Churn = NULL
subTest = churnTest
subTest$Churn = NULL

preproc = preProcess(subTrain)
normTrain = predict(preproc,subTrain)
normTest = predict(preproc,subTest)
mean(normTrain$Views)
mean(normTest$Views)

summary(normTrain)
set.seed(110)
churnKMC = kmeans(normTrain,centers=3, iter.max = 10000)
table(churnKMC$cluster)

library(flexclust)
km.kcca = as.kcca(churnKMC,normTrain)
clusterTrain = predict(km.kcca)
clusterTest = predict(km.kcca,newdata=normTest)
str(clusterTest)
table(clusterTrain)
table(churnKMC$cluster)
table(clusterTest)

churnTrain1 = subset(churnTrain,clusterTrain==1)
churnTrain2 = subset(churnTrain,clusterTrain==2)
churnTrain3 = subset(churnTrain,clusterTrain==3)

churnTest1 = subset(churnTest,clusterTest==1)
churnTest2 = subset(churnTest,clusterTest==2)
churnTest3 = subset(churnTest,clusterTest==3)

mean(churnTrain1$Churn)
mean(churnTrain2$Churn)
mean(churnTrain3$Churn)

mean(churnTest1$Churn)
mean(churnTest2$Churn)
mean(churnTest3$Churn)

churnMod1 = glm(Churn~CHI.Score + Logins + Views + Days.Since.Last.Login,data=churnTrain1,family=binomial)
churnMod2 = glm(Churn~CHI.Score + Logins + Views + Days.Since.Last.Login,data=churnTrain2,family=binomial)
churnMod3 = glm(Churn~CHI.Score + Logins + Views + Days.Since.Last.Login,data=churnTrain3,family=binomial)
summary(churnMod1)
summary(churnMod2)
summary(churnMod3)

churnMod1.pred = predict(churnMod1,newdata=churnTest1,type="response")
churnMod2.pred = predict(churnMod2,newdata=churnTest2,type="response")
churnMod3.pred = predict(churnMod3,newdata=churnTest3,type="response")

table(churnTest1$Churn,churnMod1.pred>=0.06)
ROCRchurnClustPred1 = prediction(churnMod1.pred, churnTest1$Churn)
ROCRperfClust1 = performance(ROCRchurnClustPred1, "tpr", "fpr")
plot(ROCRperfClust1, colorize=TRUE, print.cutoffs.at=seq(0,0.1,by=0.01), text.adj=c(-1,0.5))
AUCclust1 = as.numeric(performance(ROCRchurnPred1, "auc")@y.values)
AUCclust1

accuracyChurn1 = (555+41)/(555+250+14+41)
accuracyChurn1
sensChurn1 = (41)/(14+41)
sensChurn1
specChurn1 = (555)/(555+250)
specChurn1



table(churnTest2$Churn,churnMod2.pred>=0.05)
ROCRchurnClustPred2 = prediction(churnMod2.pred, churnTest2$Churn)
ROCRperfClust2 = performance(ROCRchurnClustPred2, "tpr", "fpr")
plot(ROCRperfClust2, colorize=TRUE, print.cutoffs.at=seq(0,0.1,by=0.01), text.adj=c(-1,0.5))
AUCclust2 = as.numeric(performance(ROCRchurnClustPred2, "auc")@y.values)
AUCclust2

accuracyChurn2 = (680+9)/(680+139+37+9)
accuracyChurn2
sensChurn2 = (9)/(37+9)
sensChurn2
specChurn2 = (680)/(680+139)
specChurn2



table(churnTest3$Churn,churnMod3.pred>=0.04)
ROCRchurnClustPred3 = prediction(churnMod3.pred, churnTest3$Churn)
ROCRperfClust3 = performance(ROCRchurnClustPred3, "tpr", "fpr")
plot(ROCRperfClust3, colorize=TRUE, print.cutoffs.at=seq(0,0.1,by=0.01), text.adj=c(-1,0.5))
AUCclust3 = as.numeric(performance(ROCRchurnClustPred3, "auc")@y.values)
AUCclust3

accuracyChurn3 = (399+4)/(399+146+11+4)
accuracyChurn3
sensChurn3 = (4)/(11+4)
sensChurn3
specChurn3 = (399)/(399+146)
specChurn3

allPreds = c(churnMod1.pred,churnMod2.pred,churnMod3.pred)
allObs = c(churnTest1$Churn, churnTest2$Churn, churnTest3$Churn)



table(allObs, allPreds>=0.04)
accuracy.ALL = (571+92)/(571+1598+24+92)
accuracy.ALL
sens.ALL = (92)/(92+24)
sens.ALL
spec.ALL = (571)/(571+1598)
spec.ALL

EvalSet = read.csv("/Users/andresperez/Desktop/R Files/CASE STUDY/ChurnData_F23_evalset.csv", skipNul = TRUE)
eval.set = predict(churnMod1, newdata=EvalSet,type="response")
EvalSet$Prediction = eval.set
setwd("/Users/andresperez/Desktop/churn_project/data")
write.csv(EvalSet, "EvalSet.csv")
