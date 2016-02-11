# LRTest for LR models, as part of the analysis of topicality influence for salience
# Check paper:
#  Naho Orita, Eliana Vornov, Naomi H Feldman, and Jordan Boyd-Graber. 2014.
#  Quantifying the role of discourse topicality in speakers' choices of referring expressions.
#  In Association for Computational Linguistics, Workshop on Cognitive Modeling and Computational Linguistics.

library(epiDisplay)
# read LR data 
data = read.csv("lr_models.csv",header=TRUE)
fit1 <- glm(Y~X1,data=data,family=binomial())
summary(fit1)

fit2 <- glm(Y~X1+X2,data=data,family=binomial(link = ))
summary(fit2)

lrtest(fit1,fit2)