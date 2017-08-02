library("alr4")
require(alr4)
library("lmtest")
require(lmtest)
library("xtable")
require(xtable)
library("ggplot2")
require(ggplot2)
library("gridExtra")
require(gridExtra)
library("MASS")
require(MASS)
library("scales")
require(scales)
darden <- read.csv("darden.csv", header = TRUE)

darden <- darden[,c(3, 2, 4)]

#CHANGING CATEGORICAL
darden$Inside.Outside.Footprint <- ifelse(darden$Inside.Outside.Footprint == "Inside", 1, 0)
names(darden) <- c("HHAccount", "TotalHH", "InOutFootprint")

#PLOTTING COR/MARGINALS

#TRANSFORMATIONS
lm1 <- lm(log(HHAccount) ~ log(TotalHH), data = darden)
summary(lm1)
residualPlots(lm1, main = "Residual Plots")
bptest(lm1)
QQ1 <- ggplot(darden, aes(sample = HHAccount)) + stat_qq() + xlab("Normal Quantiles") +  ylab("Residuals") + ggtitle("HHAccounts QQ")
QQ1
QQ2 <- ggplot(darden, aes(sample = TotalHH)) + stat_qq() + xlab("Normal Quantiles") +  ylab("Residuals") + ggtitle("TotalHH QQ")
QQ2
grid.arrange(QQ1, QQ2)

qqPlot(lm(log(HHAccount)~log(TotalHH) + InOutFootprint, data = darden)$residuals, distribution = "norm", main = "Log-log Transformation: Normal QQ-Plot", xlab = "Normal Quantiles", ylab = "Residuals")

#BUILDING THE MODEL
lm2 <- lm(log(HHAccount) ~ log(TotalHH) + InOutFootprint, data = darden)
xtable(summary(lm2))
bptest(lm2)
residualPlots(lm2)
resettest(lm2, power = 2, type = "fitted")
AIC(lm2)
BIC(lm2)

lm3 <- lm(log(HHAccount) ~ log(TotalHH), data = darden)
xtable(summary(lm3))
bptest(lm3)
resettest(lm3)
residualPlots(lm3)
AIC(lm3)
BIC(lm3)
