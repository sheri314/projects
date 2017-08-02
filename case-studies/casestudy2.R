library("ggplot2")
require(ggplot2)
library("lmtest")
require(lmtest)
library("alr4")
require(alr4)
#library("caret")
#require(caret)
library("MASS")
require(MASS)

bellydf <- read.csv("bellydata.csv", header = TRUE)
#attempted to analyze d
#ay of week, but seems like every day is tuesday
#only one year, months may june july
bellydf$Date <- as.Date(bellydf$Date, '%m/%d/%y')
bellydf$month <- as.numeric(format(bellydf$Date, format = "%m"))
bellydf <- bellydf[,c(-1)]
bellydf <- bellydf[,c(3, 1,2,4:12)]
bellydf$demoCount<- bellydf$Demo + bellydf$Demo1.3 + bellydf$Demo4.5

unique(bellydf$Store)

#PRELIMINARY WORK
unique(bellydf$Region)
unique(bellydf$month)
unique(bellydf$year)
salesByState <- ggplot(na.omit(bellydf), aes(x = month, y = Units.Sold)) + geom_bar(stat="identity") + facet_wrap(~ Region)
(salesByState)
#pairs( ~ Units.Sold + Average.Retail.Price, data = bellydf)
cor(bellydf[,c(1,4:12)])
plot(bellydf$Units.Sold, bellydf$Average.Retail.Price)

#first model
lmfull <- lm(Units.Sold ~ Average.Retail.Price + Sales.Rep + Endcap + Demo + Demo1.3 + Demo4.5 + Natural + Fitness + month, data = bellydf)
summary(lmfull)
step(lmfull)  #removed all insig parameters

lmsignif <- lm(Units.Sold ~ Average.Retail.Price + Sales.Rep + Endcap + Demo + Demo1.3 + Demo4.5, data = bellydf)
summary(lmsignif)
BIC(lmsignif)

Anova(lmsignif)

residualPlots(lmsignif)

bptest(lmsignif)
lmsignif1 <- lm(log(Units.Sold) ~ Average.Retail.Price + Sales.Rep + Endcap + Demo + Demo1.3 + Demo4.5, data = bellydf)

resettest(lmsignif, power = 2 , type = 'fitted')

lmsignif2 <- lm(Units.Sold ~ (Average.Retail.Price + Sales.Rep + Endcap + Demo + Demo1.3 + Demo4.5)^2, data = bellydf)
summary(lmsignif2)

lmsignif3 <- lm(Units.Sold ~ Average.Retail.Price + Sales.Rep + Endcap + Demo + Demo1.3 + Demo4.5 + Sales.Rep * Endcap, data = bellydf)
summary(lmsignif3)
residualPlots(lmsignif3)
bptest(lmsignif3) #pass
resettest(lmsignif3, power = 2 , type = 'fitted') #pass
BIC(lmsignif3)

lmsignif4 <- lm(Units.Sold ~ Average.Retail.Price + Sales.Rep + Endcap + demoCount + Sales.Rep * Endcap, data = bellydf)
summary(lmsignif4)
residualPlots(lmsignif4)
bptest(lmsignif4)
resettest(lmsignif4, power = 2, type = 'fitted')

################################################################################
set.seed(123)
train_indices <- sample(seq_len(nrow(bellydf)), size = nrow(bellydf)/2)
trainingSet <- bellydf[train_indices,]
testSet <- bellydf[-train_indices,]

lmfulltest <- lm(Units.Sold ~ Average.Retail.Price + Sales.Rep + Endcap + demoCount + Sales.Rep * Endcap , data = trainingSet)
summary(lmfulltest)
bptest(lmfulltest)
residualPlots(lmfulltest)
resettest(lmfulltest, power = 2 , type = 'fitted')

lmfulltestlog <- lm(log(Units.Sold) ~ Average.Retail.Price + Sales.Rep + Endcap + demoCount + Sales.Rep * Endcap , data = trainingSet)
summary(lmfulltestlog)
BIC(lmfulltestlog)
BIC(lmfulltest)
BIC(lmfulltest1)
bc <- boxcox(lmfulltest)
with(bc, x[which.max(y)])


histogram(bellydf$Average.Retail.Price)
histogram(bellydf$Units.Sold)

lmfulltest1<- lm(Units.Sold^(2)~ Average.Retail.Price + Sales.Rep + Endcap + demoCount + Sales.Rep * Endcap, data = trainingSet)
summary(lmfulltest1)
residualPlots(lmfulltest1)
rest
