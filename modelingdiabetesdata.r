library(ggplot2)
library(e1071)
library("class")
library(randomForest)
library("neuralnet")
library("AUC")
library("caret")
library(MASS)
library(caTools)

##------IMPORT------
data <- read.csv(file="../PimaIndiansDiabetes.csv", header=TRUE, sep=",")

##------CORRECTING DATA TYPES------
#changes Outcome column to be logical True False values
data[,9]<- data[,9] > 0
str(data)
"Q1. 
Outfome is categorical -- you either have diabetes (1) or you dont (0) essentially like the Boolean True and False logical values rather than something countable. I believe the other attributes are all numerical. Pregnancies is a count of pregnancies. Glucose, BloodPressure, SkinThickness, Insulin, and BMI are all continuous numerical measurements with a specific unit of measurement attached (eg mm Hg). DiabetesPedigreeFunction is a calculated measure of genetic influence and therefore also a continouous numerical measurement. Age is also a discrete numerical count and therefore numeric."

##------CHECKING FOR MISSING VALUES------
head(data)
str(data)
#distribution between 0 pregnancies and 17. 17 is weird but possible, 0's are definitely possible, all values in between are possible as well.
sort(data$Pregnancies)
#no 0 values, reasonable distributio of values
sort(data$DiabetesPedigreeFunction)
#no 0 values, reasonable distributio of values
sort(data$Age)

#lots of 0 values--seems physiologically impossible so change them to NAs
sort(data$Glucose)
sort(data$BloodPressure)
sort(data$SkinThickness)
sort(data$Insulin)
sort(data$BMI)
#changes 0 values to NA
data[,2:7][data[,2:7]==0]<-NA

#new dataframe with only complete cases
d = data[complete.cases(data),]
"Q3. 
There's a lot of strange 0 values for the factors Glucose, BloodPressure, SkinThickness, Insulin, and BMI. I decided all these 0 values were errors and missing data due to the fact that the likelihood an individual would be alive while having any of these measurements at 0--0 plasma glucose, 0 mm Hg blood pressure, no tricep skin at all, zero insulin in blood, or a body mass index of 0--would be quite low if not outright impossible. I suppose the skin thickness measurement of 0 could be 0 if an individual perhaps had a loss of limbs, but I decided that it was far more likely it was just missing data. Furthermore, even if a person lacked tricep skin, their skin would be thick elsewhere and thus it would not be a fair conclusion that their skin thickness in general measured 0mm and would be a logical error regardless. After doing some research, it is possible but rare for someone to have zero measured insulin, but for 374/768 people in a random sample to all have 0 insulin would be near impossible and I deduce it is an error. For these columns, I replaced the 0 values w NA's.
The 0s in the Pregnancies column seemed reasonable and very likely to be real to me, and the other columns didn't have any null values or other strange values.

I decided to only use complete cases for the models (rows containing no NA values)."

##------GRAPHING initial look at data correlations------
for(f in head(colnames(data), -1))
{
    #cor() measures correlation between two variables. 
    print(cor(d$Outcome, d[f]))
    "Q2.    
    Pregnancies: 0.256566
    Glucose: 0.5157027
    BloodPressure: 0.1926733
    SkinThickness: 0.2559357
    Insulin: 0.3014292
    BMI: 0.2701184
    DiabetesPedigreeFunction: 0.2093295
    Age: 0.3508038

    Glucose has the highest correlation (aka level of linear dependance) with Outcome. Correlation varies between -1 to 1,high positive correlation means a value close to 1, high inverse correlation means a value close to -1, and weak correlation means a value close to 0."

    #ggplot(data)+geom_violin(mapping=aes_string(x="Outcome", y=f))+geom_jitter(mapping=(aes_string(x="Outcome",y=f)))
    ggplot(d)+geom_boxplot(mapping=aes_string(x="Outcome", y=f))
    'ggsave(paste("boxplot_",f,".png", sep=""), path="./Documents")'
}

##------SPLITTING DATA------
#10 fold cross validation
folds = createFolds(d$Outcome, k = 10, list = TRUE, returnTrain = FALSE)

##------BUILDING MODELS------
rf = matrix(0, ncol=10, nrow=8)
knn = matrix(0, ncol=10, nrow=8)
svm.linear = matrix(0, ncol=10, nrow=7)
svm.polynomial = matrix(0, ncol=10, nrow=7)
svm.radial = matrix(0, ncol=10, nrow=7)
scaledsvm.linear = matrix(0, ncol=10, nrow=7)
scaledsvm.polynomial = matrix(0, ncol=10, nrow=7)
scaledsvm.radial = matrix(0, ncol=10, nrow=7)
nn.zero = matrix(0, ncol=10, nrow=1)
nn.one = matrix(0, ncol=10, nrow=5)
nn.two = matrix(0, ncol=10, nrow=7)
nn.three = matrix(0, ncol=10, nrow=7)

i=1
for(f in folds)
{
    #split to train and test
    train = d[f,]
    test = d[-f,]

    #scaling data for neural net + scaled svm
    #scale train
    maxs = apply(train[,1:8], 2, max)
    mins = apply(train[,1:8], 2, min)
    scaled.train = as.data.frame(scale(train[,1:8],
                                  center=mins,
                                  scale = maxs-mins))
    Outcome = as.numeric(train$Outcome)- 1
    scaled.train = cbind(Outcome, scaled.train)
    #scaletest
    maxs = apply(test[,1:8], 2, max)
    mins = apply(test[,1:8], 2, min)
    scaled.test = as.data.frame(scale(test[,1:8],
                                  center=mins,
                                  scale = maxs-mins))
    Outcome = as.numeric(test$Outcome)- 1
    scaled.test = cbind(Outcome, scaled.test)


    #RANDOM FOREST
    #Random forests can't really be overfit by having too many trees so I decided to go with overkill
    # mtry val has to be within # of variables (we have 8) so I tried calculating AUC for mtry = 1 - 8
    for(mtry in 1:8)
    {
        model = randomForest(as.factor(Outcome) ~ ., data=train, importance=T, mtry=mtry, ntree=1000)
        prob = predict(model, newdata = test, type="prob")
        r = roc(prob[,2], as.factor(test$Outcome))
        #each row corresponds to the mtry val equal to its index (eg row 1 = mtry val of 1)
        rf[mtry,i] = auc(r)
    }

    #KNN
    for(k in 3:10)
    {
        #I decided to test different k values by choosing 8 k values around approximately the square root of the # of data points
        #square root is around 6, so I am doing 3-10
        model = knn(train[,-9], test[,-9], as.numeric(train$Outcome)-1, k=k, prob=TRUE)
        r = roc(model, as.factor(test$Outcome))
        knn[k-2,i] = auc(r)
    }

    #SVM
    for(c in  -3:3)
    {
        #unscaled
        #trying different cost values: 0.001, 0.01, 0.1, 1, 10, 100, 1000
        model = svm(as.factor(Outcome) ~ ., data = train, kernel = "linear", cost = 10^c, probability=T)
        p = predict(model, test, probability=T)
        r = roc(p, as.factor(test$Outcome))
        svm.linear[c+4,i] = auc(r)
        model = svm(as.factor(Outcome) ~ ., data = train, kernel = "radial", cost = 10^c, probability=T)
        p = predict(model, test, probability=T)
        r = roc(p, as.factor(test$Outcome))
        svm.radial[c+4,i] = auc(r)      
        model = svm(as.factor(Outcome) ~ ., data = train, kernel = "polynomial", cost = 10^c, probability=T)
        p = predict(model, test, probability=T)
        r = roc(p, as.factor(test$Outcome))
        svm.polynomial[c+4,i] = auc(r)

        #scaled
        #trying different cost values: 0.001, 0.01, 0.1, 1, 10, 100, 1000
        model = svm(as.factor(Outcome) ~ ., data = train, kernel = "linear", cost = 10^c, probability=T)
        p = predict(model, test, probability=T)
        r = roc(p, as.factor(test$Outcome))
        scaledsvm.linear[c+4,i] = auc(r)
        model = svm(as.factor(Outcome) ~ ., data = train, kernel = "radial", cost = 10^c, probability=T)
        p = predict(model, test, probability=T)
        r = roc(p, as.factor(test$Outcome))
        scaledsvm.radial[c+4,i] = auc(r)      
        model = svm(as.factor(Outcome) ~ ., data = train, kernel = "polynomial", cost = 10^c, probability=T)
        p = predict(model, test, probability=T)
        r = roc(p, as.factor(test$Outcome))
        scaledsvm.polynomial[c+4,i] = auc(r)
    }

    #NN
    #no hidden layers
    model = neuralnet(Outcome ~ ., data=scaled.train,hidden=c(0), rep=1, linear.output = F)
    results = compute(model, scaled.test) 
    r = roc(results$net.result, as.factor(test$Outcome))
    nn.zero[1,i] = auc(r)
    #one hidden layer -- test 5 diff hidden node counts
    for(h in 3:8)
    {
         model = neuralnet(Outcome ~ ., data=scaled.train,hidden=c(h*2), rep=1, linear.output = F)
        results = compute(model, scaled.test) 
        r = roc(results$net.result, as.factor(test$Outcome))
        nn.one[h-3,i] = auc(r)   
    }
    #two hidden layers -- test 7 diff node counts
    model = neuralnet(Outcome ~ ., data=scaled.train,hidden=c(3,3), rep=1, linear.output = F)
    results = compute(model, scaled.test) 
    r = roc(results$net.result, as.factor(test$Outcome))
    nn.two[1,i] = auc(r)
    model = neuralnet(Outcome ~ ., data=scaled.train,hidden=c(5,5), rep=1, linear.output = F)
    results = compute(model, scaled.test) 
    r = roc(results$net.result, as.factor(test$Outcome))
    nn.two[2,i] = auc(r)
    model = neuralnet(Outcome ~ ., data=scaled.train,hidden=c(3,5), rep=1, linear.output = F)
    results = compute(model, scaled.test) 
    r = roc(results$net.result, as.factor(test$Outcome))
    nn.two[3,i] = auc(r)
    model = neuralnet(Outcome ~ ., data=scaled.train,hidden=c(5,3), rep=1, linear.output = F)
    results = compute(model, scaled.test) 
    r = roc(results$net.result, as.factor(test$Outcome))
    nn.two[4,i] = auc(r)
    model = neuralnet(Outcome ~ ., data=scaled.train,hidden=c(5,2), rep=1, linear.output = F)
    results = compute(model, scaled.test) 
    r = roc(results$net.result, as.factor(test$Outcome))
    nn.two[5,i] = auc(r)
    model = neuralnet(Outcome ~ ., data=scaled.train,hidden=c(6,2), rep=1, linear.output = F)
    results = compute(model, scaled.test) 
    r = roc(results$net.result, as.factor(test$Outcome))
    nn.two[6,i] = auc(r)
    model = neuralnet(Outcome ~ ., data=scaled.train,hidden=c(6,3), rep=1, linear.output = F)
    results = compute(model, scaled.test) 
    r = roc(results$net.result, as.factor(test$Outcome))
    nn.two[7,i] = auc(r)
    #three hidden layers -- test 7 diff node counts
    model = neuralnet(Outcome ~ ., data=scaled.train,hidden=c(4,4,4), rep=1, linear.output = F)
    results = compute(model, scaled.test) 
    r = roc(results$net.result, as.factor(test$Outcome))
    nn.three[1,i] = auc(r)
    model = neuralnet(Outcome ~ ., data=scaled.train,hidden=c(5,5,5), rep=1, linear.output = F)
    results = compute(model, scaled.test) 
    r = roc(results$net.result, as.factor(test$Outcome))
    nn.three[2,i] = auc(r)
    model = neuralnet(Outcome ~ ., data=scaled.train,hidden=c(6,5,3), rep=1, linear.output = F)
    results = compute(model, scaled.test) 
    r = roc(results$net.result, as.factor(test$Outcome))
    nn.three[3,i] = auc(r)
    model = neuralnet(Outcome ~ ., data=scaled.train,hidden=c(6,5,5), rep=1, linear.output = F)
    results = compute(model, scaled.test) 
    r = roc(results$net.result, as.factor(test$Outcome))
    nn.three[4,i] = auc(r)
    model = neuralnet(Outcome ~ ., data=scaled.train,hidden=c(5,4,3), rep=1, linear.output = F)
    results = compute(model, scaled.test) 
    r = roc(results$net.result, as.factor(test$Outcome))
    nn.three[5,i] = auc(r)
    model = neuralnet(Outcome ~ ., data=scaled.train,hidden=c(6,3,5), rep=1, linear.output = F)
    results = compute(model, scaled.test) 
    r = roc(results$net.result, as.factor(test$Outcome))
    nn.three[6,i] = auc(r)
    model = neuralnet(Outcome ~ ., data=scaled.train,hidden=c(5,3,5), rep=1, linear.output = F)
    results = compute(model, scaled.test) 
    r = roc(results$net.result, as.factor(test$Outcome))
    nn.three[7,i] = auc(r)

    i=i+1
}
"Q4.
I decided to test all models and choose whichever has an AUC value closest to 1.0--meaning it has a relatively high true positive rate to a relatively low false positive rate. AUC is measures sensitivity vs specificity.
"

##------SELECTING BEST MODEL------
#calculates mean AUC values across folds for each model
allmeans = function (ms)
{
    #for each model in list
    for(m in seq_along(ms))
    {
        #get rowmeans
        ms[[m]] = rowMeans(ms[[m]])
    }
    #return means
    return(ms)
}

#list of all models
all = list(rf=rf, knn=knn, svm.linear=svm.linear, svm.polynomial=svm.polynomial, svm.radial=svm.radial, scaledsvm.linear=scaledsvm.linear, scaledsvm.polynomial=scaledsvm.polynomial, scaledsvm.radial=scaledsvm.radial, nn.zero=nn.zero, nn.one=nn.one, nn.two=nn.two, nn.three=nn.three)
#list of means for all models
means = allmeans(all)
#see which performed best--pick model with highest AUC value (closest to 1.0)
means

"Q3.
I believe Random forest with mtry=1 performed the best, as it had the highest AUC score of .7935.
"

##------RANDOM FOREST WITH MTRY = 1, NTREE = 1000------
#split to train and test with ratio of 30/70
split = sample.split(as.factor(d$Outcome), SplitRatio=0.7, group=NULL)
test = d[split]
train = d[!split]
#create model
model = randomForest(as.factor(Outcome) ~ ., data=train, importance=T, mtry=1, ntree=1000)
predictions = predict(model, newdata = test)

##------PLOTTING OUTCOMES vs PREDICTIONS------
ggplot(test, aes(y=Outcome==predictions, x=Outcome, shape = Outcome, color = Outcome==predictions)) + labs(title = "Random Forest Outcome vs Prediction Scatterplot", x = "Has Diabetes", y = "Correct Diabetes Prediction") + scale_color_discrete(name = "Correct Diabetes Prediction") + scale_shape_discrete(name = "Has Diabetes") + geom_jitter()
'ggsave(paste("plot_final_model.png", sep=""), path="./Documents")'

probs = predict(model, newdata = test, type="prob")
auc(roc(probs[,2], as.factor(test$Outcome)))
#final auc: 0.7889686
