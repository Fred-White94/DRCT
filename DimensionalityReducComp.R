#Get data following SDA dimensionality reduction,
#Here both the non reduced and reduced version of the data are brought in
#VF/PCA/ICA are performed on non reduced data and the results of this and the reduced data from SDA are compared using random forest


library(caret)
library(ica)

getDat <- function(i){
  trXFM <- read.csv(paste0("trX_preFT_Bin_19_04_30_SimBin_",i))
  teXFM <- read.csv(paste0("teX_preFT_Bin_19_04_30_SimBin_",i))
  trYFM <- read.csv(paste0("trY_preFT_Bin_19_04_30_SimBin_",i))
  teYFM <- read.csv(paste0("teY_preFT_Bin_19_04_30_SimBin_",i))
  
  
  HLdat <- read.csv(paste0("BIN_19_04_30_SimBin_",i,"_HiddenL_FOR_R"))
  SmallClass <- read.csv(paste0("teY_FT_Bin_19_04_30_SimBin_",i))
  intrain <- createDataPartition(SmallClass[,2], p = .75, list = F)
  
  set.seed(123)
  trXFT <- HLdat[intrain,]
  teXFT <- HLdat[-intrain,]
  trYFT <- SmallClass[intrain,]
  teYFT <- SmallClass[-intrain,]
  
  return(list(trXFM,teXFM,trYFM,teYFM,trXFT,teXFT,trYFT,teYFT))
  
}
#####
i <- NULL
GD <- NULL
for(i in 1:10){
  GD[[i]] <- getDat(i)
}



####### for disease data
#getDatd <- function(i){
#  
#  INPUTX <- read.csv(paste0("teX_FT_Bin_",i))
#  INPUTY <- read.csv(paste0("teY_FT_Bin_",i))
#  HLdat <- read.csv(paste0("BIN_",i,"_HiddenL_FOR_R"))
#  
#  intrain <- createDataPartition(INPUTY[,2], p = .75, list = F)
#  
#  
#  set.seed(78)
#  
#  trXFM <- INPUTX[intrain,]
#  teXFM <- INPUTX[-intrain,]
#  trYFM <- INPUTY[intrain,]
#  teYFM <- INPUTY[-intrain,]
#  
#  trXFT <- HLdat[intrain,]
#  teXFT <- HLdat[-intrain,]
#  trYFT <- INPUTY[intrain,]
#  teYFT <- INPUTY[-intrain,]
#  
#  return(list(trXFM,teXFM,trYFM,teYFM,trXFT,teXFT,trYFT,teYFT))
#  
#}
########

FSEtester1 <- function(train, test, trainclass, testclass, METHOD){
  
  
  trainclass <- as.factor(trainclass)
  testclass <- as.factor(testclass)
  
  if(as.character(METHOD[1]) == "VF"){
    
    train <- train
    test <- test
    
  }
  
  else if (as.character(METHOD[1]) == "SDA"){
    train <- train
    test <- test
    
  }
  
  else if (as.character(METHOD[1]) == "PCA"){
    PCD <- prcomp(train, scale. = T)
    test <- predict(PCD,test)[,1:35]
    train <- PCD$x[,1:35]
    
  }
  else if (as.character(METHOD[1]) == "ICA"){
    ICD <- icafast(train, nc = 35, center = T)
    test <- tcrossprod(as.matrix(test),ICD$W)
    train <- ICD$S
    colnames(train) <- paste("IC",c(1:35))
    colnames(test) <- paste("IC",c(1:35))
    
  }
  
  cat("Training Random Forest")
  
  set.seed(123)
  
  fitControl <- trainControl(method = "repeatedcv", number = 10, repeats = 10)
  rffit <- caret::train(x = as.matrix(train),y = trainclass, method = "rf", trControl = fitControl)
  
  CVACC <- rffit$results[which(rffit$results$Accuracy == max(rffit$results$Accuracy)),][1,]
  
  cat("Predicting Test Data")
  
  PREDICT <- predict(rffit, newdata = test)
  TESTACC <- length(which(PREDICT == testclass))/length(PREDICT)
  
  
  return(c(CVACC[,2],TESTACC))
}


x <- c("trXFM","teXFM","trYFM","teYFM")
y <- c("trXFT","teXFT","trYFT","teYFT")
m <- c("VF","PCA","ICA","SDA")
dict <- cbind(rbind(x,x,x,y),m)
colnames(dict) <- c("train","test","trainclass","testclass","method")
rownames(dict) <- NULL



#####

#i <- NULL
#DD <- NULL
#for(i in c("DiseaseDataAD.txt","DiseaseDataPD.txt","DiseaseDataSZ.txt")){
#  DD[[i]] <- getDatd(i)
#}



RESfinal <- NULL
i <- NULL
for(i in 1:length(DD)){
  
  trXFM <- GD[[i]][[1]]
  teXFM <- GD[[i]][[2]]
  trYFM <- GD[[i]][[3]]
  teYFM <- GD[[i]][[4]]
  trXFT <- GD[[i]][[5]]
  teXFT <- GD[[i]][[6]]
  trYFT <- GD[[i]][[7]]
  teYFT <- GD[[i]][[8]]
  
  trYFM[trYFM[,2] == 1,2] <- "Perturbed"
  trYFM[trYFM[,2] == 0,2] <- "Control"
  
  j <- NULL
  for(j in 1:dim(dict)[1]){
    
    train <- eval(as.symbol(dict[j,1]))[,-1]
    test <- eval(as.symbol(dict[j,2]))[,-1]
    trainclass <- eval(as.symbol(dict[j,3]))[,-1]
    testclass <- eval(as.symbol(dict[j,4]))[,-1]
    METHOD <- dict[j,5]
    
    results <- FSEtester1(train, test, trainclass, testclass, METHOD)
    
    RESfinal <- rbind(RES7,results)
    
    print(j)
    ########ADD list element to collect all data
    
  }
  
  print(i)
  
}
