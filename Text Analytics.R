#-------------Text Analytics-----------------
#Download dataset - https://www.kaggle.com/uciml/sms-spam-collection-dataset
#Code was written by me based on very good introduction at https://www.youtube.com/watch?v=4vuw0AsHeGw&t=64s
library(ggplot2)
library(e1071) #
library(caret) #mighty pakage, very udeful for data scientists.
library(randomForest)
library(quanteda)  #main pakage for text analytics
library(irlba) #doing singular value decomposition svds: (feature extraction)

#Load up the data 
spam.raw <- read.csv("spam.csv",stringsAsFactors=FALSE) #Before doing this make sure you are in the rigth directory. 
dim(spam.raw) #5572x5
View(spam.raw) #to see how file looks
spam.raw<-spam.raw[,1:2] #to remove extra  columns.  
dim(spam.raw)#5572x2
names(spam.raw)=c("Label","Text")

## Always check for missing values in any R project...
length(which(!complete.cases(spam.raw))) #0 dataframe has complete data in it.

#Convert "Label" column into factor
spam.raw$Label<-as.factor(spam.raw$Label)

##first step is to always explore the data
#distribution of class labels i.e in terms of %
prop.table(table(spam.raw$Label)) #would have worked without.
#86% 14% ##we have take care of imbalance as data looks biased towards ham.

##lets take a look at the length of texts and if gives any clues.
spam.raw$TextLength<-nchar(spam.raw$Text)
summary(spam.raw$TextLength)
#Max is 910
#Min is 2
#median=61, mean=80, clear diaparity as seen 1stQu.=36,3rdQu=121,max=910
#mean is 19 character higer than median

#The top distribution can also be analyzed by plots
ggplot(spam.raw,aes(x=TextLength,fill=Label))+  #fill=label color code the bars with label
  theme_bw()+
  geom_histogram(binwidth=5)+
  labs(y="Text Count",x="Length of Text",title="Distribution of Text
       Lengths with class label")
#fill works only with factor, that we converetd to factor

##since data is biased dividing it in test,cv and train has to be carefully done.
##use library caret to split data properly, 

##70%30% stratified split(to keep ham spam proportions same).Here we are not using test set
set.seed(32984)
indexes<-createDataPartition(spam.raw$Label,times=1,p=0.7,list=FALSE) #caret pakage
#spam.raw$Label make sure proportion is consatnt across all splits.
#times=1 only one split
#p=0.7 randon sample of 70%, 
#list=FALSE do not return other things than indexes
train<-spam.raw[indexes,]
test<-spam.raw[-indexes,]

#Verify proportions.
prop.table(table(train$Label))
prop.table(table(test$Label))

#How to represent text as a dataframe
#Words become columns!
#process is known as tokenization. It is a wide subject itself
#I will tokenization which is broadly used

##Document-Freuency Matrix (DFM)
#Each row represent document (here text message)
#Each column represents a distinct token
#Each cell is a count of the token for a document.
#Each it like a , duck
#4    2   4   1 1 4
#notice words are not repeated, so word ordering is lost.
#"bags-of-words" model - BOW" - model name suggest not in order

##do we want all token in our DFM ?
# Casing(e.g., If vs if) Answer: usually No.
# Punctuations(e.g ",?,!,etc)? No
# Numbers (e.g 0,56,109, etc.) No
# Every word (e.g- the,an,a,etc) No
# Symbols(e.g., <,@,#,etc.) 
#similar words (e.g.,ran,run,runs,runnung) ? can we coolapse ? (stemming)
##Pre-processing is a major part of text analytics##

train$Text[21] #"I'm back &amp; we're packing the car now.."
#amp; we get in place of &, how to handle that ? 
#this is an HTML-escaped '<' and '>' characters added, so if it is there how to remove it? amp; tells palce & on the screen
#here we will leave only amp and assume it is proxy for and. there are many ways to remove it.

train$Text[38] #$lt;#&gt; is for "<" and >" what to do ? one otion is to leave only lt and gt
# "#" many be left in these days (twitter), here we will remove
train$Text[357] #A URL. can be modified, here we will split it.

#Use quantdea package #Quantitaive analysis of text data#

#Tokenizing SMS text messages
train.tokens<-tokens(train$Text,what="word",remove_numbers=TRUE,
              remove_punct=TRUE,remove_symbols=TRUE,remove_hyphens=TRUE)
#what="word" means anything that is separated by space.
train.tokens[[357]]
train$Text[357] #real text

####this is called text pipeline###

#Lowe case the tokens
train.tokens<-tokens_tolower(train.tokens)
train.tokens[[357]]

#Use quanteadas build-in stopword list for english
#NOTE- alaways inspect once if needed to be removed. quanteada has a list of words stored 
#check ?stopwords  #see if you need them.

train.tokens<-tokens_select(train.tokens,stopwords(),selection="remove")
train.tokens[[357]]

#STEMMIN-Important, similar words collapse
train.tokens<-tokens_wordstem(train.tokens,language = "english")
train.tokens[[357]] # notice topped -top,credits-crdit etc...

###create first BOW model
train.tokens.dfm<-dfm(train.tokens,tolower=FALSE,remove=stopwords())
#transform to a matrix and inspect
train.tokens.matrix<-as.matrix(train.tokens.dfm)
View(train.tokens.matrix[1:20,1:100])
dim(train.tokens.matrix) #3901x5725 data size can get extremly large
#data can very very wide, ie TEXT ANALYTICS SUFFERS FROM DIMESIONALITY CURSE
#many cells are zero:ie problem of sparsing.

##IMPORTANT:investigate the effects of stemming
colnames(train.tokens.matrix)[1:50]

####NOW BUILD THE FIRST MODEL AFTER TEXT PROCESSING IS DONE

#setup feature dataframe with labels.
train.token.df<-cbind(Label=train$Label,as.data.frame(train.tokens.dfm))
dim(train.token.df) #3901x5726
#Often,toeknization requires some additional preprocessing
names(train.token.df[c(145,147,234,237)]) #try:wall is not valid name

#?make.names to cleanup column names
names(train.token.df)<-make.names(names(train.token.df)) #it will fix it
names(train.token.df[c(145,147,234,237)])
#Use caret to create stratified folds for 10-fold cross validation repeated
#3 times (create 30 random stratified samples)

set.seed(48743)
cv.folds<-createMultiFolds(train$Label,k=10,times=3) #cool function ?look
#cv.cntrl index tells that I have already calculated the folds use them. if not used it will run.
cv.cntrl<-trainControl(method="repeatedcv",number=10,repeats=3,index=cv.folds)

#WARNING - the following code is configured to run on a workstation or server-class machine (1.e, 12 logical cores).
library(doSNOW) #use to run training in caret in parallen.to speed up cv
#30 fold can be trained paralley using 30 different CPUs

#Time the code execution
start.time<-Sys.time()
#Create a cluster to work on 10 logical cores.Here we use 3
cl<-makeCluster(16,type="SOCK")   #with 10 cores it took 4 minutes #make cluster 3 inplace 0f 16 to run on laptop
registerDoSNOW(cl) #IMPORTANT-CARET will know it is registered that is it is available to caret
#type=SOCKET cluster :create multile instances of R studio and make those available to carat if it likes to
#user train function caret -IT IS AWESOME - 
rpart.cv.1<-train(Label~.,data=train.token.df,method="rpart",
                  trControl=cv.cntrl,tuneLength=7)
#method tells which algorthm we want to use, rpart,or rf, any model..
#using rpart because it is fast (it is single decision tree),rf is collection of decision trees.
#tuneLength=7 try 7 different configuration of modeland see which one is the best, 

#Processing is done, stop cluster
stopCluster(cl) #removes instances of r studio and free up resurces

#total time of execution on workstation was 4 minutes.
total.time<-Sys.time()-start.time
total.time  #On my laptop it took 12 minutes.

#Check out our results.
rpart.cv.1

#3901 rows #7725 features
#summary on sample sized: 3511,3510,3511...all are same
#accuracy 94.29% 
#shows Document-term frequency matrices work
#Bag of words model actually works...
########################################################################
########################################################################
########################################################################
#Longer documents will tend to have higher term counts.
#Terms that appear frequently across the corpus(different articles) aren't as important
#Longer words might be more frequent in longer document with similar word counts.
##We can improve if
#Normalize documents based on their length
#Penalize terms that occur frquently across the corpus.

# TERM FREQUENCY 
#freq(t,d) => count of instance of term t in document d
#TF(t,d) => be the proportion of the count of them t in document d.
#TF=freq/summation1:nt(freq)

#IDF
#N=count distince documents in the corpus
#count(t)=count of documents in the corpus in which term t is present.
#IDF(t)=log(N/count(t))=(0 if term is in every single document.)
#THE Mighty TF-IDF
#Combine TF and IDF to enhance Document-term frequency matrices
#TF-IDF(t,d)=TF(t,d)*IDF(t)
######################################################################
#Function to calcualte (TF)
term.frequency<-function(row){
  row/sum(row)
}
#function to calculate IDF
inverse.doc.freq<-function(col){
  corpus.size<-length(col)
  doc.count<-length(which(col>0))
  log10(corpus.size/doc.count) 
}   #QUANTIDA TFID function will genrate the same output
#Our function for calculating TF-IDF
tf.idf<-function(tf,idf){
  tf*idf
}
#################################################
#Standard pipeline in any text analytics
# Do lower case,remove symbols,remove numbers,highfens,STEM and then TF-IDF
#################################################
train.tokens.df<-apply(train.tokens.matrix,1,term.frequency)
dim(train.tokens.df) #note top function has transposed row and columnes. #5725x3901
View(train.tokens.df[1:20,1:100])

#2nd step,calculate the IDF vector that we will use-both for training data and for test data!
#Note- All new data is also transformned using same calcualted tf-idf.SAVE
train.tokens.idf<-apply(train.tokens.matrix,2,inverse.doc.freq)
str(train.tokens.idf)


#Lastly, calculate TF-IDF for our training corpus
train.tokens.tfidf<-apply(train.tokens.df,2,tf.idf,idf=train.tokens.idf) #applied on noramlize matrix which is actually transposed earlier.
dim(train.tokens.tfidf) #5725x3901

View(train.tokens.tfidf[1:25,1:25]) #there are two effects on each elements. High repetition words have lower values.

#Transpose the matrix
train.tokens.tfidf<-t(train.tokens.tfidf)
dim(train.tokens.tfidf) #3901x5725
#TFIDF is done-----------------------------
#But have to check for degenrative terms ? ie what is everything was removed earlier and we have some empty strings.

#Check incomplete cases
incomplete.cases<-which(!complete.cases(train.tokens.tfidf))
train$Text[incomplete.cases] #there are few of them..one of them has all stop words...

#Fix incomplete cases (otherwise will get error while training)
train.tokens.tfidf[incomplete.cases,]<-rep(0.0,ncol(train.tokens.tfidf))
dim(train.tokens.tfidf) #3901x5725
sum(which(!complete.cases(train.tokens.tfidf))) #0 incomplete cases

#Make a clean data frame using the same process as before
train.tokens.tfidf.df<-cbind(Label=train$Label,data.frame(train.tokens.tfidf))
names(train.tokens.tfidf.df) <- make.names(names(train.tokens.tfidf.df))
View(train.tokens.tfidf.df)

library(doSNOW)
start.time<-Sys.time()
cl<-makeCluster(16,type="SOCK")   #make 16 to 3 if want to run on laptop
registerDoSNOW(cl)
rpart.cv.2<-train(Label~.,data=train.tokens.tfidf.df,method="rpart",
                  trControl=cv.cntrl,tuneLength=7)
stopCluster(cl)
total.time<-Sys.time()-start.time
total.time 
rpart.cv.2  #Accracy is 94.53 Slight improvement.
########################################################################
########################################################################
########################################################################
#How to improve it further ? N-Grams
#single term is called 1-gram.
#N-gams allow us to extend the bag-of words modelling to include word ordering.
#bi-gram  #look_like like_duck duck_swim
                #1          3       1
#NOTE- WE HAVE MORE THAN DOUBLED THE TOTAL SIZE OF OUR MATRIX
#Adding N-grams dramatically increases model accuracy,but tehre is big increase in the matrix.
#Problem of sparcity, as most of them become is zero--------

#Add bigrams to our feature matrix
train.tokens<-tokens_ngrams(train.tokens,n=1:2) #from quenteda pakage
#note n=1:2 gives both bi gram and unigram.n=2 gives only bigrams
train.tokens[[357]]

#transform to dfm and then a matrix
train.tokens.dfm <- dfm(train.tokens,tolower=FALSE)
train.tokens.matrix <- as.matrix(train.tokens.dfm)
train.tokens.dfm  #3901x29154

#Normalizing all documents via TF
train.tokens.df <- apply(train.tokens.matrix,1,term.frequency)

#Calculate the IDF vector that will be used for training and test data
train.tokens.idf<-apply(train.tokens.matrix,2,inverse.doc.freq)
####need this idf values in future so store this soemhwere....

#Calculate TF-IDF for our training corpus
train.tokens.tfidf<-apply(train.tokens.df,2,tf.idf,idf=train.tokens.idf)

#Transpose the matrix
train.tokens.tfidf <- t(train.tokens.tfidf)

#Fix the incomplete cases
incomplete.cases <- which(!complete.cases(train.tokens.tfidf))
train.tokens.tfidf[incomplete.cases,] <- rep(0.0,ncol(train.tokens.tfidf))

#Make a clean data frame
train.tokens.tfidf.df <- cbind(Label=train$Label,data.frame(train.tokens.tfidf))
names(train.tokens.tfidf.df) <- make.names(names(train.tokens.tfidf.df))

#clean up used object in memory
gc() #Important -forces R to do garbage collection.Free up any memory if not needed.
#R usually cleans up periodically.
#-----------------------------------use multi core----------------------------------
#NOTE - The following code reqires the use of command-line R to execute
# due to the large number of features(i.e., columns) in the matrix.
# https://stackoverflow.com/questions/28728774/how-to-set-max-ppsize-in-r
# Also note that running the following code requires approx. 38gb of RAM 
# and more than 4.5 hours of execution on a 10-core workstation!

#Time the code execution
start.time <- Sys.time()
cl<-makeCluster(16,type="SOCK")  #this code will not run on laptop
#Leverage single decision trees to eveluate if adding bigrams imroves the effectiveness of the model.
rpart.cv.3 <- train(Label~.,data=train.tokens.tfidf.df,method="rpart",trControl=cv.cntrl,tuneLength=7)
total.time<-Sys.time() - start.time
stopCluster(cl)
total.time
#checkout results
rpart.cv.3     #94.3%
#-----------------------------------------------------------------------------------
#the result of above processing shows that a slight decline in rpart
#bigram appers to negatively impact the single decision tree,it helps with 
#the mighty random forest.
###############################################################################
###############################################################################
#Latenet Semantic Analysis(LSA) using singular value decomposition(SVD) factorization of a term-document matrix
#make it smaller and simutaneouly inorving column information.

#Progress till now
#>Representing unstructured text data in a format amenable to analytics and machine learning
#>Building a standard text analytics data pree=processing pipeline
#>improving the bag-of words(BOW) with the use of the mighty TF-IDF.
#>Extending BOW to incorporate word ordering via n-gram.
#Some prolems
#Document-term matrices explode to be very wide 
#Dont contain lot of single.(Sparse)
#running into scalability issues
#SVD of X =X=Usummation(Vtransopse)
#Where U eigenvectors of term correlation XXT
#v contains the eigenvectors of teh document correlations, XTX
#Sigma contains the singular values of the factorization.

#LSA often remediates the curse of dimensionality problem in text analytics
#the matrix factorization has the effect of columns,potentially enriching signal in data
#29000>300 column..dramatically reduces

#Important=combining three feature result in approximation i.e loss but the
#idea is combining in single column we gain lot more inform.
#SVD is effective and is a staple of text analytics pipelines

#For any new data or Test data we need to do=
#tokenize,unigram.bigram etc
#Normalize the document vector(i.e row) using term.frequency() function
#Complete the TF-IDF projection using tf.idf() function
#Apply the SVD projection on the document vector. dhat=summation-1(UTd)
#T=Transpose,dhat is projected document,d=tfidf projected document.
##########################################################################
##########################################################################
library(irlba) #it does truncated SVD (keeps only important SVDs)
 #SVD extrats higher level feature from combination of other variables.


#Time the code execution
start.time<-Sys.time()

#Perform SVD, Specifically, reduce dimensionality down to 300 columns for LSA
train.irlba <- irlba(t(train.tokens.tfidf),nv=300,maxit=600)
#nv can also be tunes if needed.
#matrix will do 600 iteration at max to get best 300 main values.

#total time of execution on workstation was
total.time<-Sys.time()-start.time
total.time

#Take a look at the new feature data up close
View(train.irlba$v)  #see what irba returns in ?irlba u-from perspective or columns,v-documents
#names columns as V1,V2,V3.....

#As with TF-IDF, we will need to project new data(eg, the test data) into-
#into SVD semantic space. The flowing code illustrates how to do this=
#using a row of the training data that has already been transformed by TF-IDF

sigma.inverse <- 1/train.irlba$d #computes inverse
u.transpose <- t(train.irlba$u)
document <- train.tokens.tfidf[1,]  #take first document
document.hat <- sigma.inverse*u.transpose%*%document #matrix multiplication

#Look at the first 10 components of projected document and the corresponding
#row in our document semantic space (i.e., the V matrix)
document.hat[1:10]
train.irlba$v[1,1:10] #3901x300

#How do we use it in practice?

#Create new feature data frame using our document semantic space of 300
train.svd<-data.frame(Label=train$Label,train.irlba$v)
#create cluster to work on 3 logical cores
cl<-makeCluster(16,type="SOCK")  #change 16 to 3 if working on laptop
registerDoSNOW(c1)
start.time <- Sys.time()
#This will be last run using single decision trees.next we will use Random Forest
rpart.cv.4 <- train(Label~.,data=train.svd,method="rpart",
                    trControl=cv.cntrl,tuneLength=7)
stopCluster(cl)
total.time <- Sys.time()-start.time
total.time
#Check out results
rpart.cv.4    ##94.43% 


#we have lost by adding bigram and then also lost by reducing the matrix
#We will gain if we use Randon Forest.
#
#The following code takes long time to run.10-f0ld CV repeated 3 times. i.e 30 models.
#Caret to use 7 different mtry parameters.by default random forest builds 500 decision trees.
#No. of trees we are building =(10*3*7*500)+500 = 105,500 trees
#extra 500 since in the end rf builds one with best parameter using all the training data
#On a workstation using 10 cores the following code took 28 minutes.
#---------------------------------------------------------------------------------------
#Create a cluster to work on 10 logical cores 8-27:22
cl<-makeCluster(16,type="SOCK")   #will not run on laptop
registerDoSNOW(cl)
start.time<-Sys.time()
rf.cv.1 <- train(Label~.,data=train.svd,method="rf",trControl=cv.cntrl,tuneLength=7)
#stopCluster(cl)
total.time <- Sys.time()-start.time
total.time
rf.cv.1   #96.75% 
#--------------------------------------------------------------------------------------
#confusionMatrix(train.svd$Label,rf.cv.1$finalModel$predicted) in the pakage carat
#there we use all the data so accuracy is slighly higher 96.82

########################################################
########################################################
#Sensitivity 0.9665 #need to go up.
#Specificity 0.9831

#Acurracy=(TP+TN)/(TP+FP+FN+TN)
#Sensitivity=TP/(TP+FN)
#Specificity=TN/(FP+TN)

#feature engineering is supper spper important
#you have to make lots of feature to find a good one
#Its normal your feature will fail many times.
#Feature might increase accuracy buy specifity goes down.
#We will prefer Sensitivty to up definetly not necessarily specifity

#Earlier we created a feature called Textlength
#and plotted in ggplot

#add that feature length
train.svd$TextLength <- train$TextLength
#----------------------------------------------------------------------------------
#Now run the code again....
cl<-makeCluster(16,type="SOCK")     #will not run on laptop
registerDoSNOW(cl)
start.time<-Sys.time()
rf.cv.2 <- train(Label~.,data=train.svd,method="rf",trControl=cv.cntrl,tuneLength=7)
#stopCluster(cl)
total.time <- Sys.time()-start.time
total.time
rf.cv.2   #96.75% 
#----------------------------------------------------------------------------------
#check the results
#rf.cv.2 #Accuracy 97.1%
#Sensitivity 0.9665
#Specificity 0.9928
#3375 spam correctly 3 spam incorrectly
#110 spam incorrectly 413 spam correctly

#text length may be right feature. thats the kind of features youu are looking as a data scientist

#How important was the new feature
#library(randomForest)
varImpPlot(rf.cv.1$finalModel) #plots important variables
varImpPlot(rf.cv.2$finalModel) #with added new feature
#Shows text-length was the most important parameter.

#usually in text analytics words importace are weak, made features are usually more powerfull
#########################################################
#########################################################
#Can add another feature based on Cosine similarity...

#costheta between two documents tell similarity
#cos will be between 0 and 1
#metric works well even in high dimensions
#cos is improvement over the dot product
#cos=1 exact similarity
#cos(ab)=A.B/|A||B|=(SUMMATION(AiBi))/(underoot(Ai^2)*....)

#cosine can be computed using many libraries
#here we use lsa library

train.similarities<-cosine(t(as.matrix(train.svd[,-c(1,ncol(train.svd))])))
#dim(train.svd)=3901x302  should be this
#remove first and last column.
#cosine only works on matrix and column vectors so flip it
#gives a lare matrix 
#dim(train.similarity) #3901 3901

#Now engineer a feature
#Spam should have higher cosine among them.

spam.indexes <- which(train$Label=="spam")

#can also do it without for loop 
train.svd$SpamSimilarity <- rep(0.0,nrow(train.svd))
#create a new column fill it with zeros
for(i in 1:nrow(train.svd)){
  train.svd$SpamSimilarity[i] <- mean(train.similarities[i,spam.indexes])
}
#now put values row by row by computing above.

#can visualize our result usig ggplot2
ggplot(train.svd,aes(x=SpamSimilarity,fill=Label))+
  theme_bw()+
  geom_histogram(binwidth=0.05)+
  labs(y="Message Count",
       x="Mean Spam Message Cosine Similarity",
       title="Distribution of Ham vs. Spam Using Cosine Similarity")

#Visualize your designed feature.

###again run the code
#----------------------------------------------------------------------
cl<-makeCluster(16,type="SOCK")   #will not run on laptop
registerDoSNOW(cl)
start.time<-Sys.time()
rf.cv.3 <- train(Label~.,data=train.svd,method="rf",trControl=cv.cntrl,tuneLength=7)
#stopCluster(cl)
total.time <- Sys.time()-start.time
total.time
rf.cv.3   
#----------------------------------------------------------------------
#can load from github: load("rf.cv.3.RData")

#Check the results
#rf.cv.3 #Accuracy=97.77

#Look at confusion matrix
confusionMatrix(train.svd$Label,rf.cv.3$finalModel$predicted)
#Accuracy 0.979,Sensitivity 0.979,Specificity 0.9722 this is great

#How important was the feature
library(randomForest)
carImpPlot(rf.cv.3$finalModel) 
#Spam similarity looks like very powerful feature
#may be indicative of overfitting
#since it reduced sensitivity and increased specivity combined with the 
#fact that it reduces the effct of other features.
#We will check this next
#########################################################
##########################################################
#Preprocessing has to be done on test data like training data.
# 1. Tokenization
# 2. Lower casing
# 3. Stopward removal
# 4. Stemming
# 5. Adding bigrams
# 6. Transform to dfms
# 7. Ensure test dfm has same features as train dfm.

#Tokenization
test.tokens <-tokens(test$Text,what="word",remove_numbers=TRUE,
                     remove_punct=TRUE,remove_symbols=TRUE,
                     remove_hyphens=TRUE)

#Lower case the tokens
test.tokens <- tokens_tolower(test.tokens)

#Stopword removal
test.tokens <- tokens_select(test.tokens,stopwords(),selection="remove")

#Stemming
test.tokens <- tokens_wordstem(test.tokens,language="english")

#Add bigrams
test.tokens <- tokens_ngrams(test.tokens,n=1:2)

#convert n-gram to quanteda document-term frequency matrix
test.tokens.dfm <- dfm(test.tokens,tolower=FALSE)

#Explore the train and test quanteda dfms objects
train.tokens.dfm #3901x29154
test.tokens.dfm  #1671x14747
#feature size for test decreases. but algorithm expects 
#same number of features as training set. If not tfidf and svd will not be valid

#in text data model can become stale if constant new kind of words 
#keep coming. needs constant update.

#NOTE - In production we should expect that new text messages will contain
#n-grmas that did not exist in the original training data. As, such,
#We need to strip those n-grams out.


test.tokens.dfm <- dfm_select(test.tokens.dfm,train.tokens.dfm)
test.tokens.matrix <- as.matrix(test.tokens.dfm)
test.tokens.dfm  #1671x29154

##next use same TF-IDF vecor space as in training.
#1.Normalize each document (i.e each row)
#2.Perform IDF multipication using training IDF values

#Normalize the document via TF
test.tokens.df <- apply(test.tokens.matrix,1,term.frequency)
str(test.tokens.df)

#Lastly calculate TF-IDF for our training corpus
test.tokens.tfidf <- apply(test.tokens.df,2,tf.idf,idf=train.tokens.idf)
dim(test.tokens.tfidf) #29154x1671
View(test.tokens.tfidf[1:25,1:25])
#make sure pakage allows to store tfidf of trainng data
#Transpose the matrix
test.tokens.tfidf <- t(test.tokens.tfidf)

#Fix the incomplete cases
summary(test.tokens.tfidf[1,])
test.tokens.tfidf[is.na(test.tokens.tfidf)] <- 0.0
summary(test.tokens.tfidf[1,])

#With test data projected into the TF-IDF vector space of the training
#data we can now do the final projection into the training LSA semantic
#space (i.e the SVD matrix factorization)
test.svd.raw <- t(sigma.inverse*u.transpose%*%t(test.tokens.tfidf))
dim(test.svd.raw) #1671x300

#Lastly,we can now build the test data frame to feed into our trained 
#machne learning model for predictions, First up, add Label and TextLength
test.svd <- data.frame(Label=test$Label,test.svd.raw,
                      TextLength=test$TextLength)
#Next step, calculate SpamSimilarity for all the test documents. First up.
#create a spam similarity matrix.
#we added rows of training spams
test.similarities <- rbind(test.svd.raw,train.irlba$v[spam.indexes,])
test.similarities <- cosine(t(test.similarities))
# cosine calculates between column,so transpose it.

test.svd$SpamSimilarity <- rep(0.0,nrow(test.svd))
spam.cols <- (nrow(test.svd)+1):ncol(test.similarities)
for(i in 1:nrow(test.svd)){
  test.svd$SpamSimilarity[i] <- mean(test.similarities[i,spam.cols])
}
#Now we can make predictions
sum(is.na(test.svd$SpamSimilarity)) # check for NAs and also if legths of pred
#preds and test.svd$Label are equal.
preds <- predict(rf.cv.3,test.svd)
confusionMatrix(preds,test.svd$Label)
#Accuracy 86.59 much worse, ie is overfitting.
#Sensitivity is 100%, Specivicity is 0.
#May be similarity function is not good on test set.
##############################################################
##############################################################
#overfitting is doing far better on the trainning data as
#evidenced by CV than doing on a hold-out dataset(i.e our test data set)
#One potential explanation of this overfittign is the use of spam simil-
#arity feature. The hypothesis here is that spam features (i.e., text 
#content) varies highly,especially voer time. lets rebuild rf model
#without the spam similarity feature.
train.svd$SpamSimilarity <- NULL
test.svd$SpamSimilarity <- NULL
#------------------------------------------------------------
cl<-makeCluster(16,type="SOCK")  #use 3 if running on laptop
registerDoSNOW(cl)  #took 2.5 hrs with 3 cores
#time the code execution
start.tim <- Sys.time()
#Re-run the training process with the additional feature,
set.seed(3892473)
rf.cv.4 <- train(Label~.,data=train.svd,method="rf",
                 trControl=cv.cntrl,tuneLength=7,
                 importance=TRUE)
stopCluster(cl)
total.time <- Sys.time(-start.time)
total.time   
#----------------------------------------------------------
preds <- predcit(rf.cv.4,test.svd)
confusionMatrix(preds,test.svd$Label)
#Accuracy 96.47, 59 spam it predicted ham,and 1447 ham correctly,and 165 spam it predicetd as spam and 0 ham it predicetd as spam.
#Sensitivity:1
#Specificity:0.7366
###################################################################
#CV is an estimate of generalization.
#NEXT
#adding 3-grams,4-grams etc
#or try only bi-grams not bot one and two
#or may be combination of 2-gram and 4-gram
#Text length worked
#Cosine didnot work
#Random easy to tune.
#Can try boosting tree
#DEfactor standart in Text Analytics is SVM
#SVM tend to work better on low signal (many were sparse) and large-
#feature spaces....

#Book- Text Analysis with R for Students of Literature Matthew L.Jockers
#contains topic modeling visualize it as word cloud.
#Book-Taming Text (has lots of theory) -(has Java code)
#OpenNLP Pakage in JAVA, R has wrapper.
#CRAN NLP that work with text analytics
#search in google,yah......"Introduction to information Retrieval"-free online version
#Python has Natural Language toolkit(NLTK).
#Natural Language Processing with Python -has good theory.
#---------------------------------------------------------------------------














































































