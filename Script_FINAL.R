library(randomForest)
library(caret)
library(readxl)
library(bayesreg)
library(bartMachine)
library(brnn)
library(devtools)
library(keras)
library(tensorflow)
library(rBayesianOptimization)
library(psoptim)
library(knitr)
library(e1071)
library(randomForestExplainer)
library(ROCR)
library(jtools)
library(broom.mixed)
library(varrank)
library(FactoMineR)
library(factoextra)
library(coda)
library(ggplot2)
library(forecast)
library(data.table)
library(stableGR)
library(mgm)
library(hydroPSO)
library(rstanarm)
library(ggmcmc)
library(xlsx)

memory.limit(size=400000)


###### Dataset scaling #######
#The dataset is taken from the xlsx file retrived from the https://exoplanetarchive.ipac.caltech.edu/ website (TESS project candidates).
#In order to reduce the complexity of the classification the TFOPWG Disposition was set as follow: 1 for KP PC CP 0 else. If the user 
#would reproduce the .rda file he/she should consider to uncomment the first lines


#dataset_raw <- read_excel("Dataset_FINAL.xlsx")
#dataset_raw <- na.omit(dataset_raw)
#dataset_fin <- dataset_raw[,-1]
#dataset_fin[,2:17]<-scale(dataset_fin[,2:17])
#save(dataset_fin,file="Dataset_final.rda")
load(file="Dataset_final.rda")
train_set <- sample(4271,2847)
#save(train_set,file="train_set_DEF.rda") #Use these two lines for the ceteris paribus condition
#load("train_set_DEF.rda")
test_set<-dataset_fin[-train_set,]
dataset_fin_u<-dataset_fin
dataset_fin$tfopwg_disp<-as.factor(dataset_fin$tfopwg_disp)


##### Preliminary Analysis #####

res.famd <- FAMD(dataset_fin)
fviz_famd_var(res.famd, "quanti.var", col.var = "contrib", 
              gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
              repel = TRUE)
fviz_famd_var(res.famd, "quali.var", col.var = "contrib", 
              gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07")
)
fviz_famd_ind(res.famd, col.ind = "cos2", 
              gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
              repel = TRUE)


MI_var<- varrank(data= dataset_fin, variable.important = "tfopwg_disp", discretization.method = "sturges")
plot(MI_var,margins = c(8, 8, 4, 0.1),notecex=0.7,digitcell=2,labelscex=0.6,textlines = 0.,maincex =0.5)


k_model<-mgm(dataset_fin_u,type= c(rep("c",1),rep("g",16)),levels=c(rep(2,1),rep(1,16)),k=2,ruleReg = "AND",overparameterize = T,lambdaSel = "CV",lambdaFolds= 10)

qgraph::qgraph(k_model$pairwise$wadj,
               layout = "spring", repulsion = 1.3,
               edge.color = k_model$pairwise$edgecolor,
               nodeNames = colnames(dataset_fin_u),
               color = c(rep("red",1),rep("lightblue",16)),
               legend=TRUE,
               vsize = 5, esize = 15,legend.cex=.5,legend.mode="names")


###### NON bayesian algorithms ####

##### Random forest #####

rfor.planet <-randomForest(tfopwg_disp~ . ,data=dataset_fin,subset=train_set,localImp = TRUE,importance=TRUE,proximity=TRUE,ntree=1000)
rfor.predict<-data.frame(predict(rfor.planet, test_set, type = "class"))

var_imp_rforest<-data.frame(varImp(rfor.planet))
colnames(var_imp_rforest)<-c("Variable","Overall")
var_imp_rforest[,1]<-rownames(var_imp_rforest)


ggplot(var_imp_rforest, aes(y=reorder(Variable,Overall),x=Overall)) + 
  geom_point(color="blue") +
  geom_segment(aes(x=0,xend=Overall,yend=Variable),color="blue") +
  scale_color_discrete(name="Variable Group") +
  xlab("Overall importance") +
  ylab("Variable Name") + guides(color = FALSE, size = FALSE) + theme_bw()+theme(text = element_text(size=20))


rfor.predict["Test"]<-test_set$tfopwg_disp
colnames(rfor.predict)<-c("P","T")

caret::confusionMatrix(table(rfor.predict),mode = "everything")

fourfoldplot(table(rfor.predict), color = c("red","darkgreen"),conf.level = 0, margin = 1, main = "Random Forest")

pred_for<-prediction(as.numeric(rfor.predict$P),as.numeric(rfor.predict$T))

roc_for.perf <- performance(pred_for, measure = "tpr", x.measure = "fpr")

perf<-data.frame(roc_for.perf@y.values)
perf["X"]<-data.frame(roc_for.perf@x.values)
colnames(perf)<-c("TP","FP")

roc_for.phi <- performance(pred_for, measure ="phi")
roc_for.mi <- performance(pred_for, measure ="mi")

ggplot(perf, aes(y=TP,x=FP,color="red")) + geom_point() +geom_line() + theme_bw()+ theme(legend.position="none")



##### Logistic model #####


model <- glm(tfopwg_disp~ .,family=binomial(link='logit'),data=dataset_fin)

logistic.prob<-data.frame(predict(model ,dataset_fin[-train_set,],type = "response"))
colnames(logistic.prob)<-c("P")
logistic.prob<- data.frame(ifelse(logistic.prob > 0.5, "1", "0"))
logistic.prob["T"]<-dataset_fin[-train_set,"tfopwg_disp"]

colnames(logistic.prob)<-c("P","T")


plot_summs(model,scale = TRUE, plot.distributions = TRUE)

var_imp_model_logistic <-data.frame(varImp(model))
var_imp_model_logistic [,2]<-rownames(var_imp_model_logistic )
colnames(var_imp_model_logistic )<-c("Overall","Variable")


ggplot(var_imp_model_logistic , aes(y=reorder(Variable,Overall),x=Overall)) + 
  geom_point(color="blue") +
  geom_segment(aes(x=0,xend=Overall,yend=Variable),color="blue") +
  scale_color_discrete(name="Variable Group") +
  xlab("Overall importance") +
  ylab("Variable Name") + guides(color = FALSE, size = FALSE) + geom_vline(xintercept=1.96,linetype='dotted',col = 'black') +theme_bw()+theme(text = element_text(size=20))

fourfoldplot(table(logistic.prob), color = c("red","darkgreen"),conf.level = 0,margin = 1)

caret::confusionMatrix(table(logistic.prob),mode = "everything")


pred_log<-prediction(as.numeric(logistic.prob$P),as.numeric(logistic.prob$T))

roc_log.perf <- performance(pred_log, measure = "tpr", x.measure = "fpr")

perf_log<-data.frame(roc_log.perf@y.values)
perf_log["X"]<-data.frame(roc_log.perf@x.values)
colnames(perf_log)<-c("TP","FP")

roc_log.phi <- performance(pred_log, measure ="phi")
roc_log.mi <- performance(pred_log, measure ="mi")

ggplot(perf_log, aes(y=TP,x=FP,color="red")) + geom_point() +geom_line() + theme_bw()+ theme(legend.position="none")

###### Bayesian algorithms ####

##### Logistic Bayesian #####

logistic_bayes_output<-bayesreg(tfopwg_disp ~ . ,dataset_fin[train_set,],model = "logistic",prior = "lasso",n.samples = 10000,thin=5,burnin=1500)
logistic_bayes_output_summary <- summary(logistic_bayes_output,sort.rank=TRUE)
t_stat_summary<-data.frame(logistic_bayes_output_summary$t.stat[,0])
t_stat_summary["Name"]<-rownames(t_stat_summary)
t_stat_summary["Value"]<-as.data.frame(logistic_bayes_output_summary[["t.stat"]])
t_stat_summary["Importance"]<-abs(as.data.frame(logistic_bayes_output_summary[["t.stat"]]))
t_stat_summary <- na.omit(t_stat_summary)

plot_coef_logistic_bayes_output_summary<- as.data.frame(logistic_bayes_output_summary$mu.coef)
plot_coef_logistic_bayes_output_summary[,"CILOW"]<-logistic_bayes_output_summary$CI.coef[,1]
plot_coef_logistic_bayes_output_summary[,"CIUP"]<-logistic_bayes_output_summary$CI.coef[,2]
plot_coef_logistic_bayes_output_summary[,"Names"]<-rownames(logistic_bayes_output_summary$mu.coef)
colnames(plot_coef_logistic_bayes_output_summary)<-c("Value","CILOW","CIUP","Names")
plot_coef_logistic_bayes_output_summary<-plot_coef_logistic_bayes_output_summary[-17,]


ggplot(plot_coef_logistic_bayes_output_summary, aes(y=Names,x=Value))+ geom_point(color="red") + geom_segment(aes(x=CILOW ,xend=CIUP ,yend=Names),color="red")+theme_bw()+geom_vline(xintercept=0,linetype='dotted',col = 'black')+theme(text = element_text(size=20))


ggplot(t_stat_summary, aes(y=reorder(Name,abs(Value)),x=Value,color="red")) + 
  geom_point() +
  geom_segment(aes(x=0,xend=Value,yend=Name)) +
  scale_color_discrete(name="Variable Group") +
  xlab("t-statistic") +
  ylab("Variable Name") + guides(color = FALSE, size = FALSE) + geom_vline(xintercept=1.96,linetype='dotted',col = 'blue')+geom_vline(xintercept=-1.96,linetype='dotted',col = 'blue')+theme_bw()


ggplot(t_stat_summary, aes(y=reorder(Name,Importance),x=Importance,color="red")) + 
  geom_point() +
  geom_segment(aes(x=0,xend=Importance,yend=Name)) +
  scale_color_discrete(name="Variable Group") +
  xlab("Overall Importance") +
  ylab("Variable Name") + guides(color = FALSE, size = FALSE) + geom_vline(xintercept=1.96,linetype='dotted',col = 'black')+theme_bw()+theme(text = element_text(size=20))


plot_summs(logistic_bayes_output,scale = TRUE, plot.distributions = TRUE)

y_pred <- as.data.frame(predict(logistic_bayes_output, dataset_fin[-train_set,] , type='class'))
y_pred["T"]<-dataset_fin[-train_set,"tfopwg_disp"]
colnames(y_pred)<-c("P","T")
table(y_pred)
caret::confusionMatrix(table(y_pred),mode = "everything")

fourfoldplot(table(y_pred), color = c("red","darkgreen"),conf.level = 0,margin = 1)

pred_log<-prediction(as.numeric(y_pred$P),as.numeric(y_pred$T))

roc_log_bayesian.perf <- performance(pred_log, measure = "tpr", x.measure = "fpr")


perf_log_b<-data.frame(roc_log_bayesian.perf@y.values)
perf_log_b["X"]<-data.frame(roc_log_bayesian.perf@x.values)
colnames(perf_log_b)<-c("TP","FP")

ggplot(perf_log_b , aes(y=TP,x=FP,color="red")) + geom_point() +geom_line() + theme_bw()+ theme(legend.position="none")

roc_log_bayesian.phi<- performance(pred_log, measure = "phi")
roc_log_bayesian.mi<- performance(pred_log, measure = "mi")


perf_log<-data.frame(roc_log.perf@y.values)
perf_log["X"]<-data.frame(roc_log.perf@x.values)
colnames(perf_log)<-c("TP","FP")

roc_log.phi <- performance(pred_log, measure ="phi")
roc_log.mi <- performance(pred_log, measure ="mi")


acf_output<-logistic_bayes_output[["beta"]]

lines(acf_output[1,1:10000])

acf_output_f<-transpose(as.data.frame(acf_output))

colnames(acf_output_f)<-colnames(dataset_fin[,-1])

qplot(1:10000,acf_output_f[,"st_rad"])+geom_line(color="blue")+ylab("pl_rad")+xlab("iteration")+theme_bw()+theme(axis.text=element_text(size=16),axis.title=element_text(size=14,face="bold"))

ggAcf(acf_output_f[,"pl_tranmid"])+theme_bw()+theme(axis.text=element_text(size=16),axis.title=element_text(size=14,face="bold"))+ggtitle("pl_tranmid ACF") 
ggAcf(acf_output_f[,"st_tmag"])+theme_bw()+theme(axis.text=element_text(size=16),axis.title=element_text(size=14,face="bold"))+ggtitle("st_tmag ACF") 
ggAcf(acf_output_f[,"pl_eqt"])+theme_bw()+theme(axis.text=element_text(size=16),axis.title=element_text(size=14,face="bold"))+ggtitle("pl_eqt ACF") 
ggAcf(acf_output_f[,"pl_rade"])+theme_bw()+theme(axis.text=element_text(size=16),axis.title=element_text(size=14,face="bold"))+ggtitle("pl_rade ACF") 
ggAcf(acf_output_f[,"pl_trandep"])+theme_bw()+theme(axis.text=element_text(size=16),axis.title=element_text(size=14,face="bold"))+ggtitle("pl_trandep ACF") 
ggAcf(acf_output_f[,"pl_trandurh"])+theme_bw()+theme(axis.text=element_text(size=16),axis.title=element_text(size=14,face="bold"))+ggtitle("pl_trandurh ACF") 


MC<-mcmc(t(logistic_bayes_output[["beta"]]),thin=1)
gd<-geweke.diag(MC)
gd_frame<-data.frame(gd[["z"]])
gd_frame["Var_name"]<-colnames(dataset_fin[,-1])
colnames(gd_frame)<-c("Value","Var_name")

qplot(gd_frame$Value,gd_frame$Var_name)+geom_point(color="red")+geom_segment(aes(x=0,xend=gd_frame$Value,yend=gd_frame$Var_name),color="red")+scale_x_continuous(limits = c(-3, 3))+theme_bw()+geom_vline(xintercept=1.96,linetype='dotted',col = 'blue')+geom_vline(xintercept=-1.96,linetype='dotted',col = 'blue')+theme(axis.text=element_text(size=20),axis.title=element_text(size=14,face="bold"))+ylab("Variable name")+xlab("Z score")

output_stable_GR<-stable.GR(t(logistic_bayes_output[["beta"]]))

raftery_output<-raftery.diag(MC)
resmatrix<-raftery_output[["resmatrix"]]
resmatrix_final<-as.data.frame(resmatrix[,4])
resmatrix_final[,2]<-colnames(dataset_fin[,-1])
colnames(resmatrix_final)<-c("I","Name")


ggplot(resmatrix_final, aes(y=reorder(Name,abs(I)),x=I,color="red")) + 
  geom_point() +
  geom_segment(aes(x=0,xend=I,yend=Name)) +
  scale_color_discrete(name="Variable Group") +
  xlab("Dependence factor (I)") +
  ylab("Variable Name") + guides(color = FALSE, size = FALSE) + geom_vline(xintercept=5,linetype='dotted',col = 'blue')+theme_bw()+theme(text = element_text(size=20))






################Hamiltonian montecarlo for logistic regression########

t_prior <- student_t(df = 5, location = 0, scale = 0.5)
fit1 <- stan_glm(
  tfopwg_disp ~ .,
  data = dataset_fin[train_set,],
  family = binomial(link = "logit"),
  QR = TRUE,
  chains=4,
  prior = t_prior, 
  iter=20000,
  warmup = 10000,
  prior_intercept = t_prior,
  core=16
)
describe_posterior(fit1)
#save(fit1,file="output_fit_bayesian_log_df3.rda")
summary_bayesian_log0<-as.data.frame(summary(fit1))
plot(fit1, "areas", prob = 0.95, prob_outer = 1)
output_fit_bayesian_log0<-ggs(fit1)
ggs_geweke(output_fit_bayesian_log0)
ggs_running(output_fit_bayesian_log0)
ggs_autocorrelation(output_fit_bayesian_log0)
ggs_Rhat(output_fit_bayesian_log0) + xlab("R_hat")
ggs_compare_partial(output_fit_bayesian_log0)
ggs_grb(output_fit_bayesian_log0)




p0 <- 5 # prior guess for the number of relevant variables
tau0 <- p0/(16-p0) * 1/sqrt(2847)
hs_prior <- hs(df=1, global_df=1, global_scale=tau0)
t_prior <- student_t(df = 5, location = 0, scale = 1)
fit4 <- stan_glm(
  tfopwg_disp ~ .,
  data = dataset_fin[train_set,],
  family = binomial(link = "logit"),
  QR = TRUE,
  chains=4,
  iter=20000,
  warmup = 10000,
  prior = hs_prior, 
  prior_intercept = t_prior,
  core=16
)
#save(fit3,file="output_fit_bayesian_log_df10.rda")
describe_posterior(fit4)
summary_bayesian_log<-as.data.frame(summary(fit4))
plot(fit4, "areas", prob = 0.95, prob_outer = 1)
pplot2<-plot(fit4, "areas", prob = 0.95, prob_outer = 1)
output_fit_bayesian_log4<-ggs(fit4)
ggs_geweke(output_fit_bayesian_log4)
ggs_running(output_fit_bayesian_log4)
ggs_autocorrelation(output_fit_bayesian_log4)
ggs_Rhat(output_fit_bayesian_log4) + xlab("R_hat")
ggs_compare_partial(output_fit_bayesian_log4)
ggs_grb(output_fit_bayesian_log4)

#write coeff in excel format

#write.xlsx(summary_bayesian_log, file = "summary_bayesian_log0.xlsx",sheetName = "hs", append = FALSE)




y_pred<-predict(fit1,newdata=dataset_fin[-train_set,],type="response")
y_predX<-data.frame(ifelse(y_pred > 0.5, "1", "0"))
y_pred<-y_predX






y_pred["T"]<-dataset_fin[-train_set,"tfopwg_disp"]
colnames(y_pred)<-c("P","T")
table(y_pred)
caret::confusionMatrix(table(y_pred),mode = "everything")

fourfoldplot(table(y_pred), color = c("red","darkgreen"),conf.level = 0,margin = 1)

pred_log<-prediction(as.numeric(y_pred$P),as.numeric(y_pred$T))

roc_log_bayesian.perf <- performance(pred_log, measure = "tpr", x.measure = "fpr")


perf_log_b<-data.frame(roc_log_bayesian.perf@y.values)
perf_log_b["X"]<-data.frame(roc_log_bayesian.perf@x.values)
colnames(perf_log_b)<-c("TP","FP")

ggplot(perf_log_b , aes(y=TP,x=FP,color="red")) + geom_point() +geom_line() + theme_bw()+ theme(legend.position="none")

roc_log_bayesian.phi<- performance(pred_log, measure = "phi")
roc_log_bayesian.mi<- performance(pred_log, measure = "mi")


perf_log<-data.frame(roc_log.perf@y.values)
perf_log["X"]<-data.frame(roc_log.perf@x.values)
colnames(perf_log)<-c("TP","FP")

roc_log.phi <- performance(pred_log, measure ="phi")
roc_log.mi <- performance(pred_log, measure ="mi")


#Coeff plot####
load("Coeff_log.rda")



ggplot(Coeff, aes(y=Name, x=LASSO,group="LASSO",col="LASSO")) + geom_pointrange(aes(xmin=LASSO-LASSO_sd, xmax=LASSO+LASSO_sd))+ geom_point(aes(y=Name, x=HMC_S5,group="HMC_S5",col="HMC_S5"))+ geom_pointrange(aes(xmin=HMC_S5-HMC_S5_SD, xmax=HMC_S5+HMC_S5_SD))+geom_point(aes(y=Name, x=HMC_S10,group="HMC_S10",col="HMC_S10"))+ geom_pointrange(aes(xmin=HMC_S10-HMC_S10_SD, xmax=HMC_S10+HMC_S10_SD))+geom_point(aes(y=Name, x=HMC_HS,group="HMC_HS",col="HMC_HS"))+ geom_pointrange(aes(xmin=HMC_HS-HMC_HS_SD, xmax=HMC_S10+HMC_HS_SD))+geom_point(aes(y=Name, x=FREQ,group="FREQ",col="FREQ"))+ geom_pointrange(aes(xmin=FREQ-FREQ_SD, xmax=FREQ+FREQ_SD))+theme_bw()+xlab("Coeff. value") + theme(text = element_text(size=20))





###### Random forest bayesian ####

#I suggest to empty the unused memory before this passage

dataset_fin_train<-dataset_fin[train_set,]

x<-as.data.frame(dataset_fin_train[,2:17])
y<-factor(dataset_fin_train$tfopwg_disp)
bart_machine<-build_bart_machine(x,y,mem_cache_for_speed=FALSE,num_burn_in = 1000,num_iterations_after_burn_in = 5000)

plot_convergence_diagnostics(bart_machine)



cf_bart<-bart_machine$confusion_matrix
BART_VAR_IMP<-investigate_var_importance(bart_machine)
BART_VAR_IMP_FRAME <- as.data.frame(BART_VAR_IMP[["avg_var_props"]])
BART_VAR_IMP_FRAME["Names"]<- rownames(BART_VAR_IMP_FRAME)
colnames(BART_VAR_IMP_FRAME)<-c("Importance","Name")



ggplot(BART_VAR_IMP_FRAME, aes(y=reorder(Name,Importance),x=Importance,color="red")) + 
  geom_point() +
  geom_segment(aes(x=0,xend=Importance,yend=Name)) +
  scale_color_discrete(name="Variable Group") +
  xlab("Overall Importance") +
  ylab("Variable Name") + guides(color = FALSE, size = FALSE)+theme_bw()+theme(text = element_text(size=20))






dataset_fin_test<-dataset_fin[-train_set,]
x_test<-as.data.frame(dataset_fin_test[,2:17])
y_test<-factor(dataset_fin_test$tfopwg_disp)

test_bart<-bart_predict_for_test_data(bart_machine, x_test, y_test)
cf_bart<-test_bart$confusion_matrix
cf_x<-data.frame(matrix(ncol = 2, nrow = 2))

colnames(cf_x)<-c("P","T")
row.names(cf_x)<-c("P","T")

#Since the prediction algorithm directly produces the confusion matrix (in a format that is not compatible with caret)
#and not the dataframe as for the other algorithms, the author considered this way in order to produce the plot and the 
#data about the confusion matrix: a random confusion matrix is generated, and then its values are replaced with the one of BART 

y_pred<-as.data.frame(sample(0:1, 10,replace = T))
y_pred[,"P"]<-as.data.frame(sample(0:1, 10,replace = T))
colnames(y_pred)<-c("T","P")

table_bart<-table(y_pred)
table_bart[1,1]<- cf_bart[1,1]
table_bart[1,2]<- cf_bart[2,1]
table_bart[2,1]<- cf_bart[1,2]
table_bart[2,2]<- cf_bart[2,2]

caret::confusionMatrix(table_bart[1:2,1:2],mode = "everything")





#Load of the dataframe that produce the confusion matrix previously obtained. 
#The data in the xlsx file comes from the confusion matrix obtained by the author.
#In principle this part should be automatized, however considering that it is only 
#for two plots, and for the extraction of two performances values, the author preferred 
#to use this less time consuming way
#cf_bart <- as.data.frame(read_excel("Results/HI/RandomForest_bayesian/cf.xlsx"))


#pred_bart<-prediction(as.numeric(cf_bart$P),as.numeric(cf_bart$T))

#roc_log_bayesian.perf <- performance(pred_bart, measure = "tpr", x.measure = "fpr")


#perf_bart<-data.frame(roc_log_bayesian.perf@y.values)
#perf_bart["X"]<-data.frame(roc_log_bayesian.perf@x.values)
#colnames(perf_bart)<-c("TP","FP")

#ggplot(perf_bart , aes(y=TP,x=FP,color="red")) + geom_point() +geom_line() + theme_bw()+ theme(legend.position="none")

#perf_bart.phi<- performance(pred_bart, measure = "phi")
#perf_bart.mi<- performance(pred_bart, measure = "mi")


#fourfoldplot(table_bart, color = c("red","darkgreen"),conf.level = 0,margin = 1)







###### Bayesian optimization ####


dataset_fin_train<-dataset_fin[train_set,]
x<-as.data.frame(dataset_fin_train[,2:17])
y<-factor(dataset_fin_train$tfopwg_disp)

x<- na.omit(x)
y<- na.omit(y)
y<-as.integer(as.vector(y))


dataset_fin_test<-dataset_fin[-train_set,]
x_test<-as.data.frame(dataset_fin_test[,2:17])
y_test<-factor(dataset_fin_test$tfopwg_disp)


y_test<-as.integer(as.vector(y_test))
x_test<-as.matrix(x_test)
x<-as.matrix(x)
x_test<-as.matrix(x_test)



levels(y) <- c(1,2)
levels(y_test) <- c(1,2)



keras_fit3 <- function(a,b){
  
  model <- keras_model_sequential()
  model %>%
    layer_dense(units = 20, activation = "relu", input_shape = c(16)) %>%
        layer_dropout(rate = a, seed = 123) %>% 
    layer_dense(units = 2, activation = "softmax")
  
  model %>% compile(
    loss = 'binary_crossentropy',
    optimizer = optimizer_adamax(lr = b),
    metrics = c('accuracy')
  )
  
  history <- model %>% fit(
    data.matrix(x), to_categorical(as.numeric(y),num_classes = 2),
    batch_size = 1424, 
    epochs = 400,
    verbose = 0,
    validation_split = 0.3
  )
  
  history
  
  result <- list(Score = history[["metrics"]][["accuracy"]][15], 
                 Pred = 0)
  
  return(result)
}


search_bound_keras <- list(a = c(0,0.8),b=c(0, 1))

search_grid_keras <- data.frame(a = runif(40, 0, 0.7),b=runif(40, 0, 1))

search_grid_keras

bayes_keras <- BayesianOptimization(FUN = keras_fit3, bounds = search_bound_keras, 
                                    init_points = 3, init_grid_dt = search_grid_keras, 
                                    n_iter = 10,verbose = TRUE,acq = "ucb")

save(bayes_keras,file="BayesianOPT_OUTPUT.rda")

bayes_keras_plot<-bayes_keras$History

colnames(bayes_keras_plot)<-c("Round","Dropout","LR","Accuracy_VAL")

ggplot(bayes_keras_plot , aes(Dropout,LR,z=Accuracy_VAL,color="red"))+geom_density_2d(aes(colour=Accuracy_VAL),binwidth = 0.1)+theme_bw()+theme(text = element_text(size=20))





model <- keras_model_sequential()
model %>%
  layer_dense(units = 20, activation = "relu", input_shape = c(16)) %>%
  layer_dropout(rate = 0.2536) %>% 
  layer_dense(units = 2, activation = "softmax")

model %>% compile(
  loss = 'binary_crossentropy',
  optimizer = optimizer_adamax(lr=0.2062),
  metrics = c('accuracy')
)

history <- model %>% fit(
  data.matrix(x), to_categorical(as.numeric(y),num_classes = 2),
  batch_size = 1424, 
  epochs = 400,
  verbose = 0,
  validation_split = 0.3,
)

plot(history,smooth = FALSE)



p<-keras_fit(0.1,0.1)

v<-predict(model,x_test)
d<- data.frame(ifelse(v[,1] < 0.5, "1", "0"))

cf_nn<-data.frame(d[,1])
cf_nn["T"]<-y_test
colnames(cf_nn)<-c("P","T")
table(cf_nn)
caret::confusionMatrix(table(cf_nn),mode = "everything")


pred_nn<-prediction(as.numeric(cf_nn$P),as.numeric(cf_nn$T))

pred_nn.perf <- performance(pred_nn, measure = "tpr", x.measure = "fpr")

perf_nn<-data.frame(pred_nn.perf@y.values)
perf_nn["X"]<-data.frame(pred_nn.perf@x.values)
colnames(perf_nn)<-c("TP","FP")

roc_nn.phi <- performance(pred_nn, measure ="phi")
roc_nn.mi <- performance(pred_nn, measure ="mi")

fourfoldplot(table(cf_nn), color = c("red","darkgreen"),conf.level = 0,margin = 1)

ggplot(perf_nn, aes(y=TP,x=FP,color="red")) + geom_point() +geom_line() + theme_bw()+ theme(legend.position="none")



keras_fit <- function(a){
  
  model <- keras_model_sequential()
  model %>%
    layer_dense(units = 20, activation = "relu", input_shape = c(16)) %>%
    layer_dropout(rate = a[1]) %>% 
    layer_dense(units = 2, activation = "softmax")
  
  model %>% compile(
    loss = 'binary_crossentropy',
    optimizer = optimizer_adamax(lr = a[2]),
    metrics = c('accuracy')
  )
  
  history <- model %>% fit(
    data.matrix(x), to_categorical(as.numeric(y),num_classes = 2),
    batch_size = 1424, 
    epochs = 200,
    verbose = 0,
    validation_split = 0.3,
  )
  
  result <- history$metrics$val_accuracy[15]
  return(result)
}

#The following step will takes at least 10 h

pso_keras <- hydroPSO( fn = function(a) {-keras_fit(a)}, lower = c(0, 0), upper = c(0.8, 1),control=list(npart=5))


