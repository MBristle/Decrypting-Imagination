preproControl<-function(data){

library(tidyverse)

#Filter for not NA
data<-filter(data,!is.na(stimPic))
#Remove unused Cols
data<-select(data,participant,session,inSet,id,kategory,response.corr,response.rt,trials.thisTrialN)
data<-rename(data, corrResp= response.corr,rtResp=response.rt, trialN=trials.thisTrialN)
data<-separate(data = data,col = participant,into = c("participant"), sep = "_",extra = "drop")
col <- c("participant","session", "inSet","id",
         "kategory","trialN")
data[col]<-lapply(data[col], factor)
return(data)
}
