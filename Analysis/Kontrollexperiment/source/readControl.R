
readControl<-function(){
  library(readr)
  library(tidyverse)
  
  data_path<-"~/GitHub/Forschungsatelier/Kontrollexperiment/data/"
  files <- list.files(path = data_path, 
             pattern = "^[a-zA-Z][a-zA-Z]\\d\\d_\\d.*.csv" )
  
  data<-do.call("rbind",( 
    lapply(
      X=files,
      function(x){ 
        read_delim(paste0(data_path,x),",", escape_double = FALSE, trim_ws = TRUE)})))
  
  return(data)
}

        