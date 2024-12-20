library(ComBatFamily)

setwd("/home/sherry/Dropbox/PhD/MATCH")
path_TDC = "./Connectome Data/ForRcode/QCpass_singleshell/TDC"
path_ASD = "./Connectome Data/ForRcode/QCpass_singleshell/ASD"
log = FALSE
method = 'combat'

if (log) {
  conn_TDC = read.csv(file.path(path_TDC, "desikan_log_edges.csv"), row.names = "Subject")
  conn_ASD = read.csv(file.path(path_ASD, "desikan_log_edges.csv"), row.names = "Subject")
  out_name = paste('desikan_edges_log', method, sep='')
} else {
  conn_TDC = read.csv(file.path(path_TDC, "desikan_edges.csv"), row.names = "Subject")
  conn_ASD = read.csv(file.path(path_ASD, "desikan_edges.csv"), row.names = "Subject")
  out_name = paste('desikan_edges_', method, sep='')
}

batch_TDC = as.factor(read.csv(file.path(path_TDC, "batch.csv"))$index)
cohort_TDC = read.csv(file.path(path_TDC, "covbat_cohort.csv"), row.names = "Subject")
mod_TDC = read.csv(file.path(path_TDC, "mod.csv"))
batch_ASD = as.factor(read.csv(file.path(path_ASD, "batch.csv"))$index)
cohort_ASD = read.csv(file.path(path_ASD, "covbat_cohort.csv"), row.names = "Subject")
mod_ASD = read.csv(file.path(path_ASD, "mod.csv"))

if (method=='combat') {
  out_TDC = comfam(conn_TDC, batch_TDC, covar = cohort_TDC, model = lm, formula = y ~ AgeC + AgeC2 + Sex)
  out_ASD = predict(out_TDC, conn_ASD, batch_ASD, newcovar=cohort_ASD)
  conn_harmonized_TDC = cbind(Subject=row.names(cohort_TDC), out_TDC$dat.combat)
  conn_harmonized_ASD = cbind(Subject=row.names(cohort_ASD), out_ASD$dat.combat)
} else {
  out_TDC = covfam(conn_TDC, batch_TDC, covar = cohort_TDC, model = lm, formula = y ~ AgeC + AgeC2 + Sex)
  out_ASD = predict(out_TDC, conn_ASD, batch_ASD, newcovar=cohort_ASD)
  conn_harmonized_TDC = cbind(Subject=row.names(cohort_TDC), out_TDC$dat.covbat)
  conn_harmonized_ASD = cbind(Subject=row.names(cohort_ASD), out_ASD$dat.covbat)
}

save(out_TDC, file = file.path(path_TDC, paste(out_name, 'RData', sep='.')))
write.csv(conn_harmonized_TDC, file.path(path_TDC, paste(out_name, 'csv', sep='.')), row.names = FALSE)
write.csv(conn_harmonized_ASD, file.path(path_ASD, paste(out_name, 'csv', sep='.')), row.names = FALSE)

          