
df <- read.csv('cleveland.csv')
attributes <- c('age', 'sex', 'cp', 'trestbps', 
               'chol', 'fbs', 'restecg', 'thalach',
               'exang', 'oldpeak', 'slope', 'ca',
               'thal', 'num (predicted)')

df_txt <- read.table('cleveland.txt', sep=',',col.names=attributes)

write.csv(df_txt, 'cleveland_processed.csv', row.names=FALSE)
