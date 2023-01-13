#                                 INTROUCCION A MACHINE LEARNING
## 1.PREPARACION DEL ENTORNO
################################################################################################


## LIMPIAR MI ESPACIO DE TRABAJO
rm(list = ls())
## CARGA DE PAQUETES 
library(ggplot2)
library(igraph)
library(clusterSim)
library(caret)

## A. UBICAR EL DIRECTORIO
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
getwd()
## B. CARGAR LA BASE
base<- read.csv("delanteros1718.csv")

##convertir datos a factor
base[,1]<- as.factor(base[,1])
base[,2]<- as.factor(base[,2])
###############################################################################################
## ANALISIS DESCRIPTIVO BASICO
###############################################################################################

dim(base)
str(base)
summary(base)

##GRAFICOS DE VARIABLES
ggplot(base,aes(x = puntos))+
  geom_histogram(bins = 10)

##GRAFICOS INTERACTIVOS CON PLOT_LY
plotly::plot_ly(y= ~base[,3], type= "box")
xray::distributions(base[,3:6])
xray::distributions(base[,7:10])
xray::distributions(base[,8:11])

##############################################################################################
##ANALISIS DE CORRELACION
#############################################################################################
correlacion <- cor(base[c(3:12)])
corrplot::corrplot(correlacion, method="square")

##CORRELACIONES A TRAVES DE GRAFOS
correlacion_g <-graph.adjacency(correlacion,
                                weighted = TRUE,
                                diag = FALSE,
                                mode = "upper")
  ##pintar correlaciones por encima de la media
cut.off <- mean(E(correlacion_g)$weight)
correlacion_g2 <- delete_edges(correlacion_g,E(correlacion_g)[weight< cut.off])
  ##agrupamos usanño una representacion de comunidades
correlacion_g2_c <- cluster_fast_greedy(correlacion_g2)
  ##visualizar
plot(correlacion_g2_c, correlacion_g2,
     vertex.size = colSums(correlacion)*10,
     vertex.frame.color = NA,
     vertex.laberl.color = "black",
     vertex.label.cex = 0.8,
     edge.width = E(correlacion_g2)$weight *15,
     layout = layout_with_fr(correlacion_g2),
     main = "Relación variable jugadores"
     )

############################################################################################
##1. METODOS DE APRENDIZAJE NO SUPERVISADO
############################################################################################

##ANALISIS DE CLUSTERS(DETECTA GRUPOS DE COMPORTAMIENTOS SIMILARES)
##ANALISIS DE ASOCIACION(DETECCION DE COORCURENCIAS)
##ANALISIS DE COMPONENTES PRINCIPALES(REDUCCION DE DIMENSIONES O QUITAR VARIABLES QUE NO APORTAN)

### 1. ANALISIS DE COMPONENTES PRINCIPALES
pca_futbol <- prcomp(as.matrix(base[, c(3:12)]), scale = TRUE, center = TRUE)
pca_futbol
summary(pca_futbol)
##con 3 componentes principales me puedo quedar con el 75% de la información


### 2. ANALISIS CLUSTERS

   ## A. preparacion de datos
str(base)
base_num <- base[,c(3:12)]
boxplot(base_num, las =2)
datos_norm = data.Normalization(base_num, type ="n4",normalization = "column")

par(mfrow=c(1:1))
boxplot(base_num, las =2)
boxplot(datos_norm)

    ## B. Encontrar el numero optimo de perfiles
set.seed(123)
           ## Buscar perfiles en unn rango posible de  k = 2 a k = 15.
k.max <- 15
wss <- sapply(1:k.max, 
              function(k){kmeans(datos_norm, k, nstart=50,iter.max = 15 )$tot.withinss})
wss
plot(1:k.max, wss,
     type="b", pch = 19, frame = FALSE, 
     xlab="Number of clusters K",
     ylab="Total within-clusters sum of squares")

    ### c. Aplicamos el metodo de k means
kc <- kmeans(na.omit(datos_norm), 6)
kc$centers
     # Variamos algunos parÃ¡metros para visualizar mejor el grÃ¡fico
clusplot(na.omit(base[3:12]), kc$cluster, color=TRUE, shade=TRUE, labels=2, lines=0)
    
   ### D. Le asignamos a cada jugador su cluster correspondiente, como una caracterÃ­stica mÃ¡s
base$Cluster <- kc$cluster 
      # Obtener el significado de los clÃºsters
aggregate(datos_norm, by=list(kc$cluster), FUN=mean)

###########################################################################################
##2. METODOS DE APRENDIZAJE SUPERVISADO
##########################################################################################

##MODELOS DE REGRESIONES DE CLASIFICACION (DE VARIABLES CATEGORICAS)
##MODELOS DE REGRESIONES LINEALES(DE VARIABLES CUANTITATIVAS)

### 1. MODELOS DE REGRESIONES LINEALES 
    ## A. Particionar la data en tet y training
set.seed(42)
index <- createDataPartition(base$puntos, p = 0.7, list = FALSE)
train_data <- base[index, ]
test_data  <- base[-index, ]

    ## B. Desarrollo del modelo lineal
model_glm <- caret::train(puntos ~ penaltiesFallados+asistencias+amarillas+sustituido+titular+goles,
                          data = train_data,
                          method = "glm",
                          preProcess = c("scale", "center"),
                          trControl = trainControl(method = "repeatedcv", 
                                                   number = 10, 
                                                   repeats = 10, 
                                                   savePredictions = TRUE, 
                                                   verboseIter = FALSE))
    
    ## C. verificacion de parametros de performance del modelo
model_glm
summary(model_glm)


    ## D. evaluar el performance sobre los datos de test
predictions <- predict(model_glm, test_data)

data.frame(actual = test_data$puntos,
           predicted = predictions) %>%
  ggplot(aes(x = actual, y = predicted)) +
  geom_jitter() +
  geom_smooth(method = "lm")

    ##E Evaluacion de variables mas importantes
imp <- varImp(model_glm, scale = FALSE)
plot(imp)

