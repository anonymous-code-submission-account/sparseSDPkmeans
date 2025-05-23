require(sparcl)
require(mclust)
require(phyclust)


sparse_kmeans_ell1_ell2 <- function(x) {
  x_scaled <- scale(x, TRUE, TRUE)
  km.perm <- KMeansSparseCluster.permute(x_scaled, K = 2, wbounds = seq(3, 7, length.out = 15), nperms = 5)
  km.out <- KMeansSparseCluster(x_scaled, K = 2, wbounds = km.perm$bestw)
  return(km.out[[1]]$Cs)
}

sparse_hier_ell1_ell2 <- function(x) {
  x_scaled <- scale(x, TRUE, TRUE)
  perm.out <- HierarchicalSparseCluster.permute(x_scaled, wbounds=c(1.5,2:6),nperms=5)
  sparsehc <- HierarchicalSparseCluster(dists=perm.out$dists,
                                        wbound=perm.out$bestw, method="complete")
  return(cutree(sparsehc$hc, k = 2))
}

evaluate_clustering <- function(predicted, truth) {
  acc1 <- mean(predicted == truth)
  acc2 <- mean(predicted != truth)
  return(max(acc1, acc2))
}



sparse_kmeans_hillclimb <- function(x){
  cluster_est <- hill_climb(x,2,nbins=50,nperms=25,100,1e-5)
  return(cluster_est$best_result)
}

Alternate = # use the function provided by Ery Arias-Castro.

hill_climb = # #compute within-cluster distance by clustering feature by feature, initialize S of size s based on this. use the function provided by Ery Arias-Castro.
