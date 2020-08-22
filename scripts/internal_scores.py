import math

import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn import metrics
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.preprocessing import OneHotEncoder
import scripts.validation_open_ensembles as val
from scipy.spatial import distance


class validation:
    def maxAriPerClass(self, truth, pred):
        """
        This method one hot encodes every class and for each class predicted checks the separation
        with each individual class in the truth by ARI score.
        This method returns the maximum score
        """
        labels1 = np.asarray(
            OneHotEncoder(categories='auto').fit_transform(truth.reshape(
                -1, 1)).todense())
        labels2 = np.asarray(
            OneHotEncoder(categories='auto').fit_transform(pred.reshape(
                -1, 1)).todense())

        all_aris = []
        for predicted_class in range(labels2.shape[1]):
            for truth_class in range(labels1.shape[1]):
                all_aris.append(
                    adjusted_rand_score(labels2[:, predicted_class],
                                        labels1[:, truth_class]))
        max_ari = max(all_aris)
        return max_ari

    def ari(self, truth, pred):
        """
        Sklearn's adjusted_rand_score
        """
        ari = adjusted_rand_score(truth, pred)
        return ari
    def ratkowsky_lance(self, data, pred):
        """
        The Ratkowsky-Lance index adaptation which gives
        a higher score to subspaces having similar internal qualities 
        but more features.
        This adaptation encourages the genetic algorithm to discover
        subspaces of maximal size.
        """

        list_divide = []
        n_dim = data.shape[1]
        n_clusters = len(np.unique(pred))
        #iterate through the n_dim
        for i in range(n_dim):
            bgssj = 0
            tssj = 0
            feature_data = data[:, i]
            feature_center = np.mean(feature_data)
            # compute between group dispersion
            for j in np.unique(pred):
                indices = [t for t, x in enumerate(pred) if x == j]
                feature_cluster = data[indices, i]
                feature_cluster_center = np.mean(feature_cluster)
                bgssj = bgssj + len(indices) * math.pow(
                    feature_cluster_center - feature_center, 2)
            # compute total scattering or total sum of squuares
            for member in feature_data:
                tssj = tssj + math.pow(member - feature_center, 2)
            if tssj == 0:
                tssj = 1e-5
            list_divide.append(bgssj / tssj)
        r = sum(list_divide) / n_dim
        score = r / n_clusters
        return score

    def adapted_ratkowsky_lance(self, data, pred):
        """
        The Ratkowsky-Lance index adaptation which gives
        a higher score to subspaces having similar internal qualities 
        but more features.
        This adaptation encourages the genetic algorithm to discover
        subspaces of maximal size.
        """
        n_dim = data.shape[1]
        score = self.ratkowsky_lance(data, pred)
        # Adaptation for monotony
        score = score * (n_dim / (n_dim + 1))
#         score = score * ((n_dim)**(1/3))
        return score
    def point_biserial(self, data, pred):
        v = val.validation(data, pred)
        method = "point_biserial"
        score =  getattr(v, method)()
        return score
    def PBM_index(self, data, pred):
        v = val.validation(data, pred)
        method = "PBM_index"
        score =  getattr(v, method)()
        return score

    def silhouette(self, data, pred):
        """
        Silhouette: Compactness and connectedness combination that measures a ratio of within cluster distances to closest neighbors
        outside of cluster. This uses sklearn.metrics version of the Silhouette.
        """
        if len(np.unique(pred)) ==1:
            return 0
        score = metrics.silhouette_score(data, pred, metric='euclidean')
        return score
    
    def adapted_silhouette(self, data, pred):
        """
        Silhouette: Compactness and connectedness combination that measures a ratio of within cluster distances to closest neighbors
        outside of cluster. This uses sklearn.metrics version of the Silhouette.
        """
        if len(np.unique(pred)) ==1:
            return 0
        n_dim = data.shape[1]
        score = metrics.silhouette_score(data, pred, metric='euclidean')
        # Adaptation for monotony
        score = score * (n_dim / (n_dim + 1))
        return score
                         
    def silhouette_minkowski(self, data, pred):
        """
        Silhouette: Compactness and connectedness combination that measures a ratio of within cluster distances to closest neighbors
        outside of cluster. This uses sklearn.metrics version of the Silhouette.
        """

        D = squareform(pdist(data, 'minkowski', 0.5))
        score = metrics.silhouette_score(D, pred, metric='precomputed')

        return score
    
    def Wemmert_Gancarski(self, dataMatrix, classLabel):
        """
        The Wemmert-Gancarski index, the quotients of distances between the points and the barycenters of all clusters, a measure of compactness
        """
        if len(np.unique(classLabel)) == 1:
            return 0
        self.description = "The Wemmert-Gancarski index, a measure of compactness"
        sum = 0
        list_centers = []
        attributes = len(dataMatrix[0])
        numObj = len(classLabel)
        numCluster = len(np.unique(classLabel))
        #compute all the centers
        for i in np.unique(classLabel):
            indices = [t for t, x in enumerate(classLabel) if x == i]
            clusterMember = dataMatrix[indices, :]
            #compute the center of the cluster
            list_centers.append(np.mean(clusterMember, 0))
        #iterate the clusters again for Rm
        for i , ii in enumerate(np.unique(classLabel)):
            sumRm = 0
            indices = [t for t, x in enumerate(classLabel) if x == ii]
            clusterMember = dataMatrix[indices, :]
            #compute the currrent center
            clusterCenter = np.mean(clusterMember, 0)
            tempList = list_centers
            tempList = tempList[:i] + tempList[i + 1:]
            #iterate through the member and compute rm
            for member in clusterMember:
                #make it a 2d array
                memberArray = np.zeros((1, attributes))
                memberArray[0, :] = member
                #compute the pair wise distance
                list_dis = distance.cdist(memberArray, tempList)
                sumRm = sumRm + (distance.euclidean(member, clusterCenter)) / min(
                    min(list_dis))
            #compute the sum
            sum = sum + max([0, len(indices) - sumRm])
        #compute the fitness
        score = sum / numObj
        n_dim = dataMatrix.shape[1]
        score = score * (n_dim / (n_dim + 1))
        return score

    def calinski_harabasz(self, data, pred):

        n_dim = data.shape[1]

        score =metrics.calinski_harabasz_score(data, pred)

        score = (score) +(score)*0.25 * n_dim
        return score