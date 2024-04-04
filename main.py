#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 22:38:54 2024

@author: adrian kuba
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import Voronoi, voronoi_plot_2d
from sklearn.cluster import KMeans, DBSCAN
from sklearn.datasets import load_wine, load_iris, load_breast_cancer
from sklearn.metrics import silhouette_score, adjusted_rand_score, homogeneity_score, completeness_score, \
    v_measure_score
from sklearn.preprocessing import MinMaxScaler


def load_from_csv(paths):
    return [pd.read_csv(path, sep=";", header=None) for path in paths]


def scale_datasets(datasets):
    scaled_datasets = []
    scaler = MinMaxScaler(feature_range=(-5, 5))
    for dataset in datasets:
        scaled_data = scaler.fit_transform(dataset.iloc[:, 0:2])
        scaled_df = pd.DataFrame(scaled_data, columns=dataset.columns[0:2])
        if dataset.shape[1] > 2:
            scaled_df[dataset.columns[2]] = dataset.iloc[:, 2].values
        scaled_datasets.append(scaled_df)
    return scaled_datasets


def plot_voronoi(axis, dataset, labels):
    points = dataset.iloc[:, 0:2].to_numpy()
    # todo
    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    x_margin = abs(x_max - x_min) * 0.1
    y_min, y_max = points[:, 1].min(), points[:, 1].max()
    y_margin = abs(y_max - y_min) * 0.1

    # todo
    points = np.append(points, [[999, 999], [-999, 999], [999, -999], [-999, -999]], axis=0)
    vor = Voronoi(points)
    voronoi_plot_2d(vor, ax=axis, show_vertices=False, show_points=False, show_region=True, show_points_label=False)
    for r in range(len(vor.point_region)):
        region = vor.regions[vor.point_region[r]]
        if not -1 in region:
            polygon = [vor.vertices[i] for i in region]
            axis.fill(*zip(*polygon), color=plt.cm.Set1(labels[r], alpha=0.5))
    axis.scatter(dataset.iloc[:, 0], dataset.iloc[:, 1], c=dataset.iloc[:, 2])

    # todo
    axis.set_xlim(x_min - x_margin, x_max + x_margin)
    axis.set_ylim(y_min - y_margin, y_max + y_margin)


def calculate_kmeans(dataset, n_clusters):
    return KMeans(n_clusters=n_clusters, random_state=0).fit(dataset.iloc[0:, 0:2].to_numpy())


def calculate_dbscan(dataset, eps):
    # todo get rid of magic number
    return DBSCAN(eps=eps, min_samples=1).fit(dataset.iloc[0:, 0:2].to_numpy())


def charts_clusters(datasets, cluster_results):
    fig, ax = plt.subplots(1, 6, figsize=(30, 5))
    fig.tight_layout()
    for dataset, axis, cluster_result in zip(datasets, ax, cluster_results):
        plot_voronoi(axis, dataset, cluster_result.labels_)


# todo change name of the function
def charts_clusters_real(datasets):
    fig, ax = plt.subplots(1, 6, figsize=(30, 5))
    # todo
    fig.tight_layout()
    # todo
    fig.subplots_adjust(top=0.9)
    plt.suptitle('Clusters real', fontsize=20)
    for dataset, axis in zip(datasets, ax):
        labels = dataset.iloc[0:, 2]
        plot_voronoi(axis, dataset, labels)


# todo change name of the function
def charts_kmeans_clusters(datasets):
    fig, ax = plt.subplots(1, 6, figsize=(30, 5))
    fig.tight_layout()
    fig.subplots_adjust(top=0.9)
    plt.suptitle('Clustering kmeans', fontsize=20)
    for dataset, axis in zip(datasets, ax):
        kmeans = calculate_kmeans(dataset, 2)
        plot_voronoi(axis, dataset, kmeans.labels_)


def calculate_kmeans_silhouette_score(dataset, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    labels = kmeans.fit_predict(dataset.iloc[:, 0:2])
    return silhouette_score(dataset.iloc[:, 0:2], labels)


def calculate_kmeans_silhouette_scores(datasets, n_clusters_list):
    all_silhouette_scores = []
    for dataset in datasets:
        silhouette_scores = []
        for n_clusters in n_clusters_list:
            silhouette_score = calculate_kmeans_silhouette_score(dataset, n_clusters)
            silhouette_scores.append(silhouette_score)
        all_silhouette_scores.append(silhouette_scores)
    return all_silhouette_scores


def charts_kmeans_silhouette(all_silhouettes, n_clusters_list):
    fig, ax = plt.subplots(1, 6, figsize=(30, 5))
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.1)
    for silhouettes, axis in zip(all_silhouettes, ax):
        axis.plot(n_clusters_list, silhouettes)
        axis.set_xlabel("n_clusters")
        for n_clusters in n_clusters_list:
            axis.axvline(x=n_clusters, color='gray', linestyle='--', alpha=0.5)


def charts_dbscan_silhouette(all_silhouettes, eps_list, all_cluster_counts):
    fig, ax = plt.subplots(1, 6, figsize=(30, 5))
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.1)
    for silhouettes, axis, cluster_counts in zip(all_silhouettes, ax, all_cluster_counts):
        top, bottom = axis.get_ylim()
        print(top, bottom)
        axis.plot(eps_list, silhouettes)
        axis.set_xlabel("eps")

        for eps in eps_list:
            axis.axvline(x=eps, color='gray', linestyle='--', alpha=0.5)
        for claster_counts, eps in zip(cluster_counts, eps_list):
            axis.text(eps, 0, claster_counts, ha='center', va='bottom', transform=axis.transAxes)


def calculate_kmeans_silhouettes(datasets, n_clusters_list):
    all_silhouettes = []
    for dataset in datasets:
        silhouettes = []
        for n_clusters in n_clusters_list:
            kmeans = KMeans(n_clusters=n_clusters, random_state=0)
            silhouette = silhouette_score(dataset.iloc[0:, 0:2].to_numpy(),
                                          kmeans.fit_predict(dataset.iloc[0:, 0:2].to_numpy()))
            silhouettes.append(silhouette)
        all_silhouettes.append(silhouettes)
    return all_silhouettes


def calculate_dbscan_silhouettes(datasets, eps_list):
    all_silhouettes = []
    all_cluster_counts = []
    for dataset in datasets:
        silhouettes = []
        cluster_counts = []
        for eps in eps_list:
            dbscan = DBSCAN(eps=eps, min_samples=1)
            labels = dbscan.fit_predict(dataset.iloc[0:, 0:2].to_numpy())
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            cluster_counts.append(n_clusters)
            if n_clusters < 2 or len(labels) <= n_clusters:
                silhouette = float('nan')
            else:
                silhouette = silhouette_score(dataset.iloc[0:, 0:2].to_numpy(), labels)
            silhouettes.append(silhouette)
        all_silhouettes.append(silhouettes)
        all_cluster_counts.append(cluster_counts)
    # todo
    return all_silhouettes, all_cluster_counts


# change name key
def calculate_min_max_silhouette_indices(all_silhouettes):
    sil_min_indicies = []
    sil_max_indicies = []
    for silhouettes in all_silhouettes:
        sil_min_indicies.append(min(enumerate(silhouettes), key=(lambda x: x[1]))[0])
        sil_max_indicies.append(max(enumerate(silhouettes), key=(abs and (lambda x: x[1])))[0])
    return sil_min_indicies, sil_max_indicies


# todo change name
def calculate_metrics(dataset, cluster_results):
    y_true = dataset.iloc[0:, 2]
    adj = adjusted_rand_score(y_true, cluster_results.fit_predict(dataset.iloc[0:, 0:2].to_numpy()))
    homo = homogeneity_score(y_true, cluster_results.fit_predict(dataset.iloc[0:, 0:2].to_numpy()))
    como = completeness_score(y_true, cluster_results.fit_predict(dataset.iloc[0:, 0:2].to_numpy()))
    vbeta1 = v_measure_score(y_true, cluster_results.fit_predict(dataset.iloc[0:, 0:2].to_numpy()), beta=0.5)
    vbeta2 = v_measure_score(y_true, cluster_results.fit_predict(dataset.iloc[0:, 0:2].to_numpy()), beta=1)
    vbeta3 = v_measure_score(y_true, cluster_results.fit_predict(dataset.iloc[0:, 0:2].to_numpy()), beta=2)
    return [adj, homo, como, vbeta1, vbeta2, vbeta3]


def calculate_metrics2(y_true, dataset, cluster_results):
    adj = adjusted_rand_score(y_true, cluster_results.fit_predict(dataset))
    homo = homogeneity_score(y_true, cluster_results.fit_predict(dataset))
    como = completeness_score(y_true, cluster_results.fit_predict(dataset))
    vbeta1 = v_measure_score(y_true, cluster_results.fit_predict(dataset), beta=0.5)
    vbeta2 = v_measure_score(y_true, cluster_results.fit_predict(dataset), beta=1)
    vbeta3 = v_measure_score(y_true, cluster_results.fit_predict(dataset), beta=2)
    return [adj, homo, como, vbeta1, vbeta2, vbeta3]


# todo change name of the function
def chart_kmeans_measures(datasets, n_clusters_list):
    fig, ax = plt.subplots(1, 6, figsize=(30, 5))
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.1)
    for dataset, axis in zip(datasets, ax):
        k_meansNN_rate = []  # wartosci miary silhouette
        for n_clusters in n_clusters_list:
            kmeans = KMeans(n_clusters=n_clusters, random_state=0)
            metrics = calculate_metrics(dataset, kmeans)
            k_meansNN_rate.append(metrics)
        axis.plot(n_clusters_list, k_meansNN_rate)
        axis.set_xlabel("n_cluster")
        axis.legend(["adjusted rand", "homogeneity", "completeness", "v-measure beta=0.5", "v-measure beta=1",
                     "v-measure beta=2"])
        for n_clusters in n_clusters_list:
            axis.axvline(x=n_clusters, color='gray', linestyle='--', alpha=0.5)


def chart_dbscan_measures(datasets, eps_list, all_cluster_counts):
    fig, ax = plt.subplots(1, 6, figsize=(30, 5))
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.1)
    for dataset, axis, cluster_counts in zip(datasets, ax, all_cluster_counts):

        dbscan_NN_rate = []  # wartosci miary silhouette
        for eps in eps_list:
            dbscan = calculate_dbscan(dataset, eps)
            metrics = calculate_metrics(dataset, dbscan)
            dbscan_NN_rate.append(metrics)

        axis.plot(eps_list, dbscan_NN_rate)
        axis.set_xlabel("eps")
        axis.legend(["adjusted rand", "homogeneity", "completeness", "v-measure beta=0.5", "v-measure beta=1",
                     "v-measure beta=2"])
        for eps in eps_list:
            axis.axvline(x=eps, color='gray', linestyle='--', alpha=0.5)
        for claster_counts, eps in zip(cluster_counts, eps_list):
            axis.text(eps, 0, claster_counts, ha='center', va='bottom', transform=axis.transAxes)


# todo change name of the function
def charts_clusters_kmeans_silhouette(datasets, sils):
    fig, ax = plt.subplots(1, 6, figsize=(30, 5))
    fig.tight_layout()
    for dataset, sil, axis in zip(datasets, sils, ax):
        kmeans = calculate_kmeans(dataset, sil)
        labels = kmeans.labels_
        plot_voronoi(axis, dataset, labels)


# todo change name of the function
def charts_kmeans_metrics_clusters(datasets, metrics):
    fig, ax = plt.subplots(1, 6, figsize=(30, 5))
    fig.tight_layout()
    # todo
    for dataset, metric, axis in zip(datasets, metrics, ax):
        kmeans = calculate_kmeans(dataset, metric)
        plot_voronoi(axis, dataset, kmeans.labels_)


def charts_clusters_dbscan_silhouette(datasets, sils):
    fig, ax = plt.subplots(1, 6, figsize=(30, 5))
    fig.tight_layout()
    # todo
    for dataset, sil, axis in zip(datasets, sils, ax):
        dbscan = calculate_dbscan(dataset, sil)
        labels = dbscan.labels_
        plot_voronoi(axis, dataset, labels)


def main():
    paths = ["1_1.csv", "1_2.csv", "1_3.csv", "2_1.csv", "2_2.csv", "2_3.csv"]

    kmeans_met_max = [2, 2, 8, 2, 2, 7]
    kmeans_met_min = [11, 11, 2, 11, 11, 2]

    dbscan_met_max = [0.8, 0.7, 0.05, 0.55, 0.7, 0.35]
    dbscan_met_min = [0.05, 0.95, 0.55, 0.05, 0.05, 0.95]

    n_clusters_list = [n_clusters for n_clusters in range(2, 12)]
    # eps_list = [x / 100 for x in range(5, 100, 5)]
    eps_list = [x / 100 for x in range(20, 500, 20)]
    # eps_list = [x / 100 for x in range(40, 1000, 40)]

    iris = load_iris()
    chart_from_sklearn_dataset(iris, n_clusters_list, eps_list)
    wine = load_wine()
    # chart_from_sklearn_dataset(wine, n_clusters_list, eps_list)
    breast_cancer = load_breast_cancer()
    # chart_from_sklearn_dataset(breast_cancer, n_clusters_list, eps_list)

    # datasets_from_csv = load_from_csv(paths)
    # datasets_normalized = scale_datasets(datasets_from_csv)
    #
    # charts_clusters_real(datasets_normalized)
    #
    # kmeans_all_silhouettes = calculate_kmeans_silhouette_scores(datasets_normalized, n_clusters_list)
    # kmeans_sil_min_indices, kmeans_sil_max_indices = calculate_min_max_silhouette_indices(kmeans_all_silhouettes)
    # kmeans_sil_min = [n_clusters_list[i] for i in kmeans_sil_min_indices]
    # kmeans_sil_max = [n_clusters_list[i] for i in kmeans_sil_max_indices]
    #
    # charts_kmeans_silhouette(kmeans_all_silhouettes, n_clusters_list)
    # charts_clusters_kmeans_silhouette(datasets_normalized, kmeans_sil_min)
    # charts_clusters_kmeans_silhouette(datasets_normalized, kmeans_sil_max)
    #
    # chart_kmeans_measures(datasets_normalized, n_clusters_list)
    # charts_kmeans_metrics_clusters(datasets_normalized, kmeans_met_min)
    # charts_kmeans_metrics_clusters(datasets_normalized, kmeans_met_max)
    #
    # dbscan_all_silhouettes, all_cluster_counts = calculate_dbscan_silhouettes(datasets_normalized, eps_list)
    # dbscan_sil_min_indices, dbscan_sil_max_indices = calculate_min_max_silhouette_indices(dbscan_all_silhouettes)
    # dbscan_sil_min = [eps_list[i] for i in dbscan_sil_min_indices]
    # dbscan_sil_max = [eps_list[i] for i in dbscan_sil_max_indices]
    #
    # charts_dbscan_silhouette(dbscan_all_silhouettes, eps_list, all_cluster_counts)
    # charts_clusters_dbscan_silhouette(datasets_normalized, dbscan_sil_min)
    # charts_clusters_dbscan_silhouette(datasets_normalized, dbscan_sil_max)
    #
    # chart_dbscan_measures(datasets_normalized, eps_list, all_cluster_counts)
    # charts_clusters_dbscan_silhouette(datasets_normalized, dbscan_met_min)
    # charts_clusters_dbscan_silhouette(datasets_normalized, dbscan_met_max)

    plt.show()


def chart_from_sklearn_dataset(dataset, n_clusters_list, eps_list):
    X_data = dataset.data
    Y_target = dataset.target

    print(X_data)
    scaler = MinMaxScaler(feature_range=(-5, 5))
    X_scaled = scaler.fit_transform(X_data)

    print("X_scaled: ", X_scaled)

    sils_kmeans = []
    metrics_kmeans = []
    for n_clusters in n_clusters_list:
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        metrics = calculate_metrics2(Y_target, X_scaled, kmeans)
        metrics_kmeans.append(metrics)
        y_kmeans = kmeans.fit_predict(X_scaled)
        sils_kmeans.append(silhouette_score(X_scaled, y_kmeans))

    sils_dbscan = []
    metrics_dbscan = []
    for eps in eps_list:
        dbscan = DBSCAN(eps=eps, min_samples=1)
        metrics = calculate_metrics2(Y_target, X_scaled, dbscan)
        metrics_dbscan.append(metrics)
        y_dbscan = dbscan.fit_predict(X_scaled)
        n_clusters = len(set(y_dbscan)) - (1 if -1 in y_dbscan else 0)
        if n_clusters < 2 or len(y_dbscan) <= n_clusters:
            sils_dbscan.append(float('nan'))
        else:
            print("n_clusters: ", n_clusters)
            print("len(X_scaled):", len(X_scaled))
            print("len(y_dbscan):", len(y_dbscan))
            sils_dbscan.append(silhouette_score(X_scaled, y_dbscan))

    fig1, ax1 = plt.subplots(figsize=(5, 5))
    fig1.tight_layout()
    ax1.plot(n_clusters_list, sils_kmeans)
    ax1.set_xlabel("n_clusters")
    for n_clusters in n_clusters_list:
        ax1.axvline(x=n_clusters, color='gray', linestyle='--', alpha=0.5)

    fig2, ax2 = plt.subplots(figsize=(5, 5))
    fig2.tight_layout()
    ax2.plot(eps_list, sils_dbscan)
    ax2.set_xlabel("eps")
    for eps in eps_list:
        ax2.axvline(x=eps, color='gray', linestyle='--', alpha=0.5)

    fig3, ax3 = plt.subplots(figsize=(5, 5))
    fig3.tight_layout()
    ax3.plot(n_clusters_list, metrics_kmeans)
    ax3.set_xlabel("n_clusters")
    ax3.legend(["adjusted rand", "homogeneity", "completeness", "v-measure beta=0.5", "v-measure beta=1",
                "v-measure beta=2"])
    for n_clusters in n_clusters_list:
        ax3.axvline(x=n_clusters, color='gray', linestyle='--', alpha=0.5)

    fig4, ax4 = plt.subplots(figsize=(5, 5))
    fig4.tight_layout()
    ax4.plot(eps_list, metrics_dbscan)
    ax4.set_xlabel("eps")
    ax4.legend(["adjusted rand", "homogeneity", "completeness", "v-measure beta=0.5", "v-measure beta=1",
                "v-measure beta=2"])
    for eps in eps_list:
        ax4.axvline(x=eps, color='gray', linestyle='--', alpha=0.5)


if __name__ == "__main__":
    main()
