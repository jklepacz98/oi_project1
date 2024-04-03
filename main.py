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
from sklearn.metrics import silhouette_score, adjusted_rand_score, homogeneity_score, completeness_score, \
    v_measure_score


def load_from_csv(paths):
    return [pd.read_csv(path, sep=";", header=None) for path in paths]


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


# todo change name of the function
def charts_clusters(datasets):
    fig, ax = plt.subplots(1, 6, figsize=(30, 5))
    # todo
    fig.tight_layout()
    # todo
    fig.subplots_adjust(top=0.9)
    plt.suptitle('Clustering', fontsize=20)
    for dataset, axis in zip(datasets, ax):
        labels = dataset.iloc[0:, 2]
        plot_voronoi(axis, dataset, labels)


# todo change name of the function
def charts_kmeans_clusters(datasets):
    fig, ax = plt.subplots(1, 6, figsize=(30, 5))
    fig.tight_layout()
    for dataset, axis in zip(datasets, ax):
        kmeans = calculate_kmeans(dataset, 2)
        plot_voronoi(axis, dataset, kmeans.labels_)


def charts_dbscan_clusters(datasets):
    fig, ax = plt.subplots(1, 6, figsize=(30, 5))
    fig.tight_layout()
    for dataset, axis in zip(datasets, ax):
        dbscan = calculate_dbscan(dataset=dataset, eps=0.5)
        plot_voronoi(axis, dataset, dbscan.labels_)


def calculate_silhouette_score(dataset, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    labels = kmeans.fit_predict(dataset.iloc[:, 0:2])
    return silhouette_score(dataset.iloc[:, 0:2], labels)


def calculate_silhouette_scores(datasets, n_clusters_list):
    all_silhouette_scores = []
    for dataset in datasets.values():
        silhouette_scores = []
        for n_clusters in n_clusters_list:
            silhouette_score = calculate_silhouette_score(dataset, n_clusters)
            silhouette_scores.append(silhouette_score)
        all_silhouette_scores.append(silhouette_scores)
    return all_silhouette_scores


# todo change name of the function
def chart_silhoueete(datasets, n_clusters_list, sil_max, sil_min):
    # grupa wykresów z miarą silhoueete
    fig, ax = plt.subplots(1, 6, figsize=(30, 5))
    fig.tight_layout()

    k_meansN_rate = []
    linear_space = []  # ilosc klastorw

    for i in range(len(datasets)):  # dzialanie na 6 zbiorach
        for n_clusters in n_clusters_list:  # ustanowienie ilosci zbiorow

            kmeans = KMeans(n_clusters=n_clusters, random_state=0)  # ustanowienie ilosci klastrow w metodzie Kmeans
            sil = silhouette_score(datasets[i].iloc[0:, 0:2].to_numpy(),
                                   kmeans.fit_predict(datasets[i].iloc[0:, 0:2].to_numpy()))  # miara silhoueete
            k_meansN_rate.append(sil)
            linear_space.append(n_clusters - 1)

            if n_clusters == 2:  # czyszczenie danych z listy

                k_meansN_rate.clear()
                linear_space.clear()

        # selekcja minimow i maksimow
        sil_max.append(max(enumerate(k_meansN_rate), key=(abs and (lambda x: x[1])))[0] + 2)
        sil_min.append(min(enumerate(k_meansN_rate), key=(lambda x: x[1]))[0] + 2)
        # rysowanie punktow w subplots()
        ax[i].set_xlabel("n_cluster")
        ax[i].plot(linear_space, k_meansN_rate)  #


# todo change name of the function
def chart_measures(datasets, n_clusters_list):
    fig, ax = plt.subplots(1, 6, figsize=(30, 5))
    fig.tight_layout()

    for dataset, axis in zip(datasets, ax):
        y_true = dataset.iloc[0:, 2]
        k_meansNN_rate = []  # wartosci miary silhoueete
        linspacee = []
        for n_clusters in n_clusters_list:
            kmeans = KMeans(n_clusters=n_clusters, random_state=0)

            # reszta miar
            adj = adjusted_rand_score(y_true, kmeans.fit_predict(dataset.iloc[0:, 0:2].to_numpy()))
            homo = homogeneity_score(y_true, kmeans.fit_predict(dataset.iloc[0:, 0:2].to_numpy()))
            como = completeness_score(y_true, kmeans.fit_predict(dataset.iloc[0:, 0:2].to_numpy()))
            vbeta1 = v_measure_score(y_true, kmeans.fit_predict(dataset.iloc[0:, 0:2].to_numpy()), beta=0.5)
            vbeta2 = v_measure_score(y_true, kmeans.fit_predict(dataset.iloc[0:, 0:2].to_numpy()), beta=1)
            vbeta3 = v_measure_score(y_true, kmeans.fit_predict(dataset.iloc[0:, 0:2].to_numpy()), beta=2)

            # dodawanie miar do listy
            k_meansNN_rate.append([adj, homo, como, vbeta1, vbeta2, vbeta3])
            # todo understand why here is -1 and get rid of it
            linspacee.append(n_clusters - 1)

        # rysowanie od razu 6 przebiegów na każdym pojedynczym wykresie
        axis.plot(linspacee, k_meansNN_rate)  #
        axis.set_xlabel("n_cluster")
        axis.legend(["adjusted rand", "homogeneity", "completeness", "v-measure beta=0.5", "v-measure beta=1",
                     "v-measure beta=2"])


# todo change name of the function
def charts_sil_clusters(datasets, sils):
    fig, ax = plt.subplots(1, 6, figsize=(30, 5))
    fig.tight_layout()
    # todo
    for dataset, sil, axis in zip(datasets, sils, ax):
        # kmeans = KMeans(n_clusters=sil, random_state=0).fit(datasets[i].iloc[0:, 0:2].to_numpy())
        kmeans = calculate_kmeans(dataset, sil)
        labels = kmeans.labels_
        plot_voronoi(axis, dataset, labels)


# todo change name of the function
def charts_metrics_clusters(datasets, metrics):
    # wykresy najlepszy podzial dla reszty miar
    fig, ax = plt.subplots(1, 6, figsize=(30, 5))
    fig.tight_layout()
    # todo
    for dataset, metric, axis in zip(datasets, metrics, ax):
        kmeans = calculate_kmeans(dataset, metric)
        plot_voronoi(axis, dataset, kmeans.labels_)


def main():
    # ścieżki do plików
    paths = ["1_1.csv", "1_2.csv", "1_3.csv", "2_1.csv", "2_2.csv", "2_3.csv"]

    sil_max = []  # maksyma silhoueete
    sil_min = []  # minima silhoueete

    met_max = [2, 2, 6, 2, 3, 3]  # maksima innych miar
    met_min = [10, 10, 7, 10, 10, 4]  # minima innych miar

    n_clusters_list = [n_clusters for n_clusters in range(2, 15)]
    eps_list = [x / 100 for x in range(5, 165, 10)]

    datasets_from_csv = load_from_csv(paths)

    charts_clusters(datasets_from_csv)
    charts_kmeans_clusters(datasets_from_csv)

    # todo get rid of
    charts_dbscan_clusters(datasets_from_csv)

    chart_silhoueete(datasets_from_csv, n_clusters_list, sil_max, sil_min)
    chart_measures(datasets_from_csv, n_clusters_list)
    charts_sil_clusters(datasets_from_csv, sil_max)
    charts_sil_clusters(datasets_from_csv, sil_min)
    charts_metrics_clusters(datasets_from_csv, met_max)
    charts_metrics_clusters(datasets_from_csv, met_min)

    plt.show()


if __name__ == "__main__":
    main()
