#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 22:38:54 2024

@author: adrian
"""

import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial import Voronoi, voronoi_plot_2d
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, homogeneity_score, completeness_score, \
    v_measure_score


# wczytywanie plików
def load_from_csv(paths):
    return [pd.read_csv(path, sep=";", header=None) for path in paths]


def plot_voronoi(ax, points, labels):
    vor = Voronoi(points)
    voronoi_plot_2d(vor, ax=ax, show_vertices=False, show_points=False, show_region=True, show_points_label=False)
    for r in range(len(vor.point_region)):
        region = vor.regions[vor.point_region[r]]
        if not -1 in region:
            polygon = [vor.vertices[i] for i in region]
            ax.fill(*zip(*polygon), color=plt.cm.Set1(labels[r], alpha=0.5))


def plot_kmeans_clusters(ax, dataset, kmeans):
    labels = kmeans.labels_
    plot_voronoi(ax, dataset.iloc[:, 0:2].to_numpy(), labels)
    ax.scatter(dataset.iloc[:, 0], dataset.iloc[:, 1], c=labels)


def calculate_kmeans(dataset, n_clusters):
    return KMeans(n_clusters=n_clusters, random_state=0).fit(dataset.iloc[0:, 0:2].to_numpy())


# todo change name of the function
def charts(datasets):
    fig, ax = plt.subplots(1, 6, figsize=(30, 5))
    fig.tight_layout()
    for dataset, axis in zip(datasets, ax):
        labels = dataset.iloc[0:, 2]
        points = dataset.iloc[:, 0:2].to_numpy()
        plot_voronoi(axis, points, labels)


# todo change name of the function
def charts_kmeans(datasets):
    fig, ax = plt.subplots(1, 6, figsize=(30, 5))
    fig.tight_layout()
    for dataset, axis in zip(datasets, ax):
        kmeans = calculate_kmeans(dataset, 2)
        plot_kmeans_clusters(axis, dataset, kmeans)


# todo change name of the function
def chart_silhoueete(datasets, sil_max, sil_min):
    # grupa wykresów z miarą silhoueete
    fig3, ax3 = plt.subplots(1, 6, figsize=(30, 5))
    fig3.tight_layout()

    k_meansN_rate = []

    linspace = []  # ilosc klastorw

    for i in range(len(datasets)):  # dzialanie na 6 zbiorach
        for k in range(2, 12):  # ustanowienie ilosci zbiorow

            kmeans = KMeans(n_clusters=k, random_state=0)  # ustanowienie ilosci klastrow w metodzie Kmeans
            sil = silhouette_score(datasets[i].iloc[0:, 0:2].to_numpy(),
                                   kmeans.fit_predict(datasets[i].iloc[0:, 0:2].to_numpy()))  # miara silhoueete
            k_meansN_rate.append(sil)
            linspace.append(k - 1)

            if k == 2:  # czyszczenie danych z listy

                k_meansN_rate.clear()
                linspace.clear()

        # selekcja minimow i maksimow
        sil_max.append(max(enumerate(k_meansN_rate), key=(abs and (lambda x: x[1])))[0] + 2)
        sil_min.append(min(enumerate(k_meansN_rate), key=(lambda x: x[1]))[0] + 2)
        # rysowanie punktow w subplots()
        ax3[i].set_xlabel("n_cluster")
        ax3[i].plot(linspace, k_meansN_rate)  #


# todo change name of the function
def chart_measures(datasets):
    # grupa wykresów z reszta miar
    fig, ax = plt.subplots(1, 6, figsize=(30, 5))
    fig.tight_layout()

    k_meansNN_rate = []  # wartosci miary silhoueete
    linspacee = []  # ilosc klastorw

    for i in range(len(datasets)):
        for k in range(2, 12):

            y_true = datasets[i].iloc[0:, 2]

            kmeans = KMeans(n_clusters=k, random_state=0)

            # reszta miar
            adj = adjusted_rand_score(y_true, kmeans.fit_predict(datasets[i].iloc[0:, 0:2].to_numpy()))
            homo = homogeneity_score(y_true, kmeans.fit_predict(datasets[i].iloc[0:, 0:2].to_numpy()))
            como = completeness_score(y_true, kmeans.fit_predict(datasets[i].iloc[0:, 0:2].to_numpy()))
            vbeta1 = v_measure_score(y_true, kmeans.fit_predict(datasets[i].iloc[0:, 0:2].to_numpy()), beta=0.5)
            vbeta2 = v_measure_score(y_true, kmeans.fit_predict(datasets[i].iloc[0:, 0:2].to_numpy()), beta=1)
            vbeta3 = v_measure_score(y_true, kmeans.fit_predict(datasets[i].iloc[0:, 0:2].to_numpy()), beta=2)

            # dodawanie miar do listy
            k_meansNN_rate.append([adj, homo, como, vbeta1, vbeta2, vbeta3])
            linspacee.append(k - 1)
            if k == 2:
                k_meansNN_rate.clear()
                linspacee.clear()

        # rysowanie od razu 6 przebiegów na każdym pojedynczym wykresie
        ax[i].plot(linspacee, k_meansNN_rate)  #
        ax[i].set_xlabel("n_cluster")
        ax[i].legend(["adjusted rand", "homogeneity", "completeness", "v-measure beta=0.5", "v-measure beta=1",
                      "v-measure beta=2"])


# todo change name of the function
def charts_sil(datasets, sils):
    fig, ax = plt.subplots(1, 6, figsize=(30, 5))
    fig.tight_layout()
    # todo
    for dataset, sil, axis in zip(datasets, sils, ax):
        # kmeans = KMeans(n_clusters=sil, random_state=0).fit(datasets[i].iloc[0:, 0:2].to_numpy())
        kmeans = calculate_kmeans(dataset, sil)
        labels = kmeans.labels_
        points = dataset.iloc[0:, 0:2].to_numpy()
        plot_voronoi(axis, points, labels)


# todo change name of the function
def charts_metrics(datasets, metrics):
    # wykresy najlepszy podzial dla reszty miar
    fig, ax = plt.subplots(1, 6, figsize=(30, 5))
    fig.tight_layout()
    # todo
    for dataset, metric, axis in zip(datasets, metrics, ax):
        kmeans = calculate_kmeans(dataset, metric)
        plot_kmeans_clusters(axis, dataset, kmeans)


def main():
    # ścieżki do plików
    paths = ["1_1.csv", "1_2.csv", "1_3.csv", "2_1.csv", "2_2.csv", "2_3.csv"]

    sil_max = []  # maksyma silhoueete
    sil_min = []  # minima silhoueete

    met_max = [2, 2, 6, 2, 3, 3]  # maksima innych miar
    met_min = [10, 10, 7, 10, 10, 4]  # minima innych miar

    datasets_from_csv = load_from_csv(paths)

    charts(datasets_from_csv)
    charts_kmeans(datasets_from_csv)
    chart_silhoueete(datasets_from_csv, sil_max, sil_min)
    chart_measures(datasets_from_csv)
    charts_sil(datasets_from_csv, sil_max)
    charts_sil(datasets_from_csv, sil_min)
    charts_metrics(datasets_from_csv, met_max)
    charts_metrics(datasets_from_csv, met_min)

    plt.show()


if __name__ == "__main__":
    main()
