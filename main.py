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


# wykresy z etykietami prawdziwymi i przewidzianymi -rysowanie
def charts(datasets):
    # grupa wykresów z rzeczywistym podziałem
    fig1, ax1 = plt.subplots(1, 6, figsize=(30, 5))
    fig1.tight_layout()
    # grupa wykresów z przewidzianym podziałem Kmeans
    fig2, ax2 = plt.subplots(1, 6, figsize=(30, 5))
    fig2.tight_layout()
    for i in range(len(datasets)):

        #######################################################################################
        #############################  pierwsza grupa wykresów  ######################################
        #######################################################################################

        # rozdział danych
        x = datasets[i].iloc[0:, 0]
        y = datasets[i].iloc[0:, 1]
        y_true = datasets[i].iloc[0:, 2]

        # rysowanie diagramu voronoia z kolorami
        vor = Voronoi(datasets[i].iloc[0:, 0:2].to_numpy())
        voronoi_plot_2d(vor, ax=ax1[i], show_vertices=False, show_points=False, show_region=True,
                        show_points_label=False)
        for r in range(len(vor.point_region)):
            region = vor.regions[vor.point_region[r]]
            if not -1 in region:
                polygon = [vor.vertices[i] for i in region]
                ax1[i].fill(*zip(*polygon), color=plt.cm.Set1(y_true[r], alpha=0.5))

        # rysowanie punktow
        ax1[i].scatter(x, y, c=y_true)  #
        ax1[i].set_xlabel("2_cluster")

        #######################################################################################
        #############################    druga grupa wykresów    ######################################
        #######################################################################################

        kmeans = KMeans(n_clusters=2, random_state=0).fit(datasets[i].iloc[0:, 0:2].to_numpy())

        voronoi_plot_2d(vor, ax=ax2[i], show_vertices=False, show_points=False, show_region=True,
                        show_points_label=False)
        for r in range(len(vor.point_region)):
            region = vor.regions[vor.point_region[r]]
            if not -1 in region:
                polygon = [vor.vertices[i] for i in region]
                ax2[i].fill(*zip(*polygon), color=plt.cm.Set1(kmeans.labels_[r], alpha=0.5))

        ax2[i].scatter(x, y, c=kmeans.labels_)  #
        ax2[i].set_xlabel("2_cluster")


# miara silhoueete - rysowanie
def chart_silhoueete(datasets):
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


# reszta miar
def chart_measures(datasets):
    # grupa wykresów z reszta miar
    fig4, ax4 = plt.subplots(1, 6, figsize=(30, 5))
    fig4.tight_layout()

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
        ax4[i].plot(linspacee, k_meansNN_rate)  #
        ax4[i].set_xlabel("n_cluster")
        ax4[i].legend(["adjusted rand", "homogeneity", "completeness", "v-measure beta=0.5", "v-measure beta=1",
                       "v-measure beta=2"])


# najlepsze wykresy silhoueete
def charts_sil_best(datasets):
    # wykresy najlepszy podzial dla silhoueete
    fig5, ax5 = plt.subplots(1, 6, figsize=(30, 5))
    fig5.tight_layout()
    for i in range(len(datasets)):

        x = datasets[i].iloc[0:, 0]
        y = datasets[i].iloc[0:, 1]

        kmeans = KMeans(n_clusters=sil_max[i], random_state=0).fit(datasets[i].iloc[0:, 0:2].to_numpy())

        ax5[i].scatter(x, y, c=kmeans.labels_)  #
        ax5[i].set_xlabel("%s_cluster" % (sil_max[i]))

        vor = Voronoi(datasets[i].iloc[0:, 0:2].to_numpy())
        voronoi_plot_2d(vor, ax=ax5[i], show_vertices=False, show_points=False, show_region=True,
                        show_points_label=False)
        for r in range(len(vor.point_region)):
            region = vor.regions[vor.point_region[r]]
            if not -1 in region:
                polygon = [vor.vertices[i] for i in region]
                ax5[i].fill(*zip(*polygon), color=plt.cm.Set1(kmeans.labels_[r], alpha=0.5))


# najgorsze wykresy silhoueete
def charts_sil_worst(datasets):
    # wykresy najgorszy podzial dla silhoueete
    fig6, ax6 = plt.subplots(1, 6, figsize=(30, 5))
    fig6.tight_layout()
    for i in range(len(datasets)):

        x = datasets[i].iloc[0:, 0]
        y = datasets[i].iloc[0:, 1]

        kmeans = KMeans(n_clusters=sil_min[i], random_state=0).fit(datasets[i].iloc[0:, 0:2].to_numpy())

        ax6[i].scatter(x, y, c=kmeans.labels_)  #
        ax6[i].set_xlabel("%s_cluster" % (sil_min[i]))

        vor = Voronoi(datasets[i].iloc[0:, 0:2].to_numpy())
        voronoi_plot_2d(vor, ax=ax6[i], show_vertices=False, show_points=False, show_region=True,
                        show_points_label=False)
        for r in range(len(vor.point_region)):
            region = vor.regions[vor.point_region[r]]
            if not -1 in region:
                polygon = [vor.vertices[i] for i in region]
                ax6[i].fill(*zip(*polygon), color=plt.cm.Set1(kmeans.labels_[r], alpha=0.5))


# najlepsze wykresy reszta miar
def charts_other_best(datasets):
    # wykresy najlepszy podzial dla reszty miar
    fig7, ax7 = plt.subplots(1, 6, figsize=(30, 5))
    fig7.tight_layout()
    for i in range(len(datasets)):

        x = datasets[i].iloc[0:, 0]
        y = datasets[i].iloc[0:, 1]

        kmeans = KMeans(n_clusters=met_max[i], random_state=0).fit(datasets[i].iloc[0:, 0:2].to_numpy())

        ax7[i].scatter(x, y, c=kmeans.labels_)  #
        ax7[i].set_xlabel("%s_cluster" % (met_max[i]))

        vor = Voronoi(datasets[i].iloc[0:, 0:2].to_numpy())
        voronoi_plot_2d(vor, ax=ax7[i], show_vertices=False, show_points=False, show_region=True,
                        show_points_label=False)
        for r in range(len(vor.point_region)):
            region = vor.regions[vor.point_region[r]]
            if not -1 in region:
                polygon = [vor.vertices[i] for i in region]
                ax7[i].fill(*zip(*polygon), color=plt.cm.Set1(kmeans.labels_[r], alpha=0.5))


# najgorsze wykresy reszta miar
def charts_other_worst(datasets):
    # wykresy najgorszy podzial dla reszty miar
    fig8, ax8 = plt.subplots(1, 6, figsize=(30, 5))
    fig8.tight_layout()
    for i in range(len(datasets)):

        x = datasets[i].iloc[0:, 0]
        y = datasets[i].iloc[0:, 1]

        kmeans = KMeans(n_clusters=met_min[i], random_state=0).fit(datasets[i].iloc[0:, 0:2].to_numpy())

        ax8[i].scatter(x, y, c=kmeans.labels_)  #
        ax8[i].set_xlabel("%s_cluster" % (met_min[i]))

        vor = Voronoi(datasets[i].iloc[0:, 0:2].to_numpy())
        voronoi_plot_2d(vor, ax=ax8[i], show_vertices=False, show_points=False, show_region=True,
                        show_points_label=False)
        for r in range(len(vor.point_region)):
            region = vor.regions[vor.point_region[r]]
            if not -1 in region:
                polygon = [vor.vertices[i] for i in region]
                ax8[i].fill(*zip(*polygon), color=plt.cm.Set1(kmeans.labels_[r], alpha=0.5))


# ścieżki do plików
paths = ["1_1.csv", "1_2.csv", "1_3.csv", "2_1.csv", "2_2.csv", "2_3.csv"]

sil_max = []  # makszyma silhoueete
sil_min = []  # minima silhoueete

met_max = [2, 2, 6, 2, 3, 3]  # maksima innych miar
met_min = [10, 10, 7, 10, 10, 4]  # minima innych miar

datasets_from_csv = load_from_csv(paths)

charts(datasets_from_csv)
chart_silhoueete(datasets_from_csv)
chart_measures(datasets_from_csv)
charts_sil_best(datasets_from_csv)
charts_sil_worst(datasets_from_csv)
charts_other_best(datasets_from_csv)
charts_other_worst(datasets_from_csv)

plt.show()
