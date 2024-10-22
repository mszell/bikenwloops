import os
import yaml
from tqdm.notebook import tqdm
import geopandas as gpd
import igraph as ig
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import momepy as momepy
import networkx as nx
from functools import reduce
import pickle
import lzma
import shapely
from shapely import LineString
from itertools import combinations, compress
from statistics import median

np.random.seed(42)

edge_classification_colors = {
    "too_short": "#000000",
    "ideal_range": "#00CB00",
    "above_ideal": "#FFB900",
    "too_long": "#E10000",
}

loop_classification_colors = {
    "too_short": "#3182bd",
    "ideal_range": "#bdd7e7",
    "too_long": "#fd8d3c",
}

with open(r"../config.yml") as file:
    parsed_yaml_file = yaml.load(file, Loader=yaml.FullLoader)

    STUDY_AREA = parsed_yaml_file["study_area"]

    PATH = {
        "data_in_network": "../data/input/" + STUDY_AREA + "/network/processed/",
        "data_in_pois": "../data/input/" + STUDY_AREA + "/point/",
        "data_out": "../data/processed/" + STUDY_AREA + "/",
        "plot": "../plots/" + STUDY_AREA + "/",
    }
    for folder in PATH.values():
        if not os.path.exists(folder):
            os.makedirs(folder)

    STUDY_AREA_COMBINED = parsed_yaml_file["study_area_combined"]
    if STUDY_AREA in STUDY_AREA_COMBINED:  # create a nested path
        PATH = {}
        PATH["plot"] = "../plots/" + STUDY_AREA + "/"
        PATH["data_out"] = "../data/processed/" + STUDY_AREA + "/"
        for subarea in STUDY_AREA_COMBINED[STUDY_AREA]:
            PATH[subarea] = {
                "data_in_network": "../data/input/" + subarea + "/network/processed/",
                "data_in_pois": "../data/input/" + subarea + "/point/",
                "data_out": "../data/processed/" + subarea + "/",
                "plot": "../plots/" + subarea + "/",
            }

    MAXSLOPE_LIMIT = parsed_yaml_file["maxslope_limit"]
    MPERUNIT = parsed_yaml_file["mperunit"]
    FACELOOP_LIMIT = [  # 90% of face loop lengths should conform to these length limits [m]
        parsed_yaml_file["faceloop_limit_lower"],
        parsed_yaml_file["faceloop_limit_upper"],
    ]
    LINK_LIMIT = [  # Optimal length between first and second value, maximal length the last value [m]
        parsed_yaml_file["link_limit_lower"],
        parsed_yaml_file["link_limit_upper"],
        parsed_yaml_file["link_limit_max"],
    ]
    SNAP_THRESHOLD = parsed_yaml_file[
        "snap_threshold"
    ]  # Threshold to snap POIs to network links [m]
    LOOP_NUMNODE_BOUND = parsed_yaml_file[
        "loop_numnode_bound"
    ]  # From 30 it starts getting slow
    LOOP_LENGTH_BOUND = parsed_yaml_file[
        "loop_length_bound"
    ]  # Physical length threshold in meters

    RESTRICTIONS = {
        "looplength_min": parsed_yaml_file["looplength_min"],
        "looplength_max": parsed_yaml_file["looplength_max"],
        "slope_max": MAXSLOPE_LIMIT,
        "waterlength_max": parsed_yaml_file["waterlength_max"],
    }
    PLOTLOGSCALE = parsed_yaml_file[
        "plotlogscale"
    ]  # Boolean flag for using log scale in certain plots

    MAXSLOPES_AVAILABLE = parsed_yaml_file[
        "maxslopes_available"
    ]  # Boolean flag for using available max_slope data, otherwise generating random data for testing

    POIS_AVAILABLE = parsed_yaml_file[
        "pois_available"
    ]  # Boolean flag for using available poi data, otherwise generating random data for testing

    BORNHOLM_DELTA = parsed_yaml_file["bornholm_delta"]
