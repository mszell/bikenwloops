import os
import yaml

with open(r"../config.yml") as file:
    parsed_yaml_file = yaml.load(file, Loader=yaml.FullLoader)

    study_area = parsed_yaml_file["study_area"]

    PATH = {
        "data_in_network": "../data/input/" + study_area + "/network/",
        "data_in_pois": "../data/input/" + study_area + "/point/",
        "data_out": "../data/processed/" + study_area + "/",
        "plot": "../plots/" + study_area + "/",
    }

    for folder in PATH.values():
        if not os.path.exists(folder):
            os.mkdir(folder)

    maxslope_limit = parsed_yaml_file["maxslope_limit"]
    faceloop_limit = [  # 90% of face loop lengths should conform to these length limits [m]
        parsed_yaml_file["faceloop_limit_lower"],
        parsed_yaml_file["faceloop_limit_upper"],
    ]
    link_limit = [  # Optimal length between first and second value, maximal length the last value [m]
        parsed_yaml_file["link_limit_lower"],
        parsed_yaml_file["link_limit_upper"],
        parsed_yaml_file["link_limit_max"],
    ]
    snap_threshold = parsed_yaml_file[
        "snap_threshold"
    ]  # Threshold to snap POIs to network links [m]
    loop_numnode_bound = parsed_yaml_file[
        "loop_numnode_bound"
    ]  # From 30 it starts getting slow

    restrictions = {
        "looplength_min": parsed_yaml_file["looplength_min"],
        "looplength_max": parsed_yaml_file["looplength_max"],
        "slope_max": maxslope_limit,
        "waterlength_max": parsed_yaml_file["waterlength_max"],
    }