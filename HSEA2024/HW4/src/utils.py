import csv
import math
import time

import numpy as np
from scipy.spatial import distance


def rank_macros(placedb, rank_key="area"):
    """
    Sort the place ranking of the macros.

    Args:
        placedb: An instance of PlaceDB.
        rank_key: The key to rank the macros ('area' or 'area_sum').
    Returns:
        A sorted list of node IDs.
    """
    node_id_ls = list(placedb.node_info.keys()).copy()
    for node_id in node_id_ls:
        placedb.node_info[node_id]["area"] = placedb.node_info[node_id]["x"] * placedb.node_info[node_id]["y"]

    net_id_ls = list(placedb.net_info.keys()).copy()
    for net_id in net_id_ls:
        sum = 0
        for node_id in placedb.net_info[net_id]["nodes"].keys():
            sum += placedb.node_info[node_id]["area"]
        placedb.net_info[net_id]["area"] = sum
    for node_id in node_id_ls:
        placedb.node_info[node_id]["area_sum"] = 0
        for net_id in net_id_ls:
            if node_id in placedb.net_info[net_id]["nodes"].keys():
                placedb.node_info[node_id]["area_sum"] += placedb.net_info[net_id]["area"]
    node_id_ls.sort(key=lambda x: placedb.node_info[x][rank_key], reverse=True)
    return node_id_ls


def write_final_placement(best_placed_macro, dir):
    """
    Save locations of all the macros
    """
    csv_file2 = open(dir, "a+")
    csv_writer2 = csv.writer(csv_file2)
    csv_writer2.writerow([time.time()])
    for node_id in list(best_placed_macro.keys()):
        csv_writer2.writerow(
            [
                node_id,
                best_placed_macro[node_id]["bottom_left_x"],
                best_placed_macro[node_id]["bottom_left_y"],
            ]
        )
    csv_writer2.writerow([])
    csv_file2.close()


def evaluate(node_id_ls, placedb, grid_num, grid_size, place_record):
    placed_macros = {}
    hpwl_info_for_each_net = {}
    hpwl = 0

    for node_id in node_id_ls:
        x = placedb.node_info[node_id]["x"]
        y = placedb.node_info[node_id]["y"]
        scaled_x = math.ceil(x / grid_size)
        scaled_y = math.ceil(y / grid_size)
        placedb.node_info[node_id]["scaled_x"] = scaled_x
        placedb.node_info[node_id]["scaled_y"] = scaled_y
        position_mask = np.ones((grid_num, grid_num)) * 1e12
        position_mask[: grid_num - scaled_x, : grid_num - scaled_y] = 1
        wire_mask = np.ones((grid_num, grid_num)) * 0.1

        for key1 in placed_macros.keys():
            bottom_left_x = max(0, placed_macros[key1]["loc_x"] - scaled_x + 1)
            bottom_left_y = max(0, placed_macros[key1]["loc_y"] - scaled_y + 1)
            top_right_x = min(grid_num - 1, placed_macros[key1]["loc_x"] + placed_macros[key1]["scaled_x"])
            top_right_y = min(grid_num - 1, placed_macros[key1]["loc_y"] + placed_macros[key1]["scaled_y"])

            position_mask[bottom_left_x:top_right_x, bottom_left_y:top_right_y] = 1e12

        loc_x_ls = np.where(position_mask == 1)[0]
        loc_y_ls = np.where(position_mask == 1)[1]
        if len(loc_x_ls) == 0:
            print("no_legal_place")
            # No Legal position for placement. Try to improve your optimizer.
            return [], 1e12
        placed_macros[node_id] = {}
        net_ls = {}

        for net_id in placedb.net_info.keys():
            if node_id in placedb.net_info[net_id]["nodes"].keys():
                net_ls[net_id] = {}
                net_ls[net_id] = placedb.net_info[net_id]

        if len(loc_x_ls) == 0:
            print("No legal location for place")
            return [], 1e12

        for net_id in net_ls.keys():
            if net_id in hpwl_info_for_each_net.keys():
                x_offset = net_ls[net_id]["nodes"][node_id]["x_offset"] + 0.5 * x
                y_offset = net_ls[net_id]["nodes"][node_id]["y_offset"] + 0.5 * y
                for col in range(grid_num):
                    x_co = col * grid_size + x_offset
                    y_co = col * grid_size + y_offset

                    if x_co < hpwl_info_for_each_net[net_id]["x_min"]:
                        wire_mask[col, :] += hpwl_info_for_each_net[net_id]["x_min"] - x_co
                    elif x_co > hpwl_info_for_each_net[net_id]["x_max"]:
                        wire_mask[col, :] += x_co - hpwl_info_for_each_net[net_id]["x_max"]
                    if y_co < hpwl_info_for_each_net[net_id]["y_min"]:
                        wire_mask[:, col] += hpwl_info_for_each_net[net_id]["y_min"] - y_co
                    elif y_co > hpwl_info_for_each_net[net_id]["y_max"]:
                        wire_mask[:, col] += y_co - hpwl_info_for_each_net[net_id]["y_max"]
        wire_mask = np.multiply(wire_mask, position_mask)
        min_ele = np.min(wire_mask)
        # print(np.where(wire_mask == min_ele)[0][0],np.where(wire_mask == min_ele)[1][0])

        chosen_loc_x = list(np.where(wire_mask == min_ele)[0])
        chosen_loc_y = list(np.where(wire_mask == min_ele)[1])
        chosen_coor = list(zip(chosen_loc_x, chosen_loc_y))

        tup_order = []
        for tup in chosen_coor:
            tup_order.append(
                distance.euclidean(tup, (place_record[node_id]["loc_x"], place_record[node_id]["loc_y"]))
            )
        chosen_coor = list(zip(chosen_coor, tup_order))

        chosen_coor.sort(key=lambda x: x[1])

        chosen_loc_x = chosen_coor[0][0][0]
        chosen_loc_y = chosen_coor[0][0][1]
        best_hpwl = min_ele

        center_loc_x = grid_size * chosen_loc_x + 0.5 * x
        center_loc_y = grid_size * chosen_loc_y + 0.5 * y
        for net_id in net_ls.keys():
            x_offset = net_ls[net_id]["nodes"][node_id]["x_offset"]
            y_offset = net_ls[net_id]["nodes"][node_id]["y_offset"]
            if net_id not in hpwl_info_for_each_net.keys():
                hpwl_info_for_each_net[net_id] = {}
                hpwl_info_for_each_net[net_id] = {
                    "x_max": center_loc_x + x_offset,
                    "x_min": center_loc_x + x_offset,
                    "y_max": center_loc_y + y_offset,
                    "y_min": center_loc_y + y_offset,
                }
            else:
                if hpwl_info_for_each_net[net_id]["x_max"] < center_loc_x + x_offset:
                    hpwl_info_for_each_net[net_id]["x_max"] = center_loc_x + x_offset
                elif hpwl_info_for_each_net[net_id]["x_min"] > center_loc_x + x_offset:
                    hpwl_info_for_each_net[net_id]["x_min"] = center_loc_x + x_offset
                if hpwl_info_for_each_net[net_id]["y_max"] < center_loc_y + y_offset:
                    hpwl_info_for_each_net[net_id]["y_max"] = center_loc_y + y_offset
                elif hpwl_info_for_each_net[net_id]["y_min"] > center_loc_y + y_offset:
                    hpwl_info_for_each_net[net_id]["y_min"] = center_loc_y + y_offset

        hpwl += best_hpwl
        placed_macros[node_id] = {
            "scaled_x": scaled_x,
            "scaled_y": scaled_y,
            "loc_x": chosen_loc_x,
            "loc_y": chosen_loc_y,
            "x": x,
            "y": y,
            "center_loc_x": center_loc_x,
            "center_loc_y": center_loc_y,
            "bottom_left_x": chosen_loc_x * grid_size + 452,
            "bottom_left_y": chosen_loc_y * grid_size + 452,
        }
    return placed_macros, hpwl
