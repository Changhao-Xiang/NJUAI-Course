import argparse
import os
from itertools import combinations
from operator import itemgetter


def read_node_file(fopen, benchmark):
    node_info = {}
    node_info_raw_id_name = {}
    port_info = {}
    node_cnt = 0
    for line in fopen.readlines():
        if not line.startswith("\t") and not line.startswith(" "):
            continue
        line = line.strip().split()
        if line[-1] != "terminal":
            continue
        node_name = line[0]
        x = int(line[1])
        y = int(line[2])
        node_info[node_name] = {"id": node_cnt, "x": x, "y": y}
        node_info_raw_id_name[node_cnt] = node_name
        node_cnt += 1
    return node_info, node_info_raw_id_name, port_info


def read_net_file(fopen, node_info):
    net_info = {}
    net_name = None
    net_cnt = 0
    for line in fopen.readlines():
        if not line.startswith("\t") and not line.startswith("  ") and not line.startswith("NetDegree"):
            continue
        line = line.strip().split()
        if line[0] == "NetDegree":
            net_name = line[-1]
        else:
            node_name = line[0]
            if node_name in node_info:
                if not net_name in net_info:
                    net_info[net_name] = {}
                    net_info[net_name]["nodes"] = {}
                    net_info[net_name]["ports"] = {}
                if not node_name.startswith("p") and not node_name in net_info[net_name]["nodes"]:
                    x_offset = float(line[-2])
                    y_offset = float(line[-1])
                    net_info[net_name]["nodes"][node_name] = {}
                    net_info[net_name]["nodes"][node_name] = {"x_offset": x_offset, "y_offset": y_offset}
                elif node_name.startswith("p") and node_name in net_info[net_name]["ports"]:
                    x_offset = float(line[-2])
                    y_offset = float(line[-1])
                    net_info[net_name]["ports"][node_name] = {}
                    net_info[net_name]["ports"][node_name] = {"x_offset": x_offset, "y_offset": y_offset}
    for net_name in list(net_info.keys()):
        if len(net_info[net_name]["nodes"]) <= 1:
            net_info.pop(net_name)
    for net_name in net_info:
        net_info[net_name]["id"] = net_cnt
        net_cnt += 1
    return net_info


def get_comp_hpwl_dict(node_info, net_info):
    comp_hpwl_dict = {}
    for net_name in net_info:
        max_idx = 0
        for node_name in net_info[net_name]["nodes"]:
            max_idx = max(max_idx, node_info[node_name]["id"])
        if not max_idx in comp_hpwl_dict:
            comp_hpwl_dict[max_idx] = []
        comp_hpwl_dict[max_idx].append(net_name)
    return comp_hpwl_dict


def get_node_to_net_dict(node_info, net_info):
    node_to_net_dict = {}
    for node_name in node_info:
        node_to_net_dict[node_name] = set()
    for net_name in net_info:
        for node_name in net_info[net_name]["nodes"]:
            node_to_net_dict[node_name].add(net_name)
    return node_to_net_dict


def get_port_to_net_dict(port_info, net_info):
    port_to_net_dict = {}
    for port_name in port_info:
        port_to_net_dict[port_name] = set()
    for net_name in net_info:
        for port_name in net_info[net_name]["ports"]:
            port_to_net_dict[port_name].add(net_name)
    return port_to_net_dict


def read_pl_file(fopen, node_info):
    max_height = 0
    max_width = 0
    min_height = 999999
    min_width = 999999
    for line in fopen.readlines():
        if not line.startswith("o"):
            continue
        line = line.strip().split()
        node_name = line[0]
        if not node_name in node_info:
            continue
        place_x = int(line[1])
        place_y = int(line[2])
        max_height = max(max_height, node_info[node_name]["x"] + place_x)
        max_width = max(max_width, node_info[node_name]["y"] + place_y)
        min_height = min(min_height, place_x)
        min_width = min(min_width, place_y)
        node_info[node_name]["raw_x"] = place_x
        node_info[node_name]["raw_y"] = place_y
    return max(max_height, max_width), max(max_height, max_width), min_height, min_width


def read_scl_file(fopen, benchmark):
    assert "ibm" in benchmark
    for line in fopen.readlines():
        if not "Numsites" in line:
            continue
        line = line.strip().split()
        max_height = int(line[-1])
        break
    return max_height, max_height


def get_pin_cnt(net_info):
    pin_cnt = 0
    for net_name in net_info:
        pin_cnt += len(net_info[net_name]["nodes"])
    return pin_cnt


def get_total_area(node_info):
    area = 0
    for node_name in node_info:
        area += node_info[node_name]["x"] * node_info[node_name]["y"]
    return area


class PlaceDB:
    def __init__(self, benchmark="adaptec1"):
        self.benchmark = benchmark
        assert os.path.exists(os.path.join("../benchmark", benchmark))
        node_file = open(os.path.join("../benchmark", benchmark, benchmark + ".nodes"), "r")
        self.node_info, self.node_info_raw_id_name, self.port_info = read_node_file(node_file, benchmark)
        pl_file = open(os.path.join("../benchmark", benchmark, benchmark + ".pl"), "r")
        self.node_cnt = len(self.node_info)
        node_file.close()
        net_file = open(os.path.join("../benchmark", benchmark, benchmark + ".nets"), "r")
        self.net_info = read_net_file(net_file, self.node_info)
        self.net_cnt = len(self.net_info)
        net_file.close()
        pl_file = open(os.path.join("../benchmark", benchmark, benchmark + ".pl"), "r")
        self.max_height, self.max_width, self.min_height, self.min_width = read_pl_file(
            pl_file, self.node_info
        )
        pl_file.close()
        if not "ibm" in benchmark:
            self.port_to_net_dict = {}
        else:
            self.port_to_net_dict = get_port_to_net_dict(self.port_info, self.net_info)
            scl_file = open(os.path.join("../benchmark", benchmark, benchmark + ".scl"), "r")
            self.max_height, self.max_width = read_scl_file(scl_file, benchmark)

        self.node_to_net_dict = get_node_to_net_dict(self.node_info, self.net_info)

    def debug_str(self):
        print("node_cnt = {}".format(len(self.node_info)))
        print("net_cnt = {}".format(len(self.net_info)))
        print("max_height = {}".format(self.max_height))
        print("max_width = {}".format(self.max_width))
        print("pin_cnt = {}".format(get_pin_cnt(self.net_info)))
        print("port_cnt = {}".format(len(self.port_info)))
        print("area_ratio = {}".format(get_total_area(self.node_info) / (self.max_height * self.max_height)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="argparse testing")
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()
    dataset = args.dataset
    placedb = PlaceDB(dataset)
    placedb.debug_str()
    print(placedb.node_info)
