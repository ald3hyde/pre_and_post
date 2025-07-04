#!/usr/bin/env python
import itertools as it
import subprocess as sp
import glob
import os
import numpy as np
import pandas as pd
# Written by Ryuki Kayano, Chiba university.


class LammpsData():
    def __init__(self, lammpstrjfile):
        self.lammpstrj = lammpstrjfile
        self.base, _ = os.path.splitext(os.path.basename(self.lammpstrj))
        self.atoms = 0
        self.fp = open(lammpstrjfile)  # FilePointer
        self.idx = {}
        self.header = ""
        self.typelist = {}
        self.atom_id = {}
        self.step = 0
        self._read_header()

    def _read_header(self):
        for i in range(4):
            line = self.fp.readline()
        self.atoms = int(line)
        for i in range(5):
            line = self.fp.readline()
        items = np.array(line.split()[2:])
        self.data = np.empty((0, len(items)), dtype=object)
        for i, key in enumerate(items):
            self.idx[key] = i
        self.fp.seek(0)

    def _set_lattice(self, L):
        lattice = np.zeros((3, 3), dtype=float)
        if L.shape[1] == 2:  # orthogonal
            lattice[0, 0] = L[0, 1] - L[0, 0]
            lattice[1, 1] = L[1, 1] - L[1, 0]
            lattice[2, 2] = L[2, 1] - L[2, 0]
        if L.shape[1] == 3:  # triclinic
            xy, xz, yz = L[0, 2], L[1, 2], L[2, 2]
            xlo = L[0, 0] - np.min([0, xy, xz, xy+xz])
            xhi = L[0, 1] - np.max([0, xy, xz, xy+xz])
            ylo = L[1, 0] - np.min([0, yz])
            yhi = L[1, 1] - np.max([0, yz])
            lattice = np.array([[xhi - xlo, 0, 0],
                                [xy, yhi - ylo, 0],
                                [xz, yz, L[2, 1] - L[2, 0]]])
        self.lattice = lattice

    def write_eachsteptrjfile(self):
        buf = ""
        buf += self.header
        for row in self.data:
            rows = " ".join(map(str, row))
            buf += rows + "\n"
        with open(f"{self.step:05d}_tmp.lammpstrj", "w") as w:
            w.write(buf)

    def cat_alllammpstrjfile(self):
        rm = ["rm", "-rf"]
        body = ""
        tmp_trjfiles = sorted(glob.glob("*tmp.lammpstrj"))
        rm += tmp_trjfiles
        for trjfile in tmp_trjfiles:
            with open(trjfile) as o:
                body += o.read()
        with open(f"updated_{self.base}.lammpstrj", "w") as w:
            w.write(body)
        sp.run(rm)

    def get_data(self):
        self.header = ""
        body = ""
        for i in range(self.atoms + 9):
            line = self.fp.readline()
            if i < 9:
                self.header += line
            if line == "":
                self.fp.close()
                return False
            body += line
        lines = body.split("\n")
        L = np.array(" ".join(lines[5:8]).split(), dtype=float).reshape(3, -1)
        self._set_lattice(L)
        self.data = np.array(" ".join(lines[9:]).split(), dtype=object)
        self.data = self.data.reshape(self.atoms, -1)
        elem_u = np.unique(self.data[:, self.idx["element"]])
        for e in elem_u:
            self.typelist[e] = np.where(
                self.data[:, self.idx["element"]] == e)[0]
            self.atom_id[e] = self.data[self.data[:, self.idx["element"]] == e,
                                        self.idx["type"]][0]
        self.step += 1
        self.elem_u = elem_u
        return True


class GlassAnalytics():
    def __init__(self, lmp, PARAMS, network):
        self.lmp = lmp
        self.coord_idx = {}
        self.coord = {}
        self.horizon = "-" * 74
        self.network = network
        pairs = it.combinations_with_replacement(self.elems, 2)
        linkage = [f"{p[0]}-O-{p[1]}" for p in pairs]
        for e in self.elems:
            linkage.append(f"{e}-O-c")
        self.type_list = linkage
        self.pairlist = None
        self.oxygen_dict = {key: [] for key in self.type_list}
        self.silicon_dict = None
        self.cutoffs = {}
        self.oxygen_linkages_stepwise = []

        for s in self.network:
            self.cutoffs[s] = PARAMS[s]

    def _horizon(self):
        print(self.horizon)

    def _calc_distance(self, s1, symbols):
        pairlist = {}
        x, z = self.lmp.idx["x"], self.lmp.idx["z"]+1
        type_s1 = self.lmp.atom_id[s1]
        s1_idx = np.where(self.lmp.data[:, self.lmp.idx["type"]] == type_s1)[0]
        typelist = np.zeros(s1_idx.shape[0], dtype=np.int32)
        s1_xyz = self.lmp.data[s1_idx, x:z].astype(np.float64)
        pairlist[s1] = {i: [] for i in s1_idx}
        for s2 in symbols:
            if s2 == "BO":
                o = int(self.lmp.atom_id["O"])
                ob_idx = np.where(
                    self.lmp.data[:, self.lmp.idx["type"]] == f"{o*10+2}")[0]
                otb_idx = np.where(
                    self.lmp.data[:, self.lmp.idx["type"]] == f"{o*10+3}")[0]
                s2_idx = np.append(ob_idx, otb_idx)
            else:
                s2_idx = np.where(
                    self.lmp.data[:, self.lmp.idx["element"]] == s2)[0]
            s2_xyz = self.lmp.data[s2_idx, x:z].astype(np.float64)
            dr_ = s2_xyz[:, None, :] - s1_xyz  # broad casting
            # PBC処理
            dr_ = (dr_ @ np.linalg.inv(self.lmp.lattice)).astype(float)
            dr_ = (dr_ - np.rint(dr_)) @ self.lmp.lattice
            dr_ = np.sqrt(np.sum(dr_**2, axis=2).astype(float))
            if s2 == "BO" or s2 == "O":
                typelist += np.sum(dr_ < self.cutoffs[s1], axis=0)
            else:
                bond_idxs = np.where(dr_ < self.cutoffs[s2])
                for s2_index, s1_index in zip(*bond_idxs):
                    pairlist[s1][s1_idx[s1_index]].append(s2_idx[s2_index])
                typelist += np.sum(dr_ < self.cutoffs[s2], axis=0)

        if s2 == "BO" or s2 == "O":
            return s1_idx, typelist
        else:
            return s1_idx, typelist, pairlist

    def _update_atomtype(self, s1, symbols, limit):
        coord_dict = []
        if s1 == "O":
            s1_idx, s1_list, self.pairlist = self._calc_distance(s1, symbols)
        else:
            s1_idx, s1_list = self._calc_distance(s1, symbols)
        s1_type = np.empty(s1_idx.shape[0], dtype=object)
        s1_type[:] = self.lmp.atom_id[s1]
        for i in range(limit):
            new_type = f"{int(self.lmp.atom_id[s1]) * 10 + i}"
            s1_type[s1_list == i] = new_type
            coord_dict.append(len(s1_type[s1_type == new_type]))
        self.lmp.data[s1_idx, self.lmp.idx["type"]] = s1_type
        coord = np.array(coord_dict)
        if s1 in self.coord.keys():
            self.coord[s1] = np.vstack((self.coord[s1], coord))
        else:
            self.coord[s1] = coord

    def update_atomstype(self):
        # 配位数計算
        if "B" in self.lmp.elem_u:
            self._update_atomtype("B", ["O"], 10)
        if "Al" in self.lmp.elem_u:
            self._update_atomtype("Al", ["O"], 10)
        if "Zr" in self.lmp.elem_u:
            self._update_atomtype("Zr", ["O"], 10)
        if "Si" in self.lmp.elem_u:
            self._update_atomtype("Si", ["O"], 10)
        # BO, NBO判別
        self._update_atomtype("O", self.network, 7)
        # （SiのQnなど追加で計算するならここに実装）
        # if "Al" in self.lmp.elem_u:
        #     self._search_second_neigh(6)

    def search_oxygen_linkages(self):
        type_dict = {key: 0 for key in self.type_list}
        for o_idx, pair_indices in self.pairlist["O"].items():
            elem_list = [self.lmp.data[idx, self.lmp.idx["element"]]
                         for idx in pair_indices]
            flag = [elem in self.lmp.elem_u for elem in elem_list]
            if sum(flag) != len(elem_list):
                continue
            if len(elem_list) == 1:
                pair = f"{elem_list[0]}-O-c"
                type_dict[pair] += 1
            elif len(elem_list) == 2:
                pair = f"{elem_list[0]}-O-{elem_list[1]}"
                if pair not in self.type_list:
                    pair = f"{elem_list[1]}-O-{elem_list[0]}"
                type_dict[pair] += 1
            elif len(elem_list) >= 3:
                for elems in it.combinations(elem_list, 2):
                    pair = f"{elems[0]}-O-{elems[1]}"
                    if pair not in self.type_list:
                        pair = f"{elems[1]}-O-{elems[0]}"
                    type_dict[pair] += 1
        for k, v in type_dict.items():
            self.oxygen_dict[k].append(v)
        self.oxygen_linkages_stepwise.append(type_dict.copy())

    def coordination_number(self):
        self._horizon()
        for key, val in self.coord.copy().items():
            new_val = np.mean(val, axis=0) / self.lmp.step
            new_val = new_val / sum(new_val) * 100
            self.coord[key] = new_val
            for i, val_ in enumerate(new_val):
                print(f"{key}{i}:{val_:2.2f}%")
            self._horizon()
        if self.silicon_dict is not None:
            copy = {k: v / self.lmp.step for k, v in self.silicon_dict.items()}
            s = sum(list(copy.values()))
            for key, val in copy.items():
                new_val = val / s * 100
                self.silicon_dict[key] = new_val
                print(f"{key}:{float(new_val):2.2f}%")
            self._horizon()

        copy = {k: np.mean(v) for k, v in self.oxygen_dict.items()}
        s = sum(list(copy.values()))
        for k, v in copy.items():
            new_val = v / s * 100 if s > 1.0e-12 else 0.0
            self.oxygen_dict[k] = new_val
            print(f"{k}:{float(new_val):2.2f}%")
        self._horizon()


class Outputdata():
    def __init__(self, name=None):
        self.df = pd.DataFrame()
        self.name = name

    def _append_data(self, dic_1, dic_2):
        for k, v in dic_2.items():
            dic_1[k] = v

    def stack_data(self, gl):
        dic = {}
        dic["name"] = gl.lmp.base
        for k, v in gl.coord.items():
            for i, val in enumerate(v):
                key = f"{k}{i}"
                dic[key] = val
        if gl.silicon_dict is not None:
            self._append_data(dic, gl.silicon_dict)
        if gl.oxygen_dict is not None:
            self._append_data(dic, gl.oxygen_dict)
        data = pd.DataFrame(dic, index=[0])
        self.df = pd.concat([self.df, data], ignore_index=True)

    def write_xlsxfile(self):
        self.df = self.df.sort_index(axis=0)
        base = self.name if self.name is not None else "dump2glassanalysis"
        with pd.ExcelWriter(f"{base}.xlsx", engine='openpyxl') as w:
            self.df.to_excel(w, index=False)

    def write_stepwise_oxygen_linkages(self, gl):
        df_oxy = pd.DataFrame(gl.oxygen_linkages_stepwise)
        df_oxy.insert(0, 'step', range(1, len(df_oxy) + 1))
        base = self.name if self.name is not None else "dump2glassanalysis"
        with pd.ExcelWriter(f"{base}_oxygen_linkages_stepwise.xlsx", engine='openpyxl') as w:
            df_oxy.to_excel(w, index=False)


if __name__ == "__main__":
    import argparse
    par = argparse.ArgumentParser(
        description="This is a program for the analysis of oxide glasses")
    par.add_argument("lammpstrj", nargs="+")
    par.add_argument("-o", "--output", required=False, type=str)
    par.add_argument("-v", "--verbose", action="store_true")
    args = par.parse_args()

    CUTOFF = {}
    CUTOFF["Si"] = 2.3
    CUTOFF["B"] = 2.3
    CUTOFF["Al"] = 2.40
    # CUTOFF["Zr"] = 2.9
    NETWORK = ["Si", "B", "Al"]

    trjfiles = sorted(args.lammpstrj)
    data = Outputdata(args.output)

    for trjfile in trjfiles:
        lmp = LammpsData(trjfile)
        gl = GlassAnalytics(lmp, CUTOFF, NETWORK)
        while lmp.get_data():
            gl.update_atomstype()
            gl.search_oxygen_linkages()
            if args.verbose:
                print(f"{trjfile}: {lmp.step} step was completed.")
            lmp.write_eachsteptrjfile()
        gl.coordination_number()
        data.stack_data(gl)
        lmp.cat_alllammpstrjfile()
    data.write_xlsxfile()
    data.write_stepwise_oxygen_linkages(gl)
