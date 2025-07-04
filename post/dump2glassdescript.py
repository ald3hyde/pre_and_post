#!/usr/bin/env python
import numpy as np
import pandas as pd
import subprocess as sp
import glob
import itertools as it
import os


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
        # new_file = f"{self.base}_new.lammpstrj"
        # self.new_fp = open(new_file, "w")
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
            lattice = np.array([[xhi-xlo, 0,       0],
                                [xy,      yhi-ylo, 0],
                                [xz,      yz,      L[2, 1]-L[2, 0]]])
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
        with open(f"update_{self.base}.lammpstrj", "w") as w:
            w.write(body)
        sp.run(rm)

    def get_data(self):
        header_lines = []
        for _ in range(9):
            line = self.fp.readline()
            if not line:
                self.fp.close()
                return False
            header_lines.append(line)
        self.header = "".join(header_lines)
        L = np.fromstring(
            " ".join(header_lines[5:8]), sep=' ', dtype=float).reshape(3, -1)
        self._set_lattice(L)
        data_lines = []
        for _ in range(self.atoms):
            line = self.fp.readline()
            if not line:
                self.fp.close()
                return False
            data_lines.append(line.split())
        self.data = np.array(data_lines, dtype=object)
        elem_idx = self.idx["element"]
        type_idx = self.idx["type"]
        elem_col = self.data[:, elem_idx]
        type_col = self.data[:, type_idx]
        self.elem_u = elem_u = np.unique(elem_col)
        for e in elem_u:
            mask = (elem_col == e)
            self.typelist[e] = type_col[mask][0]
        self.step += 1
        return True

    def _get_data(self):
        self.header = ""
        body = ""
        for i in range(self.atoms + 9):
            line = self.fp.readline()
            if i < 9:
                self.header += line
            if line == "":
                # self.new_fp.close()
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
            e_idx = np.where(self.data[:, self.idx["element"]] == e)[0]
            self.typelist[e] = np.unique(self.data[e_idx, self.idx["type"]])[0]
        self.step += 1
        self.elem_u = elem_u
        return True


class LammpsAnalysis(LammpsData):
    def __init__(self, lammpstrj, cutoff, former):
        # 親クラス: LammpsData
        super(LammpsAnalysis, self).__init__(lammpstrj)
        self.cutoff = cutoff
        self.network_former = former

    def update_coordination_number(self, update_type):
        self.bond_dict = {idx: [] for idx in range(self.data.shape[0])}
        x, z = self.idx["x"], self.idx["z"] + 1
        lattice_inv = np.linalg.inv(self.lattice)
        # Initialize coordination number lists for each element
        cn_list = {}
        for elem in self.elem_u:
            elem_indices = np.where(
                self.data[:, self.idx["element"]] == elem)[0]
            cn_list[elem] = np.zeros(elem_indices.shape[0], dtype=int)
        for pair, cutoff in self.cutoff.items():
            elem_i, elem_j = pair.split("-")
            # Skip if either element is not present in the current data
            if elem_i not in self.elem_u or elem_j not in self.elem_u:
                continue
            # Get indices of elem_i and elem_j
            index_elem_i = np.where(
                self.data[:, self.idx["element"]] == elem_i)[0]
            index_elem_j = np.where(
                self.data[:, self.idx["element"]] == elem_j)[0]
            # Get coordinates
            xyz_elem_i = self.data[index_elem_i, x:z].astype(float)
            xyz_elem_j = self.data[index_elem_j, x:z].astype(float)

            # Compute distances considering periodic boundary conditions
            delta_ij = xyz_elem_i[:, np.newaxis, :] - xyz_elem_j
            delta_ij = (delta_ij @ lattice_inv)
            delta_ij = delta_ij - np.rint(delta_ij)
            delta_ij = delta_ij @ self.lattice
            dist_ij = np.sqrt(np.sum(delta_ij**2, axis=2))

            # Determine which pairs are within the cutoff
            norm_flag = dist_ij <= cutoff

            # Find indices where norm_flag is True
            idx_i, idx_j = np.where(norm_flag)
            # Increment CN and update bond_dict for all bonded pairs
            for i_i, i_j in zip(idx_i, idx_j):
                atom_i = index_elem_i[i_i]
                atom_j = index_elem_j[i_j]
                # Increment CN for both atoms
                cn_list[elem_i][i_i] += 1
                if self.data[atom_i, self.idx["element"]] in self.network_former:
                    cn_list[elem_j][i_j] += 1
                    self.bond_dict[atom_j].append(atom_i)
                self.bond_dict[atom_i].append(atom_j)
        if "B" in self.elem_u:
            elem_indices = np.where(
                self.data[:, self.idx["element"]] == "B")[0]
            b2_mask = cn_list["B"] == 2
            b3_mask = cn_list["B"] == 3
            b4_mask = cn_list["B"] == 4
            b5_mask = cn_list["B"] == 5
            self.data[elem_indices[b2_mask], self.idx["element"]] = "B3"
            self.data[elem_indices[b3_mask], self.idx["element"]] = "B3"
            self.data[elem_indices[b4_mask], self.idx["element"]] = "B4"
            self.data[elem_indices[b5_mask], self.idx["element"]] = "B4"
            self.elem_u = np.unique(self.data[:, self.idx["element"]])
        self.bridge_oxygen_flag = cn_list["O"] > 1
        return cn_list

    def calculate_qn(self, elems=["Si"], update_type=False):
        x, z = self.idx["x"], self.idx["z"] + 1
        # Grab bridging oxygen indices
        elems = [elem for elem in self.elem_u if elem != "O"]
        oxygen_index = np.where(self.data[:, self.idx["element"]] == "O")[0]
        bridge_oxygen = self.data[oxygen_index, x:z].astype(
            float)[self.bridge_oxygen_flag]
        inv_lattice = np.linalg.inv(self.lattice)
        qn_list = {}
        for e in elems:
            if e not in self.elem_u:
                continue
            if "B3" == e or "B4" == e:
                continue
            cutoff = self.cutoff[f"{e}-O"]
            e_index = np.where(self.data[:, self.idx["element"]] == e)[0]
            e_xyz = self.data[e_index, x:z].astype(float)
            delta = e_xyz[:, np.newaxis, :] - bridge_oxygen
            delta = (delta @ inv_lattice)
            delta = delta - np.rint(delta)
            delta = delta @ self.lattice
            delta = np.sum(delta**2, axis=2)**0.5
            qn_list[e] = np.sum(delta < cutoff, axis=1)
            if update_type:
                old_type = self.typelist[e]
                self.data[e_index, self.idx["type"]] = [
                    f"{old_type}{int(val)}" for val in qn_list[e]
                ]
        if "B3" in elems or "B4" in elems:
            b3_flag = np.where(self.data[:, self.idx["element"]] == "B3")[0]
            b4_flag = np.where(self.data[:, self.idx["element"]] == "B4")[0]
            cutoff = self.cutoff["B-O"]
            b_flag = np.append(b3_flag, b4_flag)
            e_xyz = self.data[b_flag, x:z].astype(float)
            delta = e_xyz[:, np.newaxis, :] - bridge_oxygen
            delta = (delta @ inv_lattice)
            delta = delta - np.rint(delta)
            delta = delta @ self.lattice
            delta = np.sum(delta**2, axis=2)**0.5
            qn_list["B"] = np.sum(delta < cutoff, axis=1)
        return qn_list

    def calc_linkages(self, surface):
        oxygen_indices = np.where(self.data[:, self.idx["element"]] == "O")[0]
        oxygen_indices = self.data[oxygen_indices,
                                   self.idx["id"]].astype(int) - 1
        elem_data = self.old_data[:, self.idx["element"]
                                  ] if surface else self.data[:, self.idx["element"]]
        counter = {}
        network_former = self.network_former + ["B3", "B4"]
        cnt = 0
        for oxygen_index in oxygen_indices:
            pair_index = self.bond_dict[oxygen_index]
            pair_elem = elem_data[pair_index]
            for p_index, p_pair in zip(it.combinations(pair_index, 2),
                                       it.combinations(pair_elem, 2)):
                p_pair = sorted(p_pair)
                print(f"Processing pair: {p_pair}")
                if "B" in p_pair[0] and "B" in p_pair[1]:
                    linkage = f"{p_pair[0]}-O-{p_pair[1]}"
                else:
                    p_pair = ["B" if "B" in p else p for p in p_pair]
                    linkage = f"{p_pair[0]}-O-{p_pair[1]}"
                counter[linkage] = counter.get(linkage, 0) + 1
        return counter

    def detect_surface_group(self, cutoff):
        x, z = self.idx["x"], self.idx["z"] + 1
        xyz_data = self.data.copy()
        xyz_data[:, x:z] = xyz_data[:, x:z].astype(float)
        z_values = xyz_data[:, z - 1]
        sorted_indices = np.argsort(z_values)
        xyz_data = xyz_data[sorted_indices]
        z_coord = xyz_data[:, z-1].astype(float)

        diff = np.diff(z_coord)
        surface_idx = np.argmax(diff)
        surface_z = sorted([z_coord[surface_idx], z_coord[surface_idx + 1]])
        gap = surface_z[1] - surface_z[0]

        # lower mask
        surface_mask_lower = (
            (surface_z[0] - self.data[:, z-1].astype(float) <= cutoff) &
            (np.abs(self.data[:, z-1].astype(float) - surface_z[0]) < gap)
        )
        # upper mask
        surface_mask_upper = (
            (self.data[:, z-1].astype(float) - surface_z[1] <= cutoff) &
            (np.abs(self.data[:, z-1].astype(float) - surface_z[1]) < gap)
        )
        surface_mask = surface_mask_lower | surface_mask_upper
        surface_group = self.data[surface_mask, :]
        # surface_group = surface_group[surface_group[:, self.idx["element"]] != "O", :]
        surface_index = surface_group[:, self.idx["id"]].astype(int) - 1
        surface_index_dict = {}
        for elem in self.elem_u:
            e_idx = np.where(self.data[:, self.idx["element"]] == elem)[0]
            e_index = self.data[e_idx, self.idx["id"]].astype(int) - 1
            e_surface_mask = np.isin(e_index, surface_index)
            surface_index_dict[elem] = e_surface_mask
        if "B3" in surface_index_dict.keys() or "B4" in surface_index_dict.keys():
            surface_index_dict["B"] = np.append(
                surface_index_dict["B3"], surface_index_dict["B4"])
            del surface_index_dict["B4"]
            del surface_index_dict["B3"]
        print(f"Number of surface atoms (after filter): {len(surface_group)}")
        for key, val in self.bond_dict.copy().items():
            if key in surface_index:
                val = [v for v in val if v in surface_index]
                self.bond_dict[key] = val
        return surface_group, surface_index_dict


class descriptor():
    def __init__(self, cn_list, qn_list):
        self.bond_valence = {"Al": 1.651,
                             "B": 1.371,
                             "Ca": 1.967,
                             "Li": 1.466,
                             "Mg": 1.693,
                             "Na": 1.80,
                             "Si": 1.624,
                             "Zr": 1.937}
        self.single_bond_strength = {
            "Si": 106,
            "Al": 101,
            "Zr": 81,
            "Na": 20,
            "Li": 37,
            "Ca": 32,
            "Mg": 36,
            "B4": 89,
            "B3": 119
        }
        self.single_bond_strength["B"] = (
            self.single_bond_strength["B3"] + self.single_bond_strength["B4"])/2
        self.mx = {
            "Si": 4,
            "B3": 3,
            "B4": 4,
            "Al": 4,
            "Zr": 6,
            "Na": 1,
            "Li": 1,
            "Ca": 2,
            "Mg": 2
        }
        self.cn_list = cn_list
        self.qn_list = qn_list
        self.cn = self._calc_cn(cn_list)
        self.qn = self._calc_qn(qn_list)

    def _calc_cn(self, cn_list):
        cn = {}
        for key in cn_list.keys():
            if key != "B":
                cn[key] = np.mean(cn_list[key])
            else:
                cn["B3"] = 3
                cn["B4"] = 4
                cn["B"] = np.mean(cn_list["B"])
        return cn

    def _calc_qn(self, qn_list):
        qn = {key: {} for key in qn_list}
        for key in qn_list.keys():
            u_qn = np.unique(qn_list[key])
            for n in u_qn:
                qn[key][n] = np.sum(qn_list[key] == n)
        return qn

    def calc_fnet(self, lmp, mode="NC", surface=False):
        elem_u = lmp.elem_u
        alkalis = ["Na", "Ca", "Li", "Mg"]
        atoms = lmp.n_atoms if surface else lmp.atoms
        cations = [cation for cation in elem_u if cation != "O"]
        descpt = 0.0
        print("-" * 72)
        print(f"Calculating Fnet in mode: {mode}")
        nc = self._calc_network_conectivity(
            lmp, surface) if mode == "NC" else 1.0
        for cation in cations:
            print(f"Processing element: {cation}")
            aterm = self.mx.get(cation, 1.0) if mode == "mx" else 1.0
            bterm = 1.0
            if mode == "NC":
                if cation not in alkalis:
                    bterm = self._calc_each_network_conectivity(
                        cation, lmp, surface)
                else:
                    bterm = self._calc_ratio(
                        cation, lmp, surface)
            else:
                bterm = 1.0  # No network connectivity adjustment
            n_cation = np.sum(lmp.data[:, lmp.idx["element"]] == cation)
            cn_cation = self.cn.get(cation, 0)
            sbs_cation = self.single_bond_strength.get(cation, 0)
            term = n_cation * cn_cation * sbs_cation * aterm * bterm / atoms
            print("N_cation:", n_cation)
            print("CN_cation:", cn_cation)
            print("sbs_cation:", sbs_cation)
            print("aterm:", aterm)
            print("bterm:", bterm)
            print("atoms:", atoms)
            descpt += term
            print(cation, term)
        # Apply additional multiplicative factors
        if mode == "NC":
            return descpt, nc
        else:
            return descpt

    def _calc_ratio(self, elem, lmp, surface):
        total_sites = sum(list(self.qn[elem].values()))
        elem_cn = sum([qn_val * qn_count / total_sites
                       for qn_val, qn_count in self.qn[elem].items()])
        total_cn = self.cn[elem]
        Mnc = elem_cn / total_cn
        print("Mnc:", Mnc)
        return Mnc

    def _calc_each_network_conectivity(self, elem, lmp, surface):
        if "B" not in elem:
            total_sites = sum(list(self.qn[elem].values()))
            elem_nc = sum([qn_val * qn_count / total_sites
                           for qn_val, qn_count in self.qn[elem].items()])
        else:
            total_sites = sum(list(self.qn["B"].values()))
            elem_nc = sum([qn_val * qn_count / total_sites
                           for qn_val, qn_count in self.qn["B"].items()])
        if elem == "B3":
            N4 = np.sum(self.cn_list["B"] == 4) / self.cn_list["B"].shape[0]
            elem_nc = elem_nc * (1-N4)
        elif elem == "B4":
            N4 = np.sum(self.cn_list["B"] == 4) / self.cn_list["B"].shape[0]
            elem_nc = elem_nc * N4
        print("Mnc:", elem_nc)
        return elem_nc

    def _calc_network_conectivity(self, lmp, surface):
        NC = {elem: 0 for elem in self.qn}
        atoms = lmp.n_atoms if surface else lmp.atoms
        alkali_modifiers = ["Na", "Ca", "Li", "Mg", "O"]
        alkali_modifier_count = np.sum(
            np.isin(lmp.data[:, lmp.idx["element"]], alkali_modifiers)
        )
        net_atoms = atoms - alkali_modifier_count  # ネットワーク原子数
        qn = self.qn.copy()
        for key in alkali_modifiers:
            if key in qn.keys():
                del qn[key]
        for key in qn:
            total_sites = sum(list(qn[key].values()))
            elem_qn = sum([qn_val * qn_count / total_sites
                           for qn_val, qn_count in qn[key].items()])
            NC[key] = (total_sites / net_atoms) * elem_qn
        total_nc = np.sum(list(NC.values()))
        print(f"Total NC (surface={surface}): {total_nc}")
        return total_nc


if __name__ == "__main__":
    import argparse
    import glob
    import pandas as pd
    par = argparse.ArgumentParser(
        description="This is a program for the analysis of oxide glasses")
    par.add_argument("-o", "--output", required=False, type=str)
    par.add_argument("-v", "--verbose", action="store_true")
    par.add_argument("-s", "--surface", action="store_true")
    args = par.parse_args()

    CUTOFF = {
        "Si-O": 2.30,
        "B-O": 2.30,
        "Al-O": 2.50,
        "Zr-O": 2.95,
        "Na-O": 3.12,
        "Li-O": 2.69,
        "Ca-O": 3.14,
        "Mg-O": 2.69,
    }
    network_former = ["Si", "Al", "B", "Zr"]
    CUTOFF = 5.0

    trjfiles = [trjfile for trjfile in sorted(
        glob.glob("*trj")) if "update" not in trjfile]
    data = {}
    linkage_data = {}

    for trjfile in trjfiles:
        lmp = LammpsAnalysis(trjfile, CUTOFF, former=network_former)
        fnet = {"Fnet_mx": 0.0, "Fnet_NC": 0.0,
                "NC": 0.0, "Lc": 0.0}
        counter_time_series = []  # 各ステップのカウンターを格納

        while lmp.get_data():
            if args.verbose:
                print(f"Processing Step {lmp.step}...")
            cn_list = lmp.update_coordination_number(update_type=False)
            qn_list = lmp.calculate_qn(network_former, update_type=False)

            if args.surface:
                surface_data, surface_index = lmp.detect_surface_group(
                    cutoff=CUTOFF)
                lmp.old_data = lmp.data.copy()
                lmp.data = surface_data
                lmp.n_atoms = len(surface_data)
                for key in surface_index.keys():
                    if key in cn_list.keys():
                        cn_list[key] = cn_list[key][surface_index[key]]
                    if key in qn_list.keys():
                        qn_list[key] = qn_list[key][surface_index[key]]

            lmp.counter = lmp.calc_linkages(args.surface)
            total_links = sum(lmp.counter.values())
            normalized_counter = {
                key: value / total_links for key, value in lmp.counter.items()}
            counter_time_series.append(normalized_counter)

            descpt = descriptor(cn_list, qn_list)
            for mode in ["mx", "NC"]:
                if mode == "NC":
                    fnet_, nc = descpt.calc_fnet(lmp,
                                                 mode=mode, surface=args.surface)
                    fnet[f"Fnet_{mode}"] += fnet_
                    fnet["NC"] += nc
                else:
                    fnet[f"Fnet_{mode}"] += descpt.calc_fnet(lmp,
                                                             mode=mode, surface=args.surface)
            for key, val in descpt.cn.items():
                fnet[f"CN_{key}"] = fnet.get(f"CN_{key}", 0) + val
            for key, val in descpt.qn.items():
                total_site = sum(list(val.values()))
                ave_qn = sum([k * v / total_site for k, v in val.items()])
                fnet[f"Qn_{key}"] = fnet.get(f"Qn_{key}", 0) + ave_qn
        averaged_counter = {}
        for step_counter in counter_time_series:
            for key, value in step_counter.items():
                if key not in averaged_counter:
                    averaged_counter[key] = 0
                averaged_counter[key] += value
        for key in averaged_counter.keys():
            averaged_counter[key] /= len(counter_time_series)
            averaged_counter[key] *= 100

        linkage_data[lmp.base] = averaged_counter

        for key in fnet.copy():
            fnet[key] = fnet[key] / lmp.step
        data[lmp.base] = fnet
        basename = f"analyzed_{lmp.base}"

    # データフレームに変換して出力
    linkage_df = pd.DataFrame(linkage_data).T.fillna(0)
    fnet_df = pd.DataFrame.from_dict(data).T.fillna(0)
    df = pd.concat((fnet_df, linkage_df), axis=1)

    with pd.ExcelWriter("descpt.xlsx", engine="openpyxl") as w:
        df.to_excel(w)
        # fnet_df.to_excel(w, sheet_name="Fnet")
        # linkage_df.to_excel(w, sheet_name="Linkages")

    print("descpt.xlsx was created.")
    print(fnet_df)
    print(linkage_df)
