#!/usr/bin/env python
"""Advanced structural analysis utilities for LAMMPS dump files.

This module extends the original glass analysis scripts by adding a set of
commonly used structural descriptors for oxide glasses:

* Partial radial distribution functions g(r)
* Bond angle distributions for arbitrary triplets of elements
* Primitive ring statistics of the network formers

Multiple trajectory files can be processed concurrently via the
``multiprocessing`` module, which allows the analysis of large data sets in a
reasonable time.  The script outputs results as CSV files that can easily be
post-processed with pandas, Excel or plotting utilities.

Example usage::

    python glass_advanced_analysis.py dump.lammpstrj \
        --cutoff Si-O=2.3 --cutoff Al-O=2.4 --network-formers Si Al \
        --rdf-pair Si-O --angle-triplet Si-O-Si --ring-max 12 \
        --processes 4 --output-dir analysis

"""

from __future__ import annotations

import argparse
import itertools as it
import multiprocessing as mp
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd


Array2D = np.ndarray
Pair = Tuple[str, str]
Triplet = Tuple[str, str, str]


def _build_lattice(bounds: np.ndarray) -> Array2D:
    """Construct the lattice matrix from LAMMPS bounds.

    Parameters
    ----------
    bounds:
        A ``(3, 2)`` or ``(3, 3)`` array describing the box bounds.

    Returns
    -------
    numpy.ndarray
        ``3x3`` lattice matrix.
    """

    lattice = np.zeros((3, 3), dtype=float)
    if bounds.shape[1] == 2:  # orthogonal cell
        lattice[0, 0] = bounds[0, 1] - bounds[0, 0]
        lattice[1, 1] = bounds[1, 1] - bounds[1, 0]
        lattice[2, 2] = bounds[2, 1] - bounds[2, 0]
        return lattice

    # triclinic cell with tilt factors (xy, xz, yz)
    xy, xz, yz = bounds[0, 2], bounds[1, 2], bounds[2, 2]
    xlo = bounds[0, 0] - np.min([0.0, xy, xz, xy + xz])
    xhi = bounds[0, 1] - np.max([0.0, xy, xz, xy + xz])
    ylo = bounds[1, 0] - np.min([0.0, yz])
    yhi = bounds[1, 1] - np.max([0.0, yz])
    lattice[0] = (xhi - xlo, 0.0, 0.0)
    lattice[1] = (xy, yhi - ylo, 0.0)
    lattice[2] = (xz, yz, bounds[2, 1] - bounds[2, 0])
    return lattice


@dataclass
class Frame:
    """Single snapshot of a LAMMPS trajectory."""

    step: int
    coords: Array2D
    elements: np.ndarray
    lattice: Array2D
    inv_lattice: Array2D
    volume: float
    element_indices: Dict[str, np.ndarray]

    def displacement(self, idx_from: int, idx_to: int) -> np.ndarray:
        """Return the minimum image displacement ``from -> to``."""

        delta = self.coords[idx_to] - self.coords[idx_from]
        delta = delta @ self.inv_lattice
        delta -= np.rint(delta)
        return delta @ self.lattice


class LammpsTrajectory:
    """Lazy iterator over the frames of a LAMMPS ``.lammpstrj`` file."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.base = self.path.stem

    def __iter__(self) -> Iterator[Frame]:
        with self.path.open() as fh:
            while True:
                header = [fh.readline() for _ in range(9)]
                if not header[0]:
                    break
                step = int(header[1])
                n_atoms = int(header[3])
                bounds = np.fromstring(" ".join(header[5:8]), sep=" ", dtype=float)
                bounds = bounds.reshape(3, -1)
                lattice = _build_lattice(bounds)
                inv_lattice = np.linalg.inv(lattice)
                volume = float(abs(np.linalg.det(lattice)))
                columns = header[8].split()[2:]
                data = [fh.readline().split() for _ in range(n_atoms)]
                if len(data) != n_atoms:
                    break
                arr = np.array(data, dtype=object)
                idx_map = {col: i for i, col in enumerate(columns)}
                elements = arr[:, idx_map["element"]]
                coords = arr[:, [idx_map["x"], idx_map["y"], idx_map["z"]]].astype(float)
                element_indices = {
                    elem: np.where(elements == elem)[0]
                    for elem in np.unique(elements)
                }
                yield Frame(
                    step=step,
                    coords=coords,
                    elements=elements,
                    lattice=lattice,
                    inv_lattice=inv_lattice,
                    volume=volume,
                    element_indices=element_indices,
                )


def _parse_pairs(values: Optional[Sequence[str]]) -> Optional[List[Pair]]:
    if not values:
        return None
    pairs: List[Pair] = []
    for value in values:
        elems = value.split("-")
        if len(elems) != 2:
            raise argparse.ArgumentTypeError(
                f"Invalid pair specification '{value}'. Use the form 'A-B'."
            )
        pairs.append((elems[0], elems[1]))
    return pairs


def _parse_triplets(values: Optional[Sequence[str]]) -> Optional[List[Triplet]]:
    if not values:
        return None
    triplets: List[Triplet] = []
    for value in values:
        elems = value.split("-")
        if len(elems) != 3:
            raise argparse.ArgumentTypeError(
                f"Invalid triplet specification '{value}'. Use 'A-B-C'."
            )
        triplets.append((elems[0], elems[1], elems[2]))
    return triplets


def _parse_cutoffs(values: Sequence[str]) -> Dict[Pair, float]:
    cutoff_map: Dict[Pair, float] = {}
    for item in values:
        if "=" not in item:
            raise argparse.ArgumentTypeError(
                f"Invalid cutoff specification '{item}'. Use 'A-B=value'."
            )
        pair_str, value_str = item.split("=", 1)
        elems = pair_str.split("-")
        if len(elems) != 2:
            raise argparse.ArgumentTypeError(
                f"Invalid cutoff specification '{item}'. Use 'A-B=value'."
            )
        try:
            cutoff_val = float(value_str)
        except ValueError as exc:  # pragma: no cover - defensive
            raise argparse.ArgumentTypeError(
                f"Invalid cutoff value in '{item}': {value_str}"
            ) from exc
        a, b = elems[0], elems[1]
        cutoff_map[(a, b)] = cutoff_val
        cutoff_map[(b, a)] = cutoff_val
    return cutoff_map


class Analyzer:
    """Accumulate structural descriptors frame by frame."""

    def __init__(
        self,
        cutoff_map: Dict[Pair, float],
        network_formers: Sequence[str],
        rmax: float,
        rdf_bins: int,
        angle_bins: int,
        ring_max: int,
        rdf_pairs: Optional[List[Pair]] = None,
        angle_triplets: Optional[List[Triplet]] = None,
        compute_rings: bool = True,
    ) -> None:
        self.cutoff_map = cutoff_map
        self.network_formers = list(network_formers)
        self.rdf_max = rmax
        self.rdf_bins = rdf_bins
        self.angle_bins = angle_bins
        self.ring_max = ring_max
        self.requested_pairs = rdf_pairs
        self.requested_triplets = angle_triplets
        self.compute_rings = compute_rings
        self.initialized = False

        self.rdf_edges = np.linspace(0.0, self.rdf_max, self.rdf_bins + 1)
        self.rdf_hist: Dict[Pair, np.ndarray] = {}
        self.rdf_meta: Dict[Pair, Dict[str, float]] = {}

        self.angle_edges = np.linspace(0.0, 180.0, self.angle_bins + 1)
        self.angle_hist: Dict[Triplet, np.ndarray] = {}

        self.ring_counter: Counter[int] = Counter()
        self.frame_count = 0

    def _initialize(self, frame: Frame) -> None:
        elements = sorted(frame.element_indices.keys())
        if self.requested_pairs is None:
            self.rdf_pairs = list(it.combinations_with_replacement(elements, 2))
        else:
            self.rdf_pairs = self.requested_pairs
        if self.requested_triplets is None:
            if "O" in elements:
                triplets: List[Triplet] = []
                for elem in elements:
                    if elem == "O":
                        continue
                    if elem in self.network_formers:
                        triplets.append((elem, "O", elem))
                self.angle_triplets = triplets
            else:
                self.angle_triplets = []
        else:
            self.angle_triplets = self.requested_triplets

        self.active_network_formers = [
            nf for nf in self.network_formers if nf in elements
        ]

        self.rdf_hist = {pair: np.zeros(self.rdf_bins, dtype=float)
                         for pair in self.rdf_pairs}
        self.rdf_meta = {
            pair: {"frames": 0.0, "n_a": 0.0, "n_b": 0.0, "volume": 0.0}
            for pair in self.rdf_pairs
        }
        self.angle_hist = {
            triplet: np.zeros(self.angle_bins, dtype=float)
            for triplet in self.angle_triplets
        }

        self.required_neighbor_pairs: Set[Pair] = set()
        for triplet in self.angle_triplets:
            left, center, right = triplet
            self.required_neighbor_pairs.add((center, left))
            self.required_neighbor_pairs.add((center, right))
        if self.compute_rings:
            for nf in self.active_network_formers:
                self.required_neighbor_pairs.add(("O", nf))
        self.initialized = True

    def _neighbor_cache(self, frame: Frame) -> Dict[str, Dict[str, Dict[int, List[int]]]]:
        cache: Dict[str, Dict[str, Dict[int, List[int]]]] = defaultdict(lambda: defaultdict(dict))
        for center, neighbor in self.required_neighbor_pairs:
            cutoff = self.cutoff_map.get((center, neighbor))
            if cutoff is None:
                continue
            center_indices = frame.element_indices.get(center)
            neighbor_indices = frame.element_indices.get(neighbor)
            if center_indices is None or neighbor_indices is None:
                continue
            center_xyz = frame.coords[center_indices]
            neighbor_xyz = frame.coords[neighbor_indices]
            delta = neighbor_xyz[:, None, :] - center_xyz
            delta = delta @ frame.inv_lattice
            delta -= np.rint(delta)
            delta = delta @ frame.lattice
            dist = np.linalg.norm(delta, axis=2)
            within = dist < cutoff
            rows, cols = np.where(within)
            for n_idx, c_idx in zip(rows, cols):
                center_atom = int(center_indices[c_idx])
                neighbor_atom = int(neighbor_indices[n_idx])
                cache[center][neighbor].setdefault(center_atom, []).append(neighbor_atom)
        return cache

    def _accumulate_rdf(self, frame: Frame) -> None:
        for pair in self.rdf_pairs:
            a, b = pair
            idx_a = frame.element_indices.get(a)
            idx_b = frame.element_indices.get(b)
            if idx_a is None or idx_b is None:
                continue
            coords_a = frame.coords[idx_a]
            coords_b = frame.coords[idx_b]
            delta = coords_b[:, None, :] - coords_a
            delta = delta @ frame.inv_lattice
            delta -= np.rint(delta)
            delta = delta @ frame.lattice
            dist = np.linalg.norm(delta, axis=2)
            if a == b:
                iu = np.triu_indices(len(idx_a), k=1)
                dist = dist[iu]
            else:
                dist = dist.ravel()
            meta = self.rdf_meta[pair]
            meta["frames"] += 1
            meta["n_a"] += float(len(idx_a))
            if a == b:
                meta["n_b"] += max(len(idx_b) - 1, 0)
            else:
                meta["n_b"] += float(len(idx_b))
            meta["volume"] += frame.volume
            dist = dist[dist > 1.0e-8]
            if dist.size == 0:
                continue
            hist, _ = np.histogram(dist, bins=self.rdf_edges)
            self.rdf_hist[pair] += hist

    def _accumulate_angles(
        self, frame: Frame, neighbor_cache: Dict[str, Dict[str, Dict[int, List[int]]]]
    ) -> None:
        for triplet in self.angle_triplets:
            left, center, right = triplet
            center_dict = neighbor_cache.get(center, {})
            left_neighbors = center_dict.get(left, {})
            right_neighbors = center_dict.get(right, {})
            common_centers = set(left_neighbors.keys()) & set(right_neighbors.keys())
            if not common_centers:
                continue
            angles: List[float] = []
            for center_idx in common_centers:
                left_list = left_neighbors.get(center_idx, [])
                right_list = right_neighbors.get(center_idx, [])
                if left == right:
                    neighbor_pairs = it.combinations(sorted(left_list), 2)
                else:
                    neighbor_pairs = it.product(left_list, right_list)
                for left_idx, right_idx in neighbor_pairs:
                    vec1 = frame.displacement(center_idx, left_idx)
                    vec2 = frame.displacement(center_idx, right_idx)
                    norm1 = np.linalg.norm(vec1)
                    norm2 = np.linalg.norm(vec2)
                    if norm1 < 1e-12 or norm2 < 1e-12:
                        continue
                    cos_theta = np.dot(vec1, vec2) / (norm1 * norm2)
                    cos_theta = np.clip(cos_theta, -1.0, 1.0)
                    angles.append(np.degrees(np.arccos(cos_theta)))
            if not angles:
                continue
            hist, _ = np.histogram(angles, bins=self.angle_edges)
            self.angle_hist[triplet] += hist

    def _accumulate_rings(
        self, frame: Frame, neighbor_cache: Dict[str, Dict[str, Dict[int, List[int]]]]
    ) -> None:
        if not self.compute_rings:
            return
        oxygen_neighbors: Dict[int, set[int]] = defaultdict(set)
        for nf in self.active_network_formers:
            for oxygen_idx, neighbors in neighbor_cache.get("O", {}).get(nf, {}).items():
                oxygen_neighbors[oxygen_idx].update(neighbors)
        graph: Dict[int, set[int]] = defaultdict(set)
        for neighbors in oxygen_neighbors.values():
            if len(neighbors) < 2:
                continue
            nf_list = sorted(neighbors)
            for i in range(len(nf_list)):
                for j in range(i + 1, len(nf_list)):
                    u, v = nf_list[i], nf_list[j]
                    graph[u].add(v)
                    graph[v].add(u)
        if not graph:
            return

        from collections import deque

        def shortest_path_length(start: int, goal: int) -> Optional[int]:
            if start == goal:
                return 0
            queue: deque[Tuple[int, int]] = deque([(start, 0)])
            visited = {start}
            while queue:
                node, dist = queue.popleft()
                for neigh in graph[node]:
                    if neigh == goal:
                        return dist + 1
                    if neigh in visited or neigh == start:
                        continue
                    visited.add(neigh)
                    queue.append((neigh, dist + 1))
            return None

        processed = set()
        edges = [(u, v) for u in graph for v in graph[u] if u < v]
        for u, v in edges:
            if (u, v) in processed:
                continue
            graph[u].remove(v)
            graph[v].remove(u)
            path_length = shortest_path_length(u, v)
            graph[u].add(v)
            graph[v].add(u)
            processed.add((u, v))
            if path_length is None:
                continue
            ring_size = path_length + 1
            if ring_size <= self.ring_max:
                self.ring_counter[ring_size] += 1.0 / ring_size

    def process_frame(self, frame: Frame) -> None:
        if not self.initialized:
            self._initialize(frame)
        self.frame_count += 1
        neighbor_cache = self._neighbor_cache(frame)
        self._accumulate_rdf(frame)
        if self.angle_triplets:
            self._accumulate_angles(frame, neighbor_cache)
        self._accumulate_rings(frame, neighbor_cache)

    def finalize(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        results: Dict[str, Dict[str, pd.DataFrame]] = {}
        if self.rdf_hist:
            rdf_results: Dict[str, pd.DataFrame] = {}
            dr = np.diff(self.rdf_edges)
            r_centers = self.rdf_edges[:-1] + dr / 2.0
            shell_volume = 4.0 * np.pi * r_centers**2 * dr
            for pair, hist in self.rdf_hist.items():
                meta = self.rdf_meta[pair]
                frames = meta["frames"]
                if frames == 0:
                    continue
                avg_volume = meta["volume"] / frames
                avg_n_a = meta["n_a"] / frames
                avg_n_b = meta["n_b"] / frames
                if avg_volume < 1e-12 or avg_n_a < 1e-12 or avg_n_b < 1e-12:
                    g_r = np.zeros_like(hist, dtype=float)
                else:
                    rho_b = avg_n_b / avg_volume
                    denom = frames * avg_n_a * shell_volume * rho_b
                    denom[denom < 1e-12] = np.inf
                    g_r = hist / denom
                df = pd.DataFrame({"r": r_centers, "g_r": g_r})
                rdf_results[f"{pair[0]}-{pair[1]}"] = df
            results["rdf"] = rdf_results
        if self.angle_hist:
            angle_results: Dict[str, pd.DataFrame] = {}
            angle_centers = self.angle_edges[:-1] + np.diff(self.angle_edges) / 2.0
            for triplet, hist in self.angle_hist.items():
                total = hist.sum()
                if total > 0.0:
                    prob = hist / total
                else:
                    prob = np.zeros_like(hist, dtype=float)
                df = pd.DataFrame(
                    {
                        "angle_deg": angle_centers,
                        "count": hist,
                        "probability": prob,
                    }
                )
                angle_results[f"{triplet[0]}-{triplet[1]}-{triplet[2]}"] = df
            results["angles"] = angle_results
        if self.compute_rings and self.ring_counter:
            data = sorted(self.ring_counter.items())
            df = pd.DataFrame(data, columns=["ring_size", "count_per_frame"])
            if self.frame_count > 0:
                df["count_per_frame"] = df["count_per_frame"] / self.frame_count
            results["rings"] = {"rings": df}
        return results

    def write_outputs(self, output_dir: Path, results: Dict[str, Dict[str, pd.DataFrame]]) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        rdf_results = results.get("rdf", {})
        for pair, df in rdf_results.items():
            filename = output_dir / f"{pair.replace('-', '_')}_rdf.csv"
            df.to_csv(filename, index=False)
        angle_results = results.get("angles", {})
        for triplet, df in angle_results.items():
            filename = output_dir / f"{triplet.replace('-', '_')}_angles.csv"
            df.to_csv(filename, index=False)
        ring_results = results.get("rings")
        if ring_results:
            ring_df = ring_results["rings"]
            ring_df.to_csv(output_dir / "ring_distribution.csv", index=False)


def analyze_file(
    filename: str,
    args: argparse.Namespace,
    cutoff_map: Dict[Pair, float],
) -> Tuple[str, str]:
    trajectory = LammpsTrajectory(Path(filename))
    analyzer = Analyzer(
        cutoff_map=cutoff_map,
        network_formers=args.network_formers,
        rmax=args.rmax,
        rdf_bins=args.rdf_bins,
        angle_bins=args.angle_bins,
        ring_max=args.ring_max,
        rdf_pairs=args.rdf_pairs,
        angle_triplets=args.angle_triplets,
        compute_rings=args.rings,
    )
    for frame in trajectory:
        analyzer.process_frame(frame)
    results = analyzer.finalize()
    output_root = Path(args.output_dir or ".").resolve()
    out_dir = output_root / trajectory.base
    analyzer.write_outputs(out_dir, results)
    return (filename, str(out_dir))


def _run_serial(filenames: Sequence[str], func) -> List[Tuple[str, str]]:
    return [func(name) for name in filenames]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Advanced structural analysis for oxide glasses",
    )
    parser.add_argument("lammpstrj", nargs="+", help="LAMMPS dump files to analyse")
    parser.add_argument(
        "--cutoff",
        action="append",
        default=["Si-O=2.3", "B-O=2.3", "Al-O=2.4"],
        help="Bond cutoff distances (A-B=value). Provide multiple times for different pairs.",
    )
    parser.add_argument(
        "--network-formers",
        nargs="+",
        default=["Si", "B", "Al"],
        help="Elements treated as network formers for ring detection.",
    )
    parser.add_argument(
        "--rdf-pair",
        dest="rdf_pairs",
        action="append",
        help="Element pair for g(r) calculation (A-B). Can be used multiple times.",
    )
    parser.add_argument(
        "--angle-triplet",
        dest="angle_triplets",
        action="append",
        help="Element triplet for angle distribution (A-B-C). Can be repeated.",
    )
    parser.add_argument("--rmax", type=float, default=10.0, help="Maximum radius for g(r) in angstrom")
    parser.add_argument("--rdf-bins", type=int, default=200, help="Number of bins for g(r)")
    parser.add_argument("--angle-bins", type=int, default=180, help="Number of bins for angle distribution")
    parser.add_argument("--ring-max", type=int, default=12, help="Maximum ring size to record")
    parser.add_argument(
        "--rings",
        dest="rings",
        action="store_true",
        help="Enable ring statistics (enabled by default)",
    )
    parser.add_argument(
        "--no-rings",
        dest="rings",
        action="store_false",
        help="Disable ring statistics",
    )
    parser.set_defaults(rings=True)
    parser.add_argument(
        "--processes",
        type=int,
        default=1,
        help="Number of worker processes (>=1). Use >1 for multiprocessing.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Root directory for analysis outputs (default: current directory)",
    )

    args = parser.parse_args()
    args.rdf_pairs = _parse_pairs(args.rdf_pairs)
    args.angle_triplets = _parse_triplets(args.angle_triplets)
    cutoff_map = _parse_cutoffs(args.cutoff)

    worker = partial(analyze_file, args=args, cutoff_map=cutoff_map)
    filenames = sorted(args.lammpstrj)
    if args.processes <= 1 or len(filenames) == 1:
        results = _run_serial(filenames, worker)
    else:
        with mp.Pool(processes=args.processes) as pool:
            results = pool.map(worker, filenames)

    for filename, out_dir in results:
        print(f"Processed {filename} -> {out_dir}")


if __name__ == "__main__":
    main()

