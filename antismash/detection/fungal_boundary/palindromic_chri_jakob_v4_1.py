#!/usr/bin/env python3
import argparse
import csv
import math
import random
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from contextlib import closing, nullcontext
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path

import numpy as np
from Bio import SeqIO
from scipy.stats import poisson

from utils import (
    candidate_windows,
    count_sites,
    decode_code,
    edge_evidence,
    encode_kmers,
    motif_indices,
    motif_name,
    motif_ranges,
    ordered_parallel_map,
    reverse_codes,
    reverse_complement,
    write_report,
)


VERSION = "0.4.2"
MODE_ORDER = ("anchor", "adjacent", "all")
REPEAT_TYPES = ("inverted", "direct", "linear")

_worker_genome = None
_worker_settings = None


def _init_anchor_worker(genome, settings):
    """Install read-only genome data once per worker, rather than per anchor."""
    global _worker_genome, _worker_settings
    _worker_genome = genome
    _worker_settings = settings


def _null_anchor_worker(location):
    return null_anchor_scores(_worker_genome, location, _worker_settings)


def _input_anchor_worker(row, calibration):
    return analyse_input_safely(row, _worker_genome, calibration, _worker_settings)


@dataclass(frozen=True)
class Settings:
    candidate_modes: tuple = MODE_ORDER
    repeat_types: tuple = ("direct", "inverted", "linear")
    n_flanking: int = 120
    upstream: int = 1000
    downstream: int = 50
    part_size: int = 3
    min_spacer: int = 1
    max_spacer: int = 15
    linear_length: int = 6
    linear_upstream: int = 300  # inherits the upstream value if set to 0
    linear_window_penalty: float = 0.0 # penelies longer gene windows for linear motifs - 0 disables the penalty
    min_linear_support_fraction: float = 0.75 # minimum fraction of genes in the predicted BGC (only linear) - 0 disables the filter
    selection_policy: str = "repeat_first" # or "most_significant"
    max_sites_per_promoter: int = 3
    min_bgc_size: int = 2
    max_bgc_size: int = 50
    min_genes_with_site: int = 3
    null_anchors: int = 200 # Number of random anchor genes sampled per genome to estimate how unusual the observed motif score is
    null_seed: int = 1
    null_min_flank: int = 20 # Match the calibration-anchor pool used for the saved V4 run.
    alpha: float = 0.05 # empirical p-value threshold
    edge_flank_genes: int = 3  # Genes compared just inside and outside each boundary to assess motif enrichment at the edges.
    max_predicted_genes: int = 30
    min_llr_per_gene: float = 0.75 #LLR per predicted gene required
    expansion_genes: int = 2 # Extra genes included on each side of an accepted window in the reported region
    retry_rejected: bool = True


class Genome:
    """Load GenBank genes and store their promoters, motif counts, and locations."""

    def __init__(self, path, settings):
        self.contigs = []
        self.locations = defaultdict(list)
        self.counts = []
        self.genes = []
        self.promoter_keys = []
        self.promoter_sequences = {}
        self.promoter_bounds = {}
        self.linear_promoter_sequences = {}
        self.linear_promoter_bounds = {}
        promoters = {}
        width = motif_ranges(settings)[-1][2] #storage needed
        self.total_site_counts = np.zeros(width, dtype=float)
        self.total_squared_site_counts = np.zeros(width, dtype=float)
        self.promoters_with_site = np.zeros(width, dtype=float)
        with open(path) as handle:
            for record in SeqIO.parse(handle, "genbank"):
                # v2 defines relative gene positions and random samples in feature order.
                features = [f for f in record.features if f.type == "gene"]
                if not features:
                    continue
                contig = {"id": record.id, "gene_indices": []}
                for position, feature in enumerate(features):
                    start, end = int(feature.location.start), int(feature.location.end)
                    strand = feature.location.strand
                    if strand not in (-1, 1):
                        raise ValueError(f"Gene at {record.id}:{start} has no usable strand")
                    locus = feature.qualifiers.get("locus_tag", feature.qualifiers.get("gene", ["unknown"]))[0]
                    if strand == 1:
                        # lo = gene start minus the upstream length, hi = ...
                        lo, hi = max(0, start - settings.upstream), min(len(record), start + settings.downstream)
                        promoter = record.seq[lo:hi]
                    else:
                        lo, hi = max(0, end - settings.downstream), min(len(record), end + settings.upstream)
                        promoter = record.seq[lo:hi].reverse_complement()
                    key = (record.id, start, end, strand)
                    if key not in promoters:
                        sequence = str(promoter).upper().replace("U", "T")
                        self.promoter_sequences[key] = sequence
                        self.promoter_bounds[key] = (lo, hi)
                        tss_offset = start - lo if strand == 1 else hi - end
                        crop = max(0, tss_offset - (settings.linear_upstream or settings.upstream))
                        linear_sequence = sequence[crop:]
                        self.linear_promoter_sequences[key] = linear_sequence
                        self.linear_promoter_bounds[key] = (lo + crop, hi) if strand == 1 else (lo, hi - crop)
                        if crop and "linear" in settings.repeat_types:
                            repeat_settings = replace(settings, repeat_types=tuple(t for t in settings.repeat_types if t != "linear"))
                            linear_settings = replace(settings, repeat_types=("linear",))
                            repeat_ids, repeat_counts = count_sites(sequence, repeat_settings)
                            linear_ids, linear_counts = count_sites(linear_sequence, linear_settings)
                            indices = np.r_[repeat_ids, linear_ids]
                            counts = np.r_[repeat_counts, linear_counts]
                        else:
                            indices, counts = count_sites(sequence, settings)
                        promoters[key] = (indices, counts)
                        self.total_site_counts[indices] += counts
                        self.total_squared_site_counts[indices] += counts.astype(float) ** 2
                        self.promoters_with_site[indices] += 1
                    indices, counts = promoters[key]
                    self.promoter_keys.append(key)
                    self.counts.append((indices, counts))
                    index = len(self.genes)
                    self.genes.append({"locus": locus, "start": start, "end": end, "strand": strand,
                                       "tss_proxy": start if strand == 1 else end})
                    self.locations[locus].append((len(self.contigs), position))
                    contig["gene_indices"].append(index)
                self.contigs.append(contig)
        if not self.genes:
            raise ValueError("Genome contains no gene features")
        self.background_gene_count = len(promoters)

    def locate(self, anchor):
        """finds a gene by its identifier"""
        matches = self.locations.get(anchor, [])
        if len(matches) != 1:
            raise ValueError(f"Anchor {anchor!r} matches {len(matches)} gene features; expected one")
        return matches[0]




def score_anchor(genome, location, settings):
    """
    TODO: Should be checked.
    TODO: should be more readable.
    This function asks the question: For each motif, which gene window around this anchor shows the strongest enrichment?
    """
    contig, anchor = location
    indices = genome.contigs[contig]["gene_indices"]
    first, stop = max(0, anchor - settings.n_flanking), min(len(indices), anchor + settings.n_flanking + 1)
    local = indices[first:stop]
    center = anchor - first
    counts = np.zeros((len(local), genome.total_site_counts.size), dtype=float)
    for row, index in enumerate(local):
        motifs, sites = genome.counts[index]
        counts[row, motifs] = sites
    # Duplicate feature aliases share one promoter in v2's genome background.
    unique_rows = list({genome.promoter_keys[index]: row for row, index in enumerate(local)}.values())
    excluded = counts[unique_rows]
    n_background = genome.background_gene_count - len(unique_rows)
    if n_background <= 0:
        raise ValueError("No background promoters remain outside the anchor neighborhood")
    mean = (genome.total_site_counts - excluded.sum(axis=0)) / n_background
    variance = (genome.total_squared_site_counts - (excluded ** 2).sum(axis=0)) / n_background - mean ** 2
    frequency = (genome.promoters_with_site - (excluded > 0).sum(axis=0)) / n_background
    rates = np.maximum(mean, 0.5 / n_background)
    lefts, rights = candidate_windows(len(local), center, settings.min_bgc_size, settings.max_bgc_size)
    sizes = rights - lefts + 1
    width = counts.shape[1]
    scores = {key: np.zeros(width) for key in ("llr", "sites", "genes_with_site", "expected")}
    best_windows = np.zeros(width, dtype=int)
    linear_start = 2 * 4 ** settings.part_size * (settings.max_spacer + 1)
    if settings.linear_window_penalty:
        best_windows[linear_start:] = int(sizes.argmin())

    # Expected motif occurrences = genes in each motif's chosen window × its background occurrences per promoter.
    scores["expected"] = sizes[best_windows] * rates
    # A motif absent from every local promoter has LLR=0 in every window.
    # Preserve its original first-window scores without computing empty columns.
    active = np.flatnonzero(counts.sum(axis=0))
    for first_motif in range(0, len(active), 512):
        target = active[first_motif:first_motif + 512]
        subset = np.ascontiguousarray(counts[:, target])
        prefix = np.vstack((np.zeros(subset.shape[1]), subset.cumsum(axis=0)))
        occupied = np.vstack((np.zeros(subset.shape[1]), (subset > 0).cumsum(axis=0)))
        sites = prefix[rights + 1] - prefix[lefts]
        support = occupied[rights + 1] - occupied[lefts]
        expected = sizes[:, None] * rates[None, target]
        with np.errstate(divide="ignore", invalid="ignore"):
            llr = np.where(sites > expected, sites * np.log(sites / expected) - (sites - expected), 0.0)
        llr[support < settings.min_genes_with_site] = 0
        objective = llr
        if settings.linear_window_penalty:
            objective = llr - sizes[:, None] * np.where(target >= linear_start, settings.linear_window_penalty, 0.)
        best = objective.argmax(axis=0)
        columns = np.arange(subset.shape[1])
        best_windows[target] = best
        for key, values in (("llr", llr), ("sites", sites), ("genes_with_site", support), ("expected", expected)):
            scores[key][target] = values[best, columns]
    scores.update(left=lefts[best_windows], right=rights[best_windows], size=sizes[best_windows],
                  bg_mean=mean, bg_variance=variance, bg_fraction=frequency)
    all_motifs = np.concatenate([motif_indices(family, settings) for family in settings.repeat_types])
    modes = {"all": np.sort(all_motifs), "anchor": np.flatnonzero(counts[center]),
             "adjacent": np.flatnonzero(counts[max(0, center - 1):center + 2].sum(axis=0))}
    return {"scores": scores, "counts": counts, "local": local, "center": center,
            "location": location, "modes": {m: modes[m] for m in settings.candidate_modes},
            "n_windows": len(lefts), "n_background": n_background}


def empirical_p(llr, null):
    return (1 + sum(value >= llr for value in null)) / (len(null) + 1) if len(null) else None


def null_anchor_scores(genome, location, settings):
    """Return only null maxima, keeping large score matrices inside the worker."""
    try:
        result = score_anchor(genome, location, settings)
    except ValueError as error:
        return None, str(error)
    tables = {}
    for mode, indices in result["modes"].items():
        tables[mode] = {}
        for family, start, stop in motif_ranges(settings):
            if family not in settings.repeat_types:
                continue
            subset = indices[(indices >= start) & (indices < stop)]
            tables[mode][family] = float(result["scores"]["llr"][subset].max()) if len(subset) else 0.0
    return tables, None


def calibrate(genome, settings, *, executor=None, workers=1):
    pool = []
    for contig, data in enumerate(genome.contigs):
        for position in range(settings.null_min_flank, len(data["gene_indices"]) - settings.null_min_flank):
            locus = genome.genes[data["gene_indices"][position]]["locus"]
            if locus != "unknown":
                pool.append((contig, position))
    picks = random.Random(settings.null_seed).sample(pool, min(settings.null_anchors, len(pool)))
    tables = {mode: {family: [] for family in settings.repeat_types} for mode in settings.candidate_modes}
    samples, failures = [], []
    identities, locations = [], []
    for sampled_location in picks:
        sampled_contig, sampled_position = sampled_location
        locus = genome.genes[genome.contigs[sampled_contig]["gene_indices"][sampled_position]]["locus"]
        # v2 samples locus labels and resolves duplicate labels to their first feature.
        location = genome.locations[locus][0]
        contig, position = location
        identity = {"contig": genome.contigs[contig]["id"], "gene_index": position,
                    "locus": locus, "sampled_contig": genome.contigs[sampled_contig]["id"],
                    "sampled_gene_index": sampled_position}
        identities.append(identity)
        locations.append(location)
    score = (_null_anchor_worker if executor is not None else
             partial(null_anchor_scores, genome, settings=settings))
    results = ordered_parallel_map(score, locations, executor=executor, max_pending=2 * workers)
    for identity, (maxima, error) in zip(identities, results):
        if error is not None:
            failures.append({**identity, "error": error})
            continue
        for mode in settings.candidate_modes:
            for family in settings.repeat_types:
                tables[mode][family].append(maxima[mode][family])
        samples.append(identity)
    return {"pool_size": len(pool), "requested": settings.null_anchors,
            "successful": len(samples), "samples": samples, "failures": failures,
            "scores": tables,
            "llr_95": {mode: {family: float(np.quantile(values, .95)) if values else None
                               for family, values in families.items()} for mode, families in tables.items()}}


def candidate_priority(candidate):
    """Rank significance first; finite null samples can give multiple p-value ties."""
    return (candidate["p_empirical"], -candidate["llr"], MODE_ORDER.index(candidate["mode"]),
            candidate["repeat_type"], candidate["motif_index"])


def most_significant_candidate(candidates):
    return min((c for c in candidates if c["p_empirical"] is not None and c["llr"] > 0),
               key=candidate_priority, default=None)


def apply_rules(candidates, settings):
    for candidate in candidates:
        checks = {"enriched": candidate["llr"] > 0,
                  "enough_genes_with_site": candidate["genes_with_site"] >= settings.min_genes_with_site,
                  "calibrated": candidate["p_empirical"] is not None,
                  "significant": candidate["p_empirical"] is not None and candidate["p_empirical"] < settings.alpha,
                  "span_within_limit": candidate["predicted_gene_count"] <= settings.max_predicted_genes,
                  "llr_per_gene_at_least_minimum": candidate["llr_per_gene"] >= settings.min_llr_per_gene}
        if settings.min_linear_support_fraction:
            checks["linear_support_fraction"] = (candidate["repeat_type"] != "linear" or
                candidate["genes_with_site"] / candidate["predicted_gene_count"] >= settings.min_linear_support_fraction)
        candidate["checks"] = checks
        candidate["failed_checks"] = [name for name, passed in checks.items() if not passed]
        candidate["passes_checks"] = all(checks.values())
        candidate["selected"] = False
    if settings.retry_rejected or settings.selection_policy == "repeat_first":
        options = [c for c in candidates if c["passes_checks"]]
    else:
        # Select first, then filter; the default does not substitute a weaker passing hit.
        options = candidates
    if settings.selection_policy == "repeat_first":
        chosen = min(options, key=lambda c: (c["repeat_type"] == "linear",) + candidate_priority(c), default=None)
    else:
        chosen = most_significant_candidate(options)
    if chosen is None or not chosen["passes_checks"]:
        return None
    chosen["selected"] = True
    return chosen


def make_candidates(result, calibration, settings):
    scores = result["scores"]
    nominal = poisson.sf(scores["sites"] - 1, scores["expected"])
    nominal[scores["llr"] <= 0] = 1
    candidates, mode_decisions = [], []
    for mode, indices in result["modes"].items():
        mode_candidates = []
        ranked = sorted(map(int, indices), key=lambda i: (-scores["llr"][i], i))
        best_by_type, ranks = {}, Counter()
        for index in ranked:
            family, motif = motif_name(index, settings)
            ranks[family] += 1
            left, right = int(scores["left"][index]), int(scores["right"][index])
            llr, size = float(scores["llr"][index]), int(scores["size"][index])
            null = calibration["scores"][mode][family]
            mean = float(scores["bg_mean"][index])
            candidate = {"id": f"{mode}:{index}", "mode": mode, "repeat_type": family,
                         "motif": motif, "motif_index": index, "rank_in_type": ranks[family],
                         "family_best": family not in best_by_type and llr > 0,
                         "left": left - result["center"], "right": right - result["center"],
                         "predicted_gene_count": size,
                         "llr": llr, "llr_per_gene": llr / size,
                         "sites": int(scores["sites"][index]),
                         "genes_with_site": int(scores["genes_with_site"][index]),
                         "anchor_sites": int(result["counts"][result["center"], index]),
                         "expected": float(scores["expected"][index]),
                         "background_mean": mean,
                         "background_fraction": float(scores["bg_fraction"][index]),
                         "background_dispersion": float(scores["bg_variance"][index] / mean) if mean > 0 else None,
                         "p_nominal": float(nominal[index]),
                         "p_bonferroni": min(1.0, float(nominal[index]) * len(indices) * result["n_windows"]),
                         "p_empirical": empirical_p(llr, null), "null_n": len(null),
                         "null_llr_95": calibration["llr_95"][mode][family],
                         **edge_evidence(result["counts"][:, index], left, right, settings.edge_flank_genes)}
            candidates.append(candidate)
            mode_candidates.append(candidate)
            if candidate["family_best"]:
                best_by_type[family] = candidate
        selected = most_significant_candidate(mode_candidates)
        reason = "most_significant_candidate" if selected else "no_calibrated_enriched_candidate"
        significant = selected is not None and selected["p_empirical"] is not None and selected["p_empirical"] < settings.alpha
        mode_decisions.append({"mode": mode, "candidate_count": len(indices),
                               "best_by_type": {family: c["id"] for family, c in best_by_type.items()},
                               "selected_candidate_id": selected["id"] if selected else None,
                               "reason": reason, "significant_single_call": significant})
    return candidates, mode_decisions


def reference_info(genome, location, known):
    if not known:
        return None
    contig, anchor = location
    mapped, missing, ambiguous, other_contigs = [], [], [], []
    for locus in known:
        matches = genome.locations.get(locus, [])
        if not matches:
            missing.append(locus)
        elif len(matches) != 1:
            ambiguous.append(locus)
        elif matches[0][0] != contig:
            other_contigs.append(locus)
        else:
            mapped.append({"locus": locus, "relative_position": matches[0][1] - anchor})
    complete = not (missing or ambiguous or other_contigs)
    return {"complete": complete, "genes": mapped, "missing_loci": missing,
            "ambiguous_loci": ambiguous, "loci_on_other_contigs": other_contigs,
            "left": min(g["relative_position"] for g in mapped) if mapped else None,
            "right": max(g["relative_position"] for g in mapped) if mapped else None}


def find_motif_hits(sequence, motif, repeat_type, strand, promoter_start, promoter_end, tss):
    sequence = sequence.upper().replace("U", "T")
    if repeat_type == "linear":
        arm, other, gap = motif, "", 0
    else:
        arm, spacer, other = motif.split("-")
        gap = int(spacer[1:])
    width = len(arm) + gap + len(other)
    reverse_arm = reverse_complement(arm)
    hits = []
    for position in range(len(sequence) - width + 1):
        first = sequence[position:position + len(arm)]
        second = sequence[position + len(arm) + gap:position + width]
        if first == arm and second == other:
            orientation = "both" if repeat_type == "inverted" or arm == reverse_arm else "+"
        elif (repeat_type in ("direct", "linear") and first == reverse_arm
              and (repeat_type == "linear" or second == reverse_arm)):
            orientation = "-"
        else:
            continue
        if strand == 1:
            start, end = promoter_start + position, promoter_start + position + width
            relative_start = start - tss
        else:
            start, end = promoter_end - position - width, promoter_end - position
            relative_start = tss - end
        # Offsets increase along transcription; the annotated 5' gene boundary is a TSS proxy.
        hits.append({"genomic_start": start, "genomic_end": end,
                     "tss_relative_start": relative_start,
                     "tss_relative_end": relative_start + width,
                     "orientation_relative_to_gene": orientation,
                     "sequence_in_gene_orientation": sequence[position:position + width]})
    return hits


def cluster_gene_hits(candidate, genome, result, settings):
    contig, anchor = result["location"]
    indices = genome.contigs[contig]["gene_indices"]
    first = max(0, anchor + candidate["left"] - settings.expansion_genes)
    stop = min(len(indices), anchor + candidate["right"] + settings.expansion_genes + 1)
    rows = []
    for position in range(first, stop):
        index = indices[position]
        gene = genome.genes[index]
        key = genome.promoter_keys[index]
        if candidate["repeat_type"] == "linear":
            lo, hi = genome.linear_promoter_bounds[key]
            sequence = genome.linear_promoter_sequences[key]
        else:
            lo, hi = genome.promoter_bounds[key]
            sequence = genome.promoter_sequences[key]
        hits = find_motif_hits(sequence, candidate["motif"],
                              candidate["repeat_type"], gene["strand"], lo, hi, gene["tss_proxy"])
        relative = position - anchor
        count = min(len(hits), settings.max_sites_per_promoter) if settings.max_sites_per_promoter else len(hits)
        rows.append({**gene, "relative_position": relative,
                     "in_predicted_span": candidate["left"] <= relative <= candidate["right"],
                     "promoter_start": lo, "promoter_end": hi, "has_motif": bool(hits),
                     "total_motif_hits": len(hits), "scored_motif_sites": count, "motif_hits": hits})
    return rows


def prediction_info(candidate, genome, result, reference, settings):
    contig, anchor = result["location"]
    genes = genome.contigs[contig]["gene_indices"]
    left, right = candidate["left"], candidate["right"]
    expanded_left, expanded_right = left - settings.expansion_genes, right + settings.expansion_genes
    available_left = max(-anchor, expanded_left)
    available_right = min(len(genes) - anchor - 1, expanded_right)
    prediction = {"candidate_id": candidate["id"], "mode": candidate["mode"],
                  "repeat_type": candidate["repeat_type"], "motif": candidate["motif"],
                  "used_lower_ranked_motif": not candidate["family_best"],
                  "left": left, "right": right, "predicted_gene_count": right - left + 1,
                  "predicted_genes": [genome.genes[i]["locus"] for i in genes[anchor + left:anchor + right + 1]],
                  "expanded_left": expanded_left, "expanded_right": expanded_right,
                  "expanded_gene_count": expanded_right - expanded_left + 1,
                  "available_expanded_left": available_left, "available_expanded_right": available_right,
                  "available_expanded_gene_count": available_right - available_left + 1,
                  "expanded_genes": [genome.genes[i]["locus"] for i in genes[anchor + available_left:anchor + available_right + 1]],
                  "expansion_clipped_left": available_left != expanded_left,
                  "expansion_clipped_right": available_right != expanded_right,
                  "genes": cluster_gene_hits(candidate, genome, result, settings),
                  "motif_sites_by_gene": result["counts"][:, candidate["motif_index"]].astype(int).tolist(),
                  "evaluation": None}
    # Reference annotations only enter evaluation, after the prediction has been selected.
    if reference and reference["complete"] and reference["genes"]:
        kl, kr = reference["left"], reference["right"]
        overlap = max(0, min(expanded_right, kr) - max(expanded_left, kl) + 1)
        expanded_size = expanded_right - expanded_left + 1
        known = {g["locus"] for g in reference["genes"]}
        tp = len(set(prediction["predicted_genes"]) & known)
        prediction["evaluation"] = {
            "left_error": left - kl, "right_error": right - kr,
            "within_two_genes": max(abs(left - kl), abs(right - kr)) <= 2,
            "exact": left == kl and right == kr,
            "expanded_contains_reference": expanded_left <= kl and expanded_right >= kr,
            "expanded_extra_genes": expanded_size - overlap,
            "expanded_missing_genes": kr - kl + 1 - overlap,
            "expanded_span_precision": overlap / expanded_size,
            "expanded_span_iou": overlap / (expanded_size + kr - kl + 1 - overlap),
            "gene_precision": tp / len(prediction["predicted_genes"]), "gene_recall": tp / len(known)}
    return prediction


def analyse_input(row, genome, calibration, settings):
    if row["input_error"]:
        return error_result(row, row["input_error"])
    location = genome.locate(row["anchor"])
    result = score_anchor(genome, location, settings)
    candidates, decisions = make_candidates(result, calibration, settings)
    qualifying_modes = [d["mode"] for d in decisions if d["significant_single_call"]]
    # Mode diagnostics do not gate global selection; motif-family agreement is unused.
    eligible = bool(qualifying_modes)
    selected = apply_rules(candidates, settings)
    initial = most_significant_candidate(candidates)
    decision_candidate = selected or initial
    reference = reference_info(genome, location, row["known_genes"])
    prediction = prediction_info(selected, genome, result, reference, settings) if selected else None
    if selected:
        if settings.selection_policy == "repeat_first":
            reason = "accepted_repeat_preferred_passing_candidate"
        else:
            reason = "accepted_most_significant_passing_candidate" if settings.retry_rejected else "accepted_most_significant_candidate"
    elif not eligible:
        reason = "no_significant_candidate"
    else:
        reason = "candidate_rejected_by_filters"
    failures = Counter(name for c in candidates for name in c["failed_checks"])
    failed_checks = [] if selected else (decision_candidate["failed_checks"] if decision_candidate
                                        else ["no_calibrated_enriched_candidate"])
    return {**row, "status": "accepted" if selected else "rejected", "reason": reason,
            "decision_candidate_id": decision_candidate["id"] if decision_candidate else None,
            "failed_checks": failed_checks,
            "error": None, "fallback_eligible": eligible, "qualifying_modes": qualifying_modes,
            "mode_decisions": decisions, "reference": reference,
            "contig": genome.contigs[location[0]]["id"],
            "gene_context": [{**genome.genes[index], "relative_position": position - result["center"]}
                             for position, index in enumerate(result["local"])],
            "background_promoters": result["n_background"], "windows_per_motif": result["n_windows"],
            "candidates": candidates, "passing_candidate_ids": [c["id"] for c in candidates if c["passes_checks"]],
            "failed_check_counts": dict(failures), "prediction": prediction}


def error_result(row, message):
    return {**row, "status": "error", "reason": "analysis_failed", "error": message,
            "decision_candidate_id": None, "failed_checks": ["analysis_failed"],
            "fallback_eligible": False, "qualifying_modes": [], "mode_decisions": [],
            "reference": None, "contig": None, "gene_context": [], "background_promoters": None,
            "windows_per_motif": None, "candidates": [], "passing_candidate_ids": [],
            "failed_check_counts": {}, "prediction": None}


def analyse_input_safely(row, genome, calibration, settings):
    """Keep an invalid anchor from aborting the other anchors in the genome."""
    try:
        return analyse_input(row, genome, calibration, settings)
    except (ValueError, OSError) as error:
        return error_result(row, str(error))


def run_genome(path, rows, settings, workers=1):
    print(f"Analysing {Path(path).name}: {len(rows)} input(s)", file=sys.stderr, flush=True)
    try:
        genome = Genome(path, settings)
    except (OSError, ValueError) as error:
        return {"genome_path": path, "error": str(error)}, [error_result(row, str(error)) for row in rows]
    if workers < 1:
        raise ValueError("workers must be positive")
    pool = (ProcessPoolExecutor(max_workers=workers, initializer=_init_anchor_worker,
                                initargs=(genome, settings)) if workers > 1 else nullcontext(None))
    with pool as executor:
        try:
            calibration = calibrate(genome, settings, executor=executor, workers=workers)
        except (OSError, ValueError) as error:
            return {"genome_path": path, "error": str(error)}, [error_result(row, str(error)) for row in rows]
        analyse = (partial(_input_anchor_worker, calibration=calibration) if executor is not None else
                   partial(analyse_input_safely, genome=genome, calibration=calibration, settings=settings))
        outputs = list(ordered_parallel_map(analyse, rows, executor=executor, max_pending=2 * workers))
    return {"genome_path": path, "error": None, "gene_count": len(genome.genes), **calibration}, outputs


def run_analyses(grouped, settings, workers):
    """Process genomes in order, parallelizing anchors within each genome."""
    for path, rows in grouped.items():
        yield run_genome(path, rows, settings, workers=workers)


def read_inputs(path):
    rows, header = [], None
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for line, fields in enumerate(csv.reader(handle, delimiter="\t"), 1):
            if not any(value.strip() for value in fields):
                continue
            if not rows and header is None and fields[0].strip().lower() in ("genome_path", "genome", "genome_file"):
                header = fields
                continue
            genome_path = fields[0].strip()
            anchor = fields[1].strip() if len(fields) > 1 else ""
            known = {}
            for token in fields[2].split(";") if len(fields) > 2 else []:
                if not token.strip():
                    continue
                name, separator, locus = token.strip().partition(":")
                locus = (locus if separator else name).split(",")[0].strip()
                if locus:
                    known[locus] = name.strip() if separator else ""
            resolved = Path(genome_path).expanduser()
            if not resolved.is_absolute() and not resolved.exists():
                resolved = path.parent / resolved
            error = "Missing genome path or anchor" if not genome_path or not anchor else None
            rows.append({"input_index": len(rows), "input_line": line, "input_fields": fields,
                         "genome_path": str(resolved.resolve()), "anchor": anchor,
                         "known_genes": known, "input_error": error})
    return header, rows


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i", "--tsv", dest="tsv_file", type=Path, required=True,
                        help="Same TSV input as v2: genome path, anchor, optional known genes and metadata.")
    parser.add_argument("-o", "--output", type=Path, required=True, help="Destination JSON file.")
    parser.add_argument("--workers", type=int, default=1,
                        help="Anchor worker processes per genome, including random calibration anchors. Genomes run sequentially.")
    parser.add_argument("--candidate-motifs", dest="candidate_modes", nargs="+", choices=MODE_ORDER,
                        default=list(MODE_ORDER), help="Modes to analyse; default runs all three.")
    parser.add_argument("--repeat-types", nargs="+", choices=REPEAT_TYPES, default=list(Settings().repeat_types),
                        help="Motif types to search; linear finds single contiguous sequences and pools reverse complements.")
    retry = parser.add_mutually_exclusive_group()
    retry.add_argument("--retry-rejected", dest="retry_rejected", action="store_true",
                       help="If the original choice fails, try another passing candidate.")
    retry.add_argument("--no-retry-rejected", dest="retry_rejected", action="store_false",
                       help="Do not try another candidate if the original choice fails.")
    parser.set_defaults(retry_rejected=Settings().retry_rejected)
    parser.add_argument("--selection-policy", choices=("most_significant", "repeat_first"), default=Settings().selection_policy,
                        help="repeat_first chooses passing repeats before linear fallback; most_significant ranks all motif types by significance.")
    descriptions = {
        "n_flanking": "Genes examined on each side of the anchor.",
        "upstream": "Promoter bases upstream of the gene start.",
        "downstream": "Promoter bases downstream of the gene start.",
        "part_size": "Bases in each exact repeat arm.",
        "min_spacer": "Minimum bases between repeat arms; does not affect linear motifs.",
        "max_spacer": "Maximum bases between repeat arms.",
        "linear_length": "Bases in each exact linear motif (1-8); used only when linear is selected.",
        "linear_upstream": "Upstream bases for linear promoters only; 0 inherits upstream. Must not exceed upstream.",
        "linear_window_penalty": "Subtract this value per gene when choosing linear windows; raw LLR is calibrated with the same optimization at random anchors.",
        "min_linear_support_fraction": "Minimum fraction of linear predicted promoters containing the motif (0-1). Does not change repeat support filters.",
        "max_sites_per_promoter": "Site count cap per motif and promoter; 0 disables the cap.",
        "min_bgc_size": "Minimum genes in a scored window.",
        "max_bgc_size": "Maximum genes in a scored window, before the heuristic filter.",
        "min_genes_with_site": "Minimum motif-bearing promoters in a scored window.",
        "null_anchors": "Random anchors per genome for empirical significance.",
        "null_seed": "Random seed.", "null_min_flank": "Minimum genes on both sides of a random anchor.",
        "alpha": "Strict empirical p-value cutoff.",
        "edge_flank_genes": "Genes examined on each side of a proposed boundary.",
        "max_predicted_genes": "Maximum unexpanded span accepted by the heuristic.",
        "min_llr_per_gene": "Minimum LLR divided by unexpanded gene count.",
        "expansion_genes": "Genes added at each end after acceptance; clipped at contig ends.",
    }
    for name, description in descriptions.items():
        default = getattr(Settings(), name)
        parser.add_argument("--" + name.replace("_", "-"), type=type(default), default=default, help=description)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    positive = ("workers", "part_size", "linear_length", "min_bgc_size", "max_bgc_size", "min_genes_with_site",
                "null_anchors", "edge_flank_genes", "max_predicted_genes")
    nonnegative = ("n_flanking", "upstream", "downstream", "min_spacer", "max_spacer", "max_sites_per_promoter",
                   "null_min_flank", "expansion_genes", "min_llr_per_gene", "linear_upstream",
                   "linear_window_penalty", "min_linear_support_fraction")
    if any(getattr(args, name) < 1 for name in positive) or any(getattr(args, name) < 0 for name in nonnegative):
        parser.error("Counts must be positive; margins, promoter lengths, cap, spacer and LLR threshold may be zero.")
    if not math.isfinite(args.min_llr_per_gene) or not 0 < args.alpha < 1:
        parser.error("alpha must be between 0 and 1; the LLR threshold must be finite.")
    if (not math.isfinite(args.linear_window_penalty) or not math.isfinite(args.min_linear_support_fraction)
            or args.min_linear_support_fraction > 1 or args.linear_upstream > args.upstream):
        parser.error("Linear penalty must be finite and nonnegative; support fraction must be in [0,1]; linear-upstream cannot exceed upstream.")
    if args.part_size > 6:
        parser.error("part-size above 6 exceeds the supported exhaustive motif search.")
    if args.linear_length > 8:
        parser.error("linear-length above 8 exceeds the supported exhaustive motif search.")
    if args.min_spacer > args.max_spacer:
        parser.error("min-spacer cannot exceed max-spacer.")
    if args.min_bgc_size > args.max_bgc_size or args.min_genes_with_site > args.max_bgc_size:
        parser.error("Minimum window size and site-bearing genes cannot exceed max-bgc-size.")
    if args.n_flanking < args.max_bgc_size - 1:
        parser.error("n-flanking must be at least max-bgc-size minus 1.")
    if args.upstream + args.downstream == 0:
        parser.error("Promoters must contain at least one base.")
    if len(set(args.candidate_modes)) != len(args.candidate_modes) or len(set(args.repeat_types)) != len(args.repeat_types):
        parser.error("Candidate modes and repeat types must not contain duplicates.")
    input_path, output_path = args.tsv_file.resolve(), args.output.resolve()
    if input_path == output_path:
        parser.error("The output must differ from the input file.")
    try:
        header, rows = read_inputs(input_path)
    except (OSError, csv.Error) as error:
        parser.error(str(error))
    if not rows:
        parser.error("Input TSV contains no data rows.")
    if any(output_path == Path(row["genome_path"]) for row in rows):
        parser.error("The output must differ from every genome file.")
    values = {name: getattr(args, name) for name in Settings.__dataclass_fields__}
    values["candidate_modes"] = tuple(mode for mode in MODE_ORDER if mode in args.candidate_modes)
    values["repeat_types"] = tuple(args.repeat_types)
    settings = Settings(**values)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["genome_path"]].append(row)
    with closing(run_analyses(grouped, settings, args.workers)) as results:
        summary = write_report(input_path, output_path, header, results, settings,
                               n_genomes=len(grouped), software_version=VERSION)
    print(f"Wrote {output_path}: {summary['accepted']}/{summary['inputs']} accepted; {summary['error']} errors", file=sys.stderr)
    return 1 if summary["error"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
