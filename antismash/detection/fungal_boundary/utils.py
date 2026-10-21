import hashlib
import json
import os
import sys
import tempfile
from collections import Counter, deque
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from functools import lru_cache
import numpy as np



# Section with things related to DNA and motif helpers

BASES = "ACGT"
BASE_CODES = {"A": 0, "C": 1, "G": 2, "T": 3}


@lru_cache(maxsize=None)
def reverse_codes(k):
    """Build and cache a reverse-complement lookup for all k-base sequences."""
    reverse_complement_codes = np.zeros(4 ** k, dtype=int)
    base_to_digit = str.maketrans("".join(BASES), "0123")

    for code in range(4 ** k):
        sequence = decode_code(code, k)
        complemented_sequence = reverse_complement(sequence)
        # Convert DNA letters back to a number using A=0, C=1, G=2, T=3.
        base_digits = complemented_sequence.translate(base_to_digit)
        reverse_complement_codes[code] = int(base_digits, base=4) if base_digits else 0

    return reverse_complement_codes


@lru_cache(maxsize=128)
def motif_ranges(settings):
    """Assign numeric ID ranges to motif types - each stop is exclusive."""
    arm_sequence_count = 4 ** settings.part_size
    spacer_length_count = settings.max_spacer + 1  # Includes a spacer of zero.
    ids_per_repeat_type = arm_sequence_count * spacer_length_count

    # Keep the established order: inverted repeats, direct repeats, then linear.
    inverted_start = 0
    inverted_stop = inverted_start + ids_per_repeat_type
    direct_start = inverted_stop
    direct_stop = direct_start + ids_per_repeat_type

    ranges = (
        ("inverted", inverted_start, inverted_stop),
        ("direct", direct_start, direct_stop),
    )

    if "linear" in settings.repeat_types:
        linear_sequence_count = 4 ** settings.linear_length
        linear_start = direct_stop
        linear_stop = linear_start + linear_sequence_count
        ranges += (("linear", linear_start, linear_stop),)

    return ranges


def reverse_complement(sequence):
    return sequence.translate(str.maketrans("ACGT", "TGCA"))[::-1]


def decode_code(code, length):
    """Convert a numeric DNA code into a sequence of the requested length."""
    if length <= 0:
        return ""

    # Base-4 digits represent DNA bases: 0=A, 1=C, 2=G, 3=T.
    base_digits = np.base_repr(code, base=4).zfill(length)[-length:]
    digit_to_base = str.maketrans("0123", "".join(BASES))
    return base_digits.translate(digit_to_base)


@lru_cache(maxsize=128)
def motif_indices(family, settings):
    """the numeric IDs of the motifs the script should test for a given type.
    """
    start, stop = next((start, stop) for name, start, stop in motif_ranges(settings) if name == family)
    codes = np.arange(stop - start)
    if family == "linear":
        # Test one representative of each sequence/reverse-complement pair.
        codes = codes[codes <= reverse_codes(settings.linear_length)]
    else:
        codes = codes[codes % (settings.max_spacer + 1) >= settings.min_spacer]
    return start + codes


def motif_name(index, settings):
    """numeric motif ID into its type and readable DNA pattern."""
    for family, start, stop in motif_ranges(settings):
        if start <= index < stop:
            if family == "linear":
                return family, decode_code(index - start, settings.linear_length)
            code, spacer = divmod(index - start, settings.max_spacer + 1)
            arm = decode_code(code, settings.part_size)
            other = arm if family == "direct" else reverse_complement(arm)
            return family, f"{arm}-N{spacer}-{other}"
    raise ValueError(f"Unknown motif index: {index}")


def encode_kmers(bases, k):
    """Encode all overlapping k-mers and flag those containing ambiguous bases."""
    width = max(0, len(bases) - k + 1)
    codes = np.zeros(width, dtype=np.int64)
    if not width:
        return codes, np.zeros(0, dtype=bool)
    for position in range(k):
        codes = codes * 4 + np.minimum(bases[position:position + width], 3)
    invalid = np.concatenate(([0], np.cumsum(bases == 4)))
    return codes, invalid[k:] == invalid[:-k]


def count_sites(sequence, settings):
    k, spacers = settings.part_size, settings.max_spacer + 1
    bases = np.array([BASE_CODES.get(base.upper(), 4) for base in sequence], dtype=np.uint8)
    empty = np.empty(0, dtype=np.int64)
    found = []
    if len(bases) >= 2 * k and any(t in settings.repeat_types for t in ("inverted", "direct")):
        codes, valid = encode_kmers(bases, k)
        reverse = reverse_codes(k)
        for spacer in range(settings.min_spacer, spacers):
            length = len(bases) - 2 * k - spacer + 1
            if length <= 0:
                break
            left = codes[:length]
            right = codes[k + spacer:k + spacer + length]
            usable = valid[:length] & valid[k + spacer:k + spacer + length]
            if "inverted" in settings.repeat_types:
                found.append(left[usable & (left == reverse[right])] * spacers + spacer)
            if "direct" in settings.repeat_types:
                # Merge a direct repeat and its reverse complement into one motif.
                arms = np.minimum(left, reverse[left])
                found.append(4 ** k * spacers + arms[usable & (left == right)] * spacers + spacer)
    if "linear" in settings.repeat_types and len(bases) >= settings.linear_length:
        codes, valid = encode_kmers(bases, settings.linear_length)
        codes = codes[valid]
        canonical = np.minimum(codes, reverse_codes(settings.linear_length)[codes])
        ranges = {
            family: (start, stop)
            for family, start, stop in motif_ranges(settings)
        }
        linear_start, _ = ranges["linear"]
        found.append(linear_start + canonical)
    if not found:
        return empty, empty
    indices, counts = np.unique(np.concatenate(found), return_counts=True)
    if settings.max_sites_per_promoter:
        counts = np.minimum(counts, settings.max_sites_per_promoter)
    return indices, counts





# Section with functions related to analysis

def ordered_parallel_map(function, items, *, executor=None, max_pending=1):
    """Map in input order, limiting queued work and cancelling unused tasks."""
    if executor is None:
        yield from map(function, items)
        return
    if max_pending < 1:
        raise ValueError("max_pending must be positive")
    items = iter(items)
    pending = deque()
    try:
        for _ in range(max_pending):
            try:
                item = next(items)
            except StopIteration:
                break
            pending.append(executor.submit(function, item))
        while pending:
            yield pending.popleft().result()
            try:
                item = next(items)
            except StopIteration:
                continue
            pending.append(executor.submit(function, item))
    finally:
        for future in pending:
            future.cancel()


@lru_cache(maxsize=None)
def candidate_windows(n_genes, anchor, minimum, maximum):
    """lists every consecutive gene window that contains the anchor and satisfies the minimum and maximum sizes.
    to avoid contigs with too few genes."""
    windows = [(left, right) for left in range(max(0, anchor - maximum + 1), anchor + 1)
               for right in range(anchor, min(n_genes, left + maximum))
               if right - left + 1 >= minimum]
    if not windows:
        raise ValueError("Too few genes on this contig for the minimum BGC size")
    return np.array(windows, dtype=int).T


def edge_evidence(values, left, right, flank):
    """Compare mean motif counts just inside and outside both window boundaries."""
    width = min(flank, right - left + 1)
    outside_left = values[max(0, left - width):left]
    outside_right = values[right + 1:right + 1 + width]
    delta_left = float(values[left:left + width].mean() - outside_left.mean()) if len(outside_left) else None
    delta_right = float(values[right - width + 1:right + 1].mean() - outside_right.mean()) if len(outside_right) else None
    return {"edge_left_delta": delta_left, "edge_right_delta": delta_right,
            "edge_supported": (delta_left is not None and delta_right is not None
                               and min(delta_left, delta_right) >= 0 and max(delta_left, delta_right) > 0)}


# Section with functions related to reporting

RULE_DESCRIPTIONS = {
    "enriched": "LLR > 0.",
    "enough_genes_with_site": "At least min_genes_with_site promoters contain the motif.",
    "calibrated": "At least one successful random anchor provides an empirical p-value.",
    "significant": "Empirical p < alpha (strict inequality).",
    "span_within_limit": "Unexpanded predicted_gene_count <= max_predicted_genes.",
    "llr_per_gene_at_least_minimum": "LLR / unexpanded predicted_gene_count >= min_llr_per_gene.",
    "linear_support_fraction": "For linear motifs, genes_with_site / predicted_gene_count >= min_linear_support_fraction. Repeat motifs retain their support filters.",
}


def write_report(input_path, output_path, header, results, settings, *, n_genomes, software_version):
    """Write genome results to a complete JSON report with metadata."""
    metadata = {"schema_version": "1.0", "software_version": software_version,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "input_file": str(input_path), "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
                "input_columns": header, "parameters": asdict(settings),
                "score_scope": "Window maximizing raw LLR minus linear_window_penalty * gene count for linear motifs; raw LLR for repeats. All tested motifs are reported. Raw LLR is calibrated at null anchors using the identical window-selection procedure.",
                "calibration_scope": "Empirical p compares each motif LLR with the maximum LLR per motif type and mode at random anchors. No combined p-value across modes/types is claimed.",
                "motif_matching": "Exact repeat arms or exact contiguous linear sequences. Direct and linear motifs pool reverse complements; linear_length is independent of repeat part_size and max_spacer.",
                "selection_policy": settings.selection_policy,
                "selection_order": ("Prefer passing repeat candidates, otherwise passing linear candidates; within each group choose lowest empirical p, then highest LLR, mode, type and index. No agreement gate."
                                    if settings.selection_policy == "repeat_first" else
                                    "Lowest empirical p across all tested motifs, families and modes, then filters. Ties: highest LLR; anchor, adjacent, all; motif type; motif index. With retry_rejected, choose the most significant passing candidate. No family agreement requirement."),
                "linear_promoters": "linear_upstream=0 inherits upstream. A positive value changes linear promoters and their genomic background only; repeat promoters retain upstream. Reported linear motif hits use the same cropped promoters as scoring.",
                "legacy_fields": {"fallback_eligible": "Diagnostic alias: at least one mode has a significant candidate; not an acceptance filter."},
                "rule_descriptions": RULE_DESCRIPTIONS,
                "coordinate_conventions": {"gene_positions": "Relative to anchor = 0, inclusive endpoints",
                                           "genomic_positions": "Zero-based start, exclusive end",
                                           "expanded_window": "Nominal expansion matches the benchmark; available_expanded_* and gene lists are clipped to actual contig genes.",
                                           "tss_reference": "Annotated gene 5-prime boundary (start on + strand, end on - strand), not an experimentally measured TSS.",
                                           "motif_tss_positions": "Start inclusive, end exclusive, oriented along transcription; negative is upstream. Includes both repeat arms and spacer, or the complete linear motif."}}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter(inputs=0, accepted=0, rejected=0, error=0, fallback_eligible=0,
                     candidates=0, passing_candidates=0, lower_ranked_predictions=0)
    accepted, calibrations = [], []
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output_path.parent,
                                         prefix=f".{output_path.name}.", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(json.dumps(metadata, allow_nan=False, indent=2)[:-2] + ',\n  "inputs": [\n')
            first = True

            for calibration, outputs in results:
                calibration_index = len(calibrations)
                calibrations.append(calibration)
                for row in outputs:
                    row["calibration_index"] = calibration_index
                    handle.write(("" if first else ",\n") + json.dumps(row, allow_nan=False, separators=(",", ":")))
                    first = False
                    counts["inputs"] += 1
                    counts[row["status"]] += 1
                    counts["fallback_eligible"] += row["fallback_eligible"]
                    counts["candidates"] += len(row["candidates"])
                    counts["passing_candidates"] += len(row["passing_candidate_ids"])
                    if row["prediction"]:
                        accepted.append(row["input_index"])
                        counts["lower_ranked_predictions"] += row["prediction"]["used_lower_ranked_motif"]
                print(f"Completed {len(calibrations)}/{n_genomes} genomes; {counts['accepted']} accepted", file=sys.stderr, flush=True)

            handle.write('\n  ],\n  "calibrations": ')
            json.dump(calibrations, handle, allow_nan=False, separators=(",", ":"))
            handle.write(',\n  "accepted_input_indices": ' + json.dumps(sorted(accepted)))
            handle.write(',\n  "summary": ' + json.dumps(dict(counts)) + "\n}\n")
            handle.flush()
            os.fsync(handle.fileno())
        # Publish only a complete JSON document, preserving any previous report on failure.
        os.replace(temporary, output_path)
    finally:
        if temporary and temporary.exists():
            temporary.unlink()
    return dict(counts)
