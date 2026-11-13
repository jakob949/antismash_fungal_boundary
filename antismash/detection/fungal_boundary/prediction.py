# License: GNU Affero General Public License v3 or later
# A copy of GNU AGPL v3 should have been included in this software package in
# LICENSE.txt.

""" Predict fungal BGC boundaries from promoter motif enrichment around core genes.

    Candidate boundaries must contain a core gene and a motif
    enriched relative to other motifs of its type on the same contig.
    Empirical significance ("pvalue") is assessed against non-core genes outside the
    candidate windows (core gene neighbourhoob).
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import asdict, dataclass
from enum import IntEnum
from typing import Any, Self

import numpy as np

from antismash.common.module_results import DetectionResults
from antismash.common.secmet import CDSFeature, FeatureLocation, GeneFunction, Record, SubRegion

from . import utils
from .config import config


@dataclass
class GeneWithMotifs:  # pylint: disable=too-many-instance-attributes
    """ One gene's basic annotations like promoter sequences and motif counts
    """

    name: str
    start: int
    end: int
    strand: int
    promoter: str
    promoter_to_cds_start_dist: int
    linear_promoter: str
    linear_promoter_to_cds_start_dist: int
    motif_ids: np.ndarray
    is_anchor: bool
    counts: np.ndarray # counts are cap to a max count within a single promoter
    uncapped_counts: np.ndarray
    is_random_anchor: bool = False

    @classmethod
    def from_feature(cls, record: Record, cds: CDSFeature) -> GeneWithMotifs:
        """ Extracts the CDS’s promoters, identifies motifs, and counts occurrences
            of each motif.

            Repeat and linear motif promoters extend up to config.UPSTREAM_BASES
            and config.LINEAR_UPSTREAM_BASES bases upstream of the CDS's 5' boundary,
            respectively. Both extend up to config.DOWNSTREAM_BASES bases downstream,
            stop at contig boundaries, and follow the gene's reading direction.
            Only CDS features assigned GeneFunction.CORE are anchors.

            Arguments:
                record: the record containing the CDS and DNA sequence of the contig
                cds: the CDS feature whose promoters are to be extracted

            Returns:
                the gene's annotations
        """
        cds_start_coord = int(cds.location.start)
        cds_end_coord = int(cds.location.end)
        strand = cds.location.strand

        if strand == -1:
            promoter_window_start = max(0, cds_end_coord - config.DOWNSTREAM_BASES)
            promoter_window_end = min(len(record.seq), cds_end_coord +
                                      config.UPSTREAM_BASES)
            promoter_seq = str(
                record.seq[promoter_window_start:promoter_window_end].
                reverse_complement()
            )
            promoter_to_cds_start_dist = promoter_window_end - cds_end_coord
        else:
            promoter_window_start = max(0, cds_start_coord - config.UPSTREAM_BASES)
            promoter_window_end = min(len(record.seq), cds_start_coord +
                                      config.DOWNSTREAM_BASES)
            promoter_seq = str(
                record.seq[promoter_window_start:promoter_window_end])
            promoter_to_cds_start_dist = (cds_start_coord
                                          - promoter_window_start)
        crop = max(0, promoter_to_cds_start_dist - config.LINEAR_UPSTREAM_BASES)
        linear_promoter = promoter_seq[crop:]
        motif_counts = count_motifs(promoter_seq, linear_promoter,
                                    capped_max_count=None)

        return cls(
            name=cds.get_name(),
            start=cds_start_coord,
            end=cds_end_coord,
            strand=strand,
            promoter=promoter_seq,
            promoter_to_cds_start_dist=promoter_to_cds_start_dist,
            linear_promoter=linear_promoter,
            linear_promoter_to_cds_start_dist=(
                promoter_to_cds_start_dist - crop),
            motif_ids=motif_counts.motif_ids,
            is_anchor=cds.gene_function == GeneFunction.CORE,
            counts=np.minimum(motif_counts.counts, config.MAX_MOTIFS_PER_PROMOTER),
            uncapped_counts=motif_counts.counts,
        )

    @staticmethod
    def select_random_acting_anchors(
            genes: dict[str, GeneWithMotifs],
            enriched_motifs: dict[str, EnrichedMotif | None],
            max_genes_in_bgc: int = config.MAX_GENES_IN_BGC,
            *, max_random_anchor_overlap: int = config.MAX_RANDOM_ANCHOR_OVERLAP,
            random_seed: int = config.RANDOM_SEED) -> None:
        """ Marks a spaced, random sample of genes to act as "fake" anchors for the
            calculation of p_empirical.

            "Fake" anchors candidates come from _get_eligible_anchor_positions and
            are spaced so that no two of their windows overlap by more than
            max_random_anchor_overlap genes.
            "Fake" anchors are set with is_random_anchor.

            Arguments:
                genes: the contig's genes annotations
                enriched_motifs: each anchor's result from
                                 find_enriched_motifs,
                                 whose windows are excluded from the candidates
                max_genes_in_bgc: the most genes a window may span
                max_random_anchor_overlap: the most genes two control windows may share
                random_seed: the random seed used for reproduciblity

            Returns:
                None - becouse the selection is recorded on the (GeneWithMotifs instance) of genes themselves
        """
        gene_list = list(genes.values())
        for gene in gene_list:
            gene.is_random_anchor = False

        # A window reaches this far either side of its anchor, so two windows
        # sharing at most max_random_anchor_overlap genes must have their
        # anchors at least this far apart.
        reach = max_genes_in_bgc - 1
        # max_random_anchor_overlap and reach should maybe be the same?
        spacing = max(1, 2 * reach + 1 - max_random_anchor_overlap)
        candidates = _get_eligible_anchor_positions(gene_list, enriched_motifs, reach)

        chosen = _sample_spaced_positions(candidates, spacing,
                                          np.random.default_rng(random_seed))
        for position in chosen:
            gene_list[int(position)].is_random_anchor = True


class SelectionMode(IntEnum):
    """ Defines the modes - or where a motif must occur relative to the anchor gene.

        ANCHOR requires that the motif is in the anchor's promoter.
        ADJACENT allows that a motif is found in immediate neighbour genes promoter.
        ALL allows a motif hit anywhere in the candidate window.
    """

    ANCHOR = 0
    ADJACENT = 1
    ALL = 2


@dataclass(slots=True)
class EnrichedMotif:
    """ Finding the most enriched motif and its gene window around an anchor.

        first_gene_offset gives the window's start relative to the anchor:
        negative values count genes before it, positive values count genes after it,
        and zero means the window starts at the anchor.
        gene_count is the number of genes in the window.
        mode indicates the motif's proximity to the anchor (0, 1 , 2 in SelectionMode).
        p_empirical is None when no empirical p-value has been calculated.
    """
    anchor_id: str
    motif_id: int
    llr: float
    mode: SelectionMode
    first_gene_offset: int
    gene_count: int
    p_empirical: float | None = None

    def calculate_p_value(  # pylint: disable=too-many-arguments
            self, genes: dict[str, GeneWithMotifs], background_frequencies: dict[int, float],
            sample_size: int, random_scores: dict[tuple[int, SelectionMode], np.ndarray], *,
            min_llr: float = config.MIN_LLR,
            min_random_anchors: int = config.MIN_RANDOM_ANCHORS,
            minimum_num_of_genes: int = config.MIN_GENES_WITH_MOTIF,
            max_genes_in_bgc: int = config.MAX_GENES_IN_BGC,
            motif_type_id_boundaries: tuple[int, int] = config.MOTIF_TYPE_ID_BOUNDARIES) -> None:
        """ Calculates this motif's empirical p-value against random acting anchors.

            Leaves p_empirical as None when the LLR or random-anchor count is
            below the required minimum.

            Arguments:
                genes: the contig's genes annotations
                background_frequencies: each motif ID mapped to its background
                                        frequency within its own motif type
                sample_size: the number of selected random acting anchors
                random_scores: cached random-anchor LLR scores for this contig,
                               keyed by motif ID and selection mode
                min_llr: the minimum log-likelihood ratio required for testing
                min_random_anchors: the fewest random acting anchors required for testing
                minimum_num_of_genes: the fewest genes that must contain a rescored motif
                max_genes_in_bgc: the most genes a window may span
                motif_type_id_boundaries: the first direct-repeat and linear motif IDs,
                                          in that order
        """
        self.p_empirical = None
        if self.llr < min_llr or sample_size < min_random_anchors:
            return

        key = (self.motif_id, self.mode)
        if key not in random_scores:
            random_results = find_enriched_motifs(
                genes, background_frequencies,
                minimum_num_of_genes=minimum_num_of_genes,
                max_genes_in_bgc=max_genes_in_bgc, enriched_motif=self,
                motif_type_id_boundaries=motif_type_id_boundaries,
            )
            random_scores[key] = np.array([
                0.0 if random_result is None else random_result.llr
                for random_result in random_results.values()
            ])
        self.p_empirical = float(
            (1 + np.count_nonzero(random_scores[key] >= self.llr))
            / (sample_size + 1)
        )


@dataclass
class BoundaryPrediction:
    """ An accepted boundary and its supporting motif evidence.

        motif_counts maps gene names to uncapped/true motif counts in CDS order.
    """
    anchor_id: str
    start: int
    end: int
    motif_type: str
    motif_string_dna: str
    motif_counts: dict[str, int]
    p_empirical: float

    def to_json(self) -> dict[str, Any]:
        """Serialise a BoundaryPrediction"""
        return asdict(self)

    @classmethod
    def from_json(cls, json: dict[str, Any]) -> Self:
        return cls(
            anchor_id=json["anchor_id"],
            start=json["start"],
            end=json["end"],
            motif_type=json["motif_type"],
            motif_string_dna=json["motif_string_dna"],
            p_empirical=json["p_empirical"],
            motif_counts=dict(json["motif_counts"]),
        )


@dataclass
class BoundaryDetectionResults(DetectionResults):
    """ Accepted boundaries and their supporting motif evidence for one contig """

    schema_version = 1
    contig: str
    predictions: list[BoundaryPrediction]

    def __init__(self, contig: str, predictions: list[BoundaryPrediction]) -> None:
        super().__init__(contig)
        self.contig = contig
        self.predictions = predictions

    def to_json(self) -> dict[str, Any]:
        """Serialize the report and its schema version"""
        return {
            "schema_version": self.schema_version,
            "record_id": self.record_id,
            "report": {
                "contig": self.contig,
                "predictions": [p.to_json() for p in self.predictions],
            },
        }

    @staticmethod
    def from_json(json: dict[str, Any], record: Record) -> BoundaryDetectionResults | None:
        """Reconstruct boundary detection results for the original record."""
        if json.get("schema_version") != BoundaryDetectionResults.schema_version:
            return None
        if json.get("record_id") != record.id:
            return None
        report_data = json["report"]
        predictions = [
            BoundaryPrediction.from_json(data) for data in report_data.get("predictions", [])
        ]
        return BoundaryDetectionResults(
            contig=report_data["contig"],
            predictions=predictions,
        )

    def get_predicted_subregions(self) -> list[SubRegion]:
        """Create subregions for the predicted boundaries."""
        return [
            SubRegion(FeatureLocation(prediction.start, prediction.end),
                      tool="boundary_fungal", label=prediction.anchor_id)
            for prediction in self.predictions
        ]


def count_motifs(promoter: str, linear_promoter: str,
                 capped_max_count: int | None = config.MAX_MOTIFS_PER_PROMOTER
                 ) -> utils.MotifCounts:
    """Counts the repeat and linear motifs in a gene's two promoter.

        Repeat motifs are counted over the full promoter and linear motifs
        over the shorter linear promoter, then the two sets are concatenated.
        The returned motif IDs are in increasing order.

        Arguments:
            promoter: the full promoter sequence, used for repeat motifs
            linear_promoter: the shorter promoter used for linear motifs
            capped_max_count: the maximum count kept for each motif in
                              either promoter, or None to keep uncapped counts

        Returns:
            a MotifCounts instance covering both kinds of motif
    """
    repeat_counts = utils.count_repeat_motifs(
        promoter, capped_max_count=capped_max_count)
    linear_counts = utils.count_linear_motifs(
        linear_promoter, capped_max_count=capped_max_count)

    # Repeat IDs precede linear IDs, preserving sorted order here.
    return utils.MotifCounts(
        motif_ids=np.concatenate((repeat_counts.motif_ids,
                                  linear_counts.motif_ids)),
        counts=np.concatenate((repeat_counts.counts, linear_counts.counts)),
    )


def get_motif_id_to_bg_freq(
        genes: dict[str, GeneWithMotifs], *,
        motif_type_id_boundaries: tuple[int, int] = config.MOTIF_TYPE_ID_BOUNDARIES,
) -> dict[int, float]:
    """ Returns each motif ID mapped to its frequency within its own type

        Counts are summed over every gene, using the capped counts, and
        each motif's total is then divided by the total for its motif type.
        The frequencies of each observed type therefore sum to 1, so a motif
        can only be enriched relative to other motifs of the same kind.

        Arguments:
            genes: the contig's genes in CDS order, keyed by gene name
            motif_type_id_boundaries: the first direct-repeat and linear motif IDs,
                                      in that order

        Returns:
            each observed motif ID mapped to its background frequency
    """
    background_freqs: dict[int, float] = {}
    occurrences_per_motif_type = {
        motif_type: 0.0 for motif_type in utils.MotifType
    }

    # Sum counts for each motif and each motif type across all genes.
    for gene in genes.values():
        for motif_id, count in zip(gene.motif_ids, gene.counts):
            motif_id = int(motif_id)
            count = float(count)
            motif_type = utils.get_motif_type(
                motif_id, motif_type_id_boundaries=motif_type_id_boundaries)
            background_freqs[motif_id] = (
                background_freqs.get(motif_id, 0.0) + count)
            occurrences_per_motif_type[motif_type] += count

    # Divide each motif's count by the total count for its type.
    for motif_id in background_freqs:
        motif_type = utils.get_motif_type(
            motif_id, motif_type_id_boundaries=motif_type_id_boundaries)
        background_freqs[motif_id] /= occurrences_per_motif_type[motif_type]

    return background_freqs


@dataclass
class _WindowCandidates:
    """ Motif evidence for candidate windows sharing a starting gene. """

    first: int
    stops: np.ndarray
    observed: np.ndarray
    expected: np.ndarray
    eligible: np.ndarray
    in_anchor: np.ndarray
    in_adjacent: np.ndarray

    def select_best_motif_candidates(self, motif_ids: np.ndarray, anchor_id: str,
                                     anchor_index: int, enriched_motif: EnrichedMotif | None
                                     ) -> Iterator[EnrichedMotif]:
        """ Yields every eligible motif and window tied for the highest LLR.

            Scores eligible candidates using the log-likelihood ratio
            of observed to expected motif counts. 

            Arguments:
                motif_ids: motif IDs 
                anchor_id: the name of the anchor gene
                anchor_index: the anchor's index within its gene neighbourhood
                enriched_motif: 

            Returns:
                an iterator of EnrichedMotif instances tied for the highest LLR
                among these candidates, empty if no candidates are eligible
        """
        rows, columns = np.nonzero(self.eligible)
        if columns.size == 0:
            return
        hits = self.observed[rows, columns]
        expected_hits = self.expected[rows, columns]
        scores = hits * np.log(hits / expected_hits) - (hits - expected_hits)
        highest_llr = float(scores.max())
        for winner in np.flatnonzero(scores == highest_llr):
            row, column = rows[winner], columns[winner]
            if enriched_motif is not None:
                mode = enriched_motif.mode
            elif self.in_anchor[column]:
                mode = SelectionMode.ANCHOR
            elif self.in_adjacent[row, column]:
                mode = SelectionMode.ADJACENT
            else:
                mode = SelectionMode.ALL
            yield EnrichedMotif(
                anchor_id=anchor_id,
                motif_id=int(motif_ids[column]),
                llr=highest_llr,
                mode=mode,
                first_gene_offset=self.first - anchor_index,
                gene_count=int(self.stops[row]) - self.first,
            )


@dataclass
class _WindowMotifCounts:
    """ Cumulative motif counts for the genes surrounding one anchor.

        Each cumulative array starts with a zero row, so subtracting the
        row at a window's start from the row at its stop gives its total.
    """

    motif_ids: np.ndarray
    motif_types: np.ndarray
    background_frequencies: np.ndarray
    occurrences: np.ndarray
    genes_with_motif: np.ndarray
    type_occurrences: np.ndarray
    anchor_index: int

    @classmethod
    def from_genes(  # pylint: disable=too-many-locals
            cls, genes: list[GeneWithMotifs], anchor_index: int,
            motif_ids: np.ndarray, background_frequencies: dict[int, float], *,
            single_motif: bool, minimum_num_of_genes: int,
            motif_type_id_boundaries: tuple[int, int] = config.MOTIF_TYPE_ID_BOUNDARIES,
    ) -> _WindowMotifCounts | None:
        """ Prepares cumulative motif evidence for scoring windows around an anchor.

            Computes running totals (cumulative) across genes so that any candidate window's
            counts can be calculated by subtracting its start from its stop.

            Arguments:
                genes: genes in the anchor's neighbourhood
                anchor_index: the position of the anchor within genes
                motif_ids: complete ordered range of motif IDs, or a 1D array
                           containing a single ID for a single-motif search
                background_frequencies: each motif ID mapped to its contig-wide
                                        frequency within its motif type
                single_motif: whether to prepare counts for one motif being
                              rescored instead of all motifs
                minimum_num_of_genes: fewest genes that must contain the motif
                                      for a single-motif search to proceed
                motif_type_id_boundaries: the first direct-repeat and linear
                                          motif IDs, in that order

            Returns:
                a _WindowMotifCounts instance with cumulative evidence for
                candidate window scoring, or None if a single motif search
                has too few genes containing the motif
        """
        num_motif_types = len(utils.MotifType)
        # initialize windows length x num_motifs
        # which stores the count data, for the genes in the current window
        # that is gene_0 has 2 occurrences of motif_id 10
        counts = np.zeros((len(genes), len(motif_ids)), dtype=np.int64)
        # Store counts for each gene and motif type.
        type_counts = np.zeros((len(genes), num_motif_types), dtype=np.int64)
        # Loop through each gene object and its zero-based position within
        # the window.
        for index, gene in enumerate(genes):
            if single_motif:
                counts[index, 0] = gene.counts[gene.motif_ids == motif_ids[0]].sum()
            else:
                counts[index, gene.motif_ids] = gene.counts
            # To find out what kind of motif_type (inverted, direct, or linear) the motif_ids is
            gene_types = np.searchsorted(motif_type_id_boundaries,
                                         gene.motif_ids, side="right")
            # adds up motif-id occurrences by motif-type for the current gene
            # np.bincount: Count number of occurrences of each value in array
            type_counts[index] = np.bincount(
                gene_types, weights=gene.counts, minlength=num_motif_types)

        if single_motif and np.count_nonzero(counts) < minimum_num_of_genes:
            return None
        # initialize a 2d matrix: (number of genes in the window +1) x motif IDs
        # +1 extra row is used for cumulative sums later
        occurrences = np.zeros((len(genes) + 1, len(motif_ids)), dtype=np.int64)
        genes_with_motif = np.zeros_like(occurrences)
        # Store cumulative motif occurrence counts by type:
        # inverted, direct, linear.
        type_occurrences = np.zeros((len(genes) + 1, num_motif_types), dtype=np.int64)
        # Calculate running totals for each motif across genes.
        # Example showing cumulative counts for three motifs across three genes:
        # counts       cumulative_occurrences_per_motif
        #              [0, 0, 0]  <- no genes counted yet
        # [2, 0, 1]    [2, 0, 1]  <- gene 0
        # [1, 3, 0]    [3, 3, 1]  <- genes 0 and 1
        # [0, 1, 2]    [3, 4, 3]  <- all three genes
        # Here, occurrences is the cumulative_occurrences_per_motif column in the example.
        np.cumsum(counts, axis=0, out=occurrences[1:])
        np.cumsum(counts > 0, axis=0, out=genes_with_motif[1:])
        np.cumsum(type_counts, axis=0, out=type_occurrences[1:])
        return cls(
            motif_ids=motif_ids,
            # To find out what kind of motif_type (inverted, direct, or linear)
            # the motif_ids is
            motif_types=np.searchsorted(motif_type_id_boundaries, motif_ids, side="right"),
            background_frequencies=np.array([
                background_frequencies.get(motif_id, 0.0) for motif_id in motif_ids
            ], dtype=float),
            occurrences=occurrences,
            genes_with_motif=genes_with_motif,
            type_occurrences=type_occurrences,
            anchor_index=anchor_index,
        )

    def calculate_candidate_window_motif_counts(
            self, first: int, minimum_num_of_genes: int,
            max_genes_in_bgc: int, required_mode: SelectionMode) -> _WindowCandidates:
        """ Scores the counts and eligibility of windows with the given start.

            Arguments:
                first: index of the starting gene in the candidate window
                        genes supplied to from_genes
                minimum_num_of_genes: fewest genes whose promoters must contain
                                     the motif for it to be eligible
                max_genes_in_bgc: most genes a candidate window may span
                required_mode: where the motif must occur relative to the anchor
        """
        # first_stop and last_stop defines the earliest and latest allowed index for window
        # These bounds are calculated directly in np.arange below.
        stops = np.arange(
            max(self.anchor_index + 1, first + minimum_num_of_genes),
            min(len(self.occurrences) - 1, first + max_genes_in_bgc) + 1,
        )
        # the idea is to look at the
        # Total inside a candidate window = running total at its end − running total before its start.
        observed = self.occurrences[stops] - self.occurrences[first]
        genes_with_motif = self.genes_with_motif[stops] - self.genes_with_motif[first]
        type_totals = self.type_occurrences[stops] - self.type_occurrences[first]
        # Use each motif's contig-wide frequency within its type as the baseline,
        # so an abundance of that motif type alone does not imply enrichment.
        expected = type_totals[:, self.motif_types] * self.background_frequencies
        in_anchor = (self.genes_with_motif[self.anchor_index + 1]
                     - self.genes_with_motif[self.anchor_index]) > 0
        nearby_stops = np.minimum(stops, self.anchor_index + 2)
        in_adjacent = (self.genes_with_motif[nearby_stops]
                       - self.genes_with_motif[max(first, self.anchor_index - 1)]) > 0
        eligible = (genes_with_motif >= minimum_num_of_genes) & (observed > expected)
        if required_mode == SelectionMode.ANCHOR:
            eligible &= in_anchor
        elif required_mode == SelectionMode.ADJACENT:
            eligible &= in_adjacent
        return _WindowCandidates(first, stops, observed, expected, eligible, in_anchor, in_adjacent)


def _rank_motif_candidate(result: EnrichedMotif, *,
                          linear_motif_id_start: int = config.LINEAR_MOTIF_ID_START
                          ) -> tuple[float, bool, SelectionMode, int, int]:
    """ Ranks by LLR, repeat preference, proximity, window size, and motif ID.

        linear_motif_id_start is the first ID reserved for linear motifs.
    """
    # Compare priorities if different candidatas achieve
    # the same score.
    return (
        -result.llr,                                    # Higher LLR first.
        result.motif_id >= linear_motif_id_start,        # Repeats win when LLR scores tie.
        result.mode,                                    # Anchor, then adjacent, then all.
        result.gene_count,                              # Fewer genes in the window first.
        result.motif_id,                                # Lower motif ID breaks the final tie.
    )


def _score_anchor(window: _WindowMotifCounts, anchor_id: str,
                  minimum_num_of_genes: int, max_genes_in_bgc: int,
                  enriched_motif: EnrichedMotif | None, *,
                  linear_motif_id_start: int = config.LINEAR_MOTIF_ID_START
                  ) -> EnrichedMotif | None:
    """ Selects the best motif window containing the anchor, preserving ties.

        linear_motif_id_start is the first ID reserved for linear motifs.
    """
    # Search all promoters before classifying the winning motif's mode.
    required_mode = SelectionMode.ALL if enriched_motif is None else enriched_motif.mode
    best = None
    best_rank = None
    # Loop which tries different starting genes for candidate
    # windows containing the anchor gen
    # in most cases (not right next to contig end), will the range be from 0 to 31
    for first in range(max(0, window.anchor_index - max_genes_in_bgc + 1), window.anchor_index + 1):
        candidates = window.calculate_candidate_window_motif_counts(
            first, minimum_num_of_genes, max_genes_in_bgc, required_mode)
        for candidate in candidates.select_best_motif_candidates(
                window.motif_ids, anchor_id, window.anchor_index, enriched_motif):
            candidate_rank = _rank_motif_candidate(
                candidate, linear_motif_id_start=linear_motif_id_start)
            # Keep the first candidate when all ranking values are equal.
            if best_rank is None or candidate_rank < best_rank:
                best_rank = candidate_rank
                best = candidate
    return best


def _meets_motif_proximity_requirement(gene_list: list[GeneWithMotifs], anchor_index: int,
                                       enriched_motif: EnrichedMotif) -> bool:
    """ Checks the required proximity before rescoring a random anchor. """
    if enriched_motif.mode == SelectionMode.ALL:
        return True
    if enriched_motif.mode == SelectionMode.ANCHOR:
        # The motif must occur in the anchor itself.
        genes_to_check = [gene_list[anchor_index]]
    else:
        # The motif may occur in the anchor or either adjacent gene.
        genes_to_check = gene_list[max(0, anchor_index - 1):anchor_index + 2]
    # at least one gene in genes_to_check contains the target motif
    return any(enriched_motif.motif_id in gene.motif_ids for gene in genes_to_check)


def find_enriched_motifs(  # pylint: disable=too-many-locals
        genes: dict[str, GeneWithMotifs], background_frequencies: dict[int, float],
        minimum_num_of_genes: int = config.MIN_GENES_WITH_MOTIF,
        max_genes_in_bgc: int = config.MAX_GENES_IN_BGC,
        enriched_motif: EnrichedMotif | None = None,
        *, num_motifs: int = config.N_MOTIFS,
        motif_type_id_boundaries: tuple[int, int] = config.MOTIF_TYPE_ID_BOUNDARIES,
) -> dict[str, EnrichedMotif | None]:
    """ Finds the most enriched motif in a window of genes around each anchor

        Runs in one of two modes. With enriched_motif left as None, every
        motif is scored at every gene antiSMASH called core gene. Given an
        enriched_motif, only that motif is scored, only at the genes marked
        as random acting anchors, and only in its own mode, which builds the
        null distribution for an earlier result.

        Windows are ranked by a Poisson log-likelihood ratio of observed
        against expected counts, the expectation being the motif's background
        frequency times the total count of its motif type in the window. To be
        eligible a motif must occur in at least minimum_num_of_genes genes of
        the window and exceed its expected count. Repeat motifs are preferred
        over linear ones only when their LLR scores are equal. Remaining ties
        are resolved by mode, then the fewest genes in the window, then the
        lowest motif ID.

        Arguments:
            genes: the contig's genes in CDS order, keyed by gene name
            background_frequencies: each motif ID mapped to its background
                                    frequency within its own motif type
            minimum_num_of_genes: the fewest genes of a window a motif must
                                 occur in to be considered
            max_genes_in_bgc: the most genes a window may span, and the
                              furthest it may reach from its anchor
            enriched_motif: an earlier result to rescore for a null
                            distribution, or None to search all motifs
                            at the real anchors
            num_motifs: the number of reserved motif IDs
            motif_type_id_boundaries: the first direct-repeat and linear motif IDs,
                                      in that order

        Returns:
            each acting anchor's name mapped to its best EnrichedMotif, or to
            None where no motif was eligible
    """
    if enriched_motif is None:
        # Search all possible motif IDs at the real anchors.
        # len(motif_ids) >= 1
        motif_ids: np.ndarray = np.arange(num_motifs)
    else:
        # Find the most enriched window for the supplied motif.
        # len(motif_ids) == 1
        motif_ids = np.array([enriched_motif.motif_id], dtype=int)

    gene_list = list(genes.values())
    results: dict[str, EnrichedMotif | None] = {}
    for anchor_index, anchor in enumerate(gene_list):
        # treat "is_random_anchor" and "is_anchor" as acting_anchor
        acting_anchor = (anchor.is_anchor if enriched_motif is None
                         else anchor.is_random_anchor)
        if not acting_anchor:
            continue
        results[anchor.name] = None
        if enriched_motif is not None and not _meets_motif_proximity_requirement(
                gene_list, anchor_index, enriched_motif):
            continue
        # making sure that the window is inside the contig/gene_list
        first_index = max(0, anchor_index - max_genes_in_bgc)
        stop_index = min(len(gene_list), anchor_index + max_genes_in_bgc + 1)
        window = _WindowMotifCounts.from_genes(
            gene_list[first_index:stop_index], anchor_index - first_index,
            motif_ids, background_frequencies,
            single_motif=enriched_motif is not None,
            minimum_num_of_genes=minimum_num_of_genes,
            motif_type_id_boundaries=motif_type_id_boundaries,
        )
        if window is not None:
            results[anchor.name] = _score_anchor(
                window, anchor.name, minimum_num_of_genes, max_genes_in_bgc, enriched_motif,
                linear_motif_id_start=motif_type_id_boundaries[1],
            )
    return results


def _get_eligible_anchor_positions(gene_list: list[GeneWithMotifs],
                                   enriched_motifs: dict[str, EnrichedMotif | None],
                                   reach: int) -> np.ndarray:
    """ Identifies non-core controls whose spans avoid candidate cluster windows.

        A gene is eligible when antiSMASH did not call it a core gene, it has
        max_genes_in_bgc genes either side of it within the contig, and no
        gene in that span is already covered by one of the windows in
        enriched_motifs.

        Arguments:
            gene_list: the contig's genes in CDS order
            enriched_motifs: candidate results whose windows exclude control
                             genes, regardless of score or p-value
            reach: the required number of neighbouring genes on each side
                   that must lie within the contig and outside candidate windows

        Returns:
            an array of positions into gene_list, in increasing order
    """
    gene_indices = {gene.name: index for index, gene in enumerate(gene_list)}

    # Core biosynthetic genes act as anchors for candidate boundary searches.
    # Random acting anchors are non-core controls for empirical p-values.
    # Their windows must avoid candidate cluster windows, so expanding each
    # candidate window by reach excludes the corresponding positions.
    is_blocked = np.array([gene.is_anchor for gene in gene_list], dtype=bool)
    for result in enriched_motifs.values():
        if result is None:
            continue
        # first_gene_offset counts from the anchor itself and is negative when
        # the window opens before it, so adding it gives an absolute position.
        window_start = (gene_indices[result.anchor_id] +
                        result.first_gene_offset)
        window_end = window_start + result.gene_count
        # max() because a negative slice bound would wrap to the contig's end
        is_blocked[max(0, window_start - reach):window_end + reach] = True

    # Only a gene with a full window's worth of neighbours on either side can
    # act as an anchor, which rules out the first and last few on the contig.
    positions = np.arange(reach, len(gene_list) - reach)
    return positions[~is_blocked[positions]]


def _sample_spaced_positions(candidates: np.ndarray, spacing: int,
                             rng: np.random.Generator
                             ) -> np.ndarray:
    """ Draws a random subset of candidates that are spaced apart

        Candidates are visited in a random order and kept when they clear
        every candidate kept so far, so the spacing always holds.

        The result is a maximal selection rather than the largest possible
        one: an early pick can block two later ones that would have fitted
        together, so a luckier order would sometimes fit more.

        Arguments:
            candidates: the sorted positions to choose from
            spacing: the fewest positions two chosen candidates may differ by
            rng: the generator to draw from

        Returns:
            an array of the chosen positions, in increasing order
    """
    # Basically chose different anchors that are separated in `spacing´
    # so we are selecting valuable anchors and not too close ones.
    chosen: list[int] = []
    for index in rng.permutation(len(candidates)):
        position = int(candidates[index])
        if all(abs(position - taken) >= spacing for taken in chosen):
            chosen.append(position)
    return np.array(sorted(chosen), dtype=np.int64)


def build_report(contig: str, genes: dict[str, GeneWithMotifs],
                 enriched_motifs: dict[str, EnrichedMotif | None], *,
                 p_val_max: float = config.P_VAL_MAX
                 ) -> BoundaryDetectionResults:
    """ Collects the accepted results into a report for one contig.

        Only results with an empirical p-value below p_val_max are
        reported. Each accepted window is located from its anchor's
        position plus the result's first_gene_offset, spanning gene_count
        genes.

        Arguments:
            contig: the record ID the report covers
            genes: the contig's genes in CDS order, keyed by gene name
            enriched_motifs: each anchor's result
            p_val_max: the exclusive upper limit for reported empirical p-values

        Returns:
            the accepted boundary predictions
    """
    gene_list = list(genes.values())
    gene_indices = {gene.name: index for index, gene in enumerate(gene_list)}
    predictions: list[BoundaryPrediction] = []
    for motif in enriched_motifs.values():
        if (motif is None or motif.p_empirical is None
                or motif.p_empirical >= p_val_max):
            continue

        first_index = gene_indices[motif.anchor_id] + motif.first_gene_offset
        window_genes = gene_list[first_index:first_index + motif.gene_count]
        motif_type = utils.get_motif_type(motif.motif_id)
        motif_counts = {
            gene.name: int(gene.uncapped_counts[
                gene.motif_ids == motif.motif_id].sum())
            for gene in window_genes
        }

        predictions.append(BoundaryPrediction(
            anchor_id=motif.anchor_id,
            start=min(gene.start for gene in window_genes),
            end=max(gene.end for gene in window_genes),
            motif_type=motif_type.value,
            motif_string_dna=utils.convert_motif_id_to_text(motif.motif_id),
            motif_counts=motif_counts,
            p_empirical=motif.p_empirical,
        ))
    return BoundaryDetectionResults(contig=contig, predictions=predictions)


def predict_boundaries(record: Record) -> BoundaryDetectionResults:
    """ Predict fungal BGC boundaries on a contig.

        Each CDS becomes a GeneWithMotifs, then every gene antiSMASH
        called a core is scored for the most enriched motif. A sample
        of other genes is used to re-score the same motif to build an
        empirical p-value. Only the motifs beating config.P_VAL_MAX
        reach the report.

        Arguments:
            record: the record whose CDS features will be scored

        Returns:
            BoundaryDetectionResults containing the accepted boundary predictions.
    """
    cds_features = record.get_cds_features()
    if len(cds_features) < config.MIN_GENES_PER_CONTIG:
        return BoundaryDetectionResults(contig=record.id, predictions=[])

    annotated_genes = {}
    for cds in cds_features:
        gene = GeneWithMotifs.from_feature(record, cds)
        annotated_genes[gene.name] = gene

    motif_id_to_bg_freq = get_motif_id_to_bg_freq(annotated_genes)
    enriched_motifs = find_enriched_motifs(
        genes=annotated_genes,
        background_frequencies=motif_id_to_bg_freq,
    )

    GeneWithMotifs.select_random_acting_anchors(annotated_genes, enriched_motifs)
    sample_size = sum(gene.is_random_anchor for gene in annotated_genes.values())

    random_scores: dict[tuple[int, SelectionMode], np.ndarray] = {}
    for result in enriched_motifs.values():
        if result is not None:
            result.calculate_p_value(annotated_genes, motif_id_to_bg_freq, sample_size, random_scores)

    return build_report(record.id, annotated_genes, enriched_motifs)
