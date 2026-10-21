"""Predict fungal gene cluster boundaries using shared DNA motifs."""

from __future__ import annotations
from collections import defaultdict
from contextlib import contextmanager
from copy import deepcopy
import csv
import hashlib
import json
import logging
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from typing import Any, Dict, Iterable, Iterator, List, Optional

from Bio import SeqIO

from antismash.common import module_results, path, subprocessing
from antismash.common.errors import AntismashInputError
from antismash.common.html_renderer import FileTemplate, HTMLSections
from antismash.common.layers import OptionsLayer, RecordLayer, RegionLayer
from antismash.common.secmet import SubRegion, FeatureLocation, GeneFunction, Record
from antismash.common.serialiser import feature_to_json, feature_from_json
from antismash.config import ConfigType, update_config
from antismash.config.args import ModuleArgs
from antismash.detection import DetectionStage

NAME = "fungal_boundary"
SHORT_DESCRIPTION = "Detect cluster boundaries using shared motifs"
DETECTION_STAGE = DetectionStage.AREA_REFINEMENT
PREDICTOR_PATH = Path(__file__).resolve().with_name("palindromic_chri_jakob_v4_1.py")


def get_arguments() -> ModuleArgs:
    """Define the enable switch and the predictor options exposed by antiSMASH."""
    args = ModuleArgs('Fungal boundary options', 'fungal', override_safeties=True)
    args.add_analysis_toggle('--fungal_boundary', dest='fungal_boundary', action='store_true', default=False,
                             help="Fungal BGC boundry detection using shared motifs.")
    args.add_option('--fungal-min-spacer', dest='fungal_min_spacer', type=int, default=1,
                    help="Minimum spacer bases between repeat arms (default: %(default)s).")
    args.add_option('--fungal-max-spacer', dest='fungal_max_spacer', type=int, default=15,
                    help="Maximum spacer bases between repeat arms (default: %(default)s).")
    args.add_option('--fungal-repeat-types', dest='fungal_repeat_types', type=str, nargs='+',
                    choices=['direct', 'inverted', 'linear'], default=['direct', 'inverted', 'linear'],
                    help="Motif types to search (default: %(default)s).")
    args.add_option('--fungal-linear-length', dest='fungal_linear_length', type=int, default=6,
                    help="Length of linear motifs, from 1 to 8 bases (default: %(default)s).")
    return args


def is_enabled(options: ConfigType) -> bool:
    """ Is the module enabled on CLI"""
    return options.fungal_boundary


def check_options(options: ConfigType) -> List[str]:
    """ Make sure the options are sane """
    problems = []
    if not is_enabled(options):
        return problems
    if options.taxon != "fungi":
        problems.append("fungal_boundary cluster prediction only works for fungal sequences.")
    if not 0 <= options.fungal_min_spacer <= options.fungal_max_spacer:
        problems.append("Fungal spacer limits must satisfy 0 <= min-spacer <= max-spacer.")
    if not 1 <= options.fungal_linear_length <= 8:
        problems.append("Fungal linear motifs must have between 1 and 8 bases.")
    if len(set(options.fungal_repeat_types)) != len(options.fungal_repeat_types):
        problems.append("Fungal repeat types must not contain duplicates.")
    return problems


def check_prereqs(options: ConfigType) -> List[str]:
    """Check the predictor files and imports by running its --help command."""
    if not options.fungal_boundary:
        return []
    for predictor_file in (PREDICTOR_PATH, PREDICTOR_PATH.with_name("utils.py")):
        if not predictor_file.is_file():
            return [f"Failed to locate fungal_boundary predictor file: {predictor_file}"]
    try:
        check = subprocessing.execute([sys.executable, str(PREDICTOR_PATH), "--help"])
    except OSError as error:
        return [f"Failed to run fungal_boundary predictor: {error}"]
    if not check.successful():
        return [f"fungal_boundary predictor dependencies could not be loaded: {check.stderr.strip()}"]
    return []


@contextmanager
def prepare_genome(records: Iterable[Record], options: ConfigType) -> Iterator[Path]:
    """creates a temporary GenBank file with all records, and updates the config to point to it.
    Export all processed contigs, including skipped ones, for genome-wide background.

    Keep gene features only, so added CDS annotations cannot change the fingerprint.

    """
    previous_path = options.get("fungal_boundary_genome")
    with TemporaryDirectory() as temp_dir:
        genome_path = Path(temp_dir) / "genome.gbk"
        with genome_path.open("w", encoding="utf-8") as handle:
            for record in records:
                bio_record = record.to_biopython()
                bio_record.features = [feature for feature in bio_record.features if feature.type == "gene"]
                SeqIO.write(bio_record, handle, "genbank")
        update_config({"fungal_boundary_genome": str(genome_path)})
        try:
            yield genome_path
        finally:
            update_config({"fungal_boundary_genome": previous_path})


def run_on_record(record: Record, results: Optional[FungalBoundaryResults],
                  options: ConfigType) -> FungalBoundaryResults:
    """Reuse validated results (in the case of ) or run the predictor for this record."""
    if record.is_circular():
        raise AntismashInputError("At this point, fungal_boundary cannot be used on circular genomes")
    return results if results is not None else detect(record, options)


def detect(record: Record, options: ConfigType) -> FungalBoundaryResults:
    """
    Run the unchanged predictor-fundries CLI for this record's core gene anchors."""
    logging.info("Detecting gene cluster regions using the palindromic predictor")
    results = FungalBoundaryResults(record.id, get_backend_signature(options, record))
    anchors = list(dict.fromkeys(get_anchor_gene_names(record)))
    logging.debug("Record has %d anchor genes", len(anchors))
    if not anchors:
        return results
    if not record.get_genes():
        logging.warning("fungal_boundary predictor requires annotated gene features in record %r, skipping", record.id)
        return results

    with TemporaryDirectory() as temp_dir:
        input_path = Path(temp_dir) / "input.tsv"
        output_path = Path(temp_dir) / "output.json"
        with input_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle, delimiter="\t")
            writer.writerow(["genome_path", "anchor"])
            for anchor in anchors:
                writer.writerow([str(_genome_path(options).resolve()), anchor])
        command = [sys.executable, str(PREDICTOR_PATH), "-i", str(input_path),
                   "-o", str(output_path), *_predictor_arguments(options)]
        logging.info("Running palindromic predictor with %d worker(s)", options.cpus)
        try:
            execution = subprocessing.execute(command)
        except OSError as error:
            raise AntismashInputError(f"Failed to run fungal_boundary predictor: {error}") from error
        if execution.return_code not in (0, 1) or not output_path.is_file():
            raise AntismashInputError(
                f"fungal_boundary predictor failed (exit {execution.return_code}): {execution.stderr.strip()}"
            )
        try:
            with output_path.open(encoding="utf-8") as handle:
                report = json.load(handle)
        except (OSError, ValueError) as error:
            raise AntismashInputError(f"Cannot read fungal_boundary predictor JSON: {error}") from error
    results.subregions = predictions_to_subregions(report, record, anchors)
    if execution.return_code == 1 and not any(entry["status"] == "error" for entry in report["inputs"]):
        raise AntismashInputError(
            f"fungal_boundary predictor failed without an error entry: {execution.stderr.strip()}"
        )
    results.motifs = get_motif_summaries(report)
    return results


def get_anchor_gene_names(record: Record) -> List[str]:
    """Find core/anchor CDS names"""
    return [feature.get_name() for feature in record.get_cds_features()
            if feature.gene_function == GeneFunction.CORE]


def _predictor_arguments(options: ConfigType) -> List[str]:
    """Pass the CPU budget and motif options to prediction and calibration."""
    return ["--workers", str(options.cpus),
            "--min-spacer", str(options.fungal_min_spacer),
            "--max-spacer", str(options.fungal_max_spacer),
            "--linear-length", str(options.fungal_linear_length),
            "--repeat-types", *options.fungal_repeat_types]


def _genome_path(options: ConfigType) -> Path:
    genome_path = options.get("fungal_boundary_genome")
    if not genome_path:
        raise AntismashInputError("fungal_boundary requires a prepared whole-genome input -> call prepare_genome first")
    return Path(genome_path)


def predictions_to_subregions(report: Dict[str, Any], record: Record,
                              anchors: Iterable[str]) -> List[SubRegion]:
    """Validate the anchor batch and convert accepted expanded gene lists to SubRegions."""
    if not isinstance(report, dict) or report.get("schema_version") != "1.0":
        raise AntismashInputError("Unsupported fungal_boundary predictor JSON schema")
    entries = report.get("inputs")
    if not isinstance(entries, list):
        raise AntismashInputError("fungal_boundary predictor JSON is missing its inputs list")
    expected = set(anchors)
    seen = set()
    subregions = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise AntismashInputError("Invalid fungal_boundary predictor input entry")
        anchor = entry.get("anchor")
        if not isinstance(anchor, str) or anchor not in expected or anchor in seen:
            raise AntismashInputError(f"Unexpected or duplicate fungal_boundary predictor anchor: {anchor!r}")
        seen.add(anchor)
        status = entry.get("status")
        if status == "error":
            logging.warning("fungal_boundary predictor skipped anchor %r: %s",
                            anchor, entry.get("error") or entry.get("reason"))
            continue
        if status == "rejected":
            logging.debug("fungal_boundary predictor rejected anchor %r: %s", anchor, entry.get("reason"))
            continue
        if status != "accepted":
            raise AntismashInputError(f"Invalid fungal_boundary predictor status for anchor {anchor!r}: {status!r}")
        subregions.append(_prediction_to_subregion(entry, record, anchor))
    if seen != expected:
        raise AntismashInputError(f"fungal_boundary predictor omitted anchors: {', '.join(sorted(expected - seen))}")
    return subregions


def _prediction_to_subregion(entry: Dict[str, Any], record: Record, anchor: str) -> SubRegion:
    """Validate an accepted prediction's genes and build its expanded subregion."""
    if entry.get("contig") != record.id:
        raise AntismashInputError(f"fungal_boundary predictor contig does not match record for anchor {anchor!r}")
    prediction = entry.get("prediction")
    genes = prediction.get("genes") if isinstance(prediction, dict) else None
    if not isinstance(genes, list) or not genes:
        raise AntismashInputError(f"fungal_boundary predictor has no predicted genes for anchor {anchor!r}")
    bounds = []
    gene_names = set()
    for gene in genes:
        if not isinstance(gene, dict):
            raise AntismashInputError(f"Invalid predicted gene for anchor {anchor!r}")
        name, start, end = gene.get("locus"), gene.get("start"), gene.get("end")
        if (not isinstance(name, str) or not isinstance(start, int) or isinstance(start, bool)
                or not isinstance(end, int) or isinstance(end, bool)
                or not 0 <= start < end <= len(record)):
            raise AntismashInputError(f"Invalid predicted gene coordinates for anchor {anchor!r}")
        if not any(int(existing.location.start) == start and int(existing.location.end) == end
                   for existing in record.get_genes_by_name(name)):
            raise AntismashInputError(f"Predicted gene {name!r} does not match record {record.id!r}")
        gene_names.add(name)
        bounds.append((start, end))
    if anchor not in gene_names:
        raise AntismashInputError(f"fungal_boundary predicted region does not contain anchor {anchor!r}")
    start = min(bound[0] for bound in bounds)
    end = max(bound[1] for bound in bounds)
    return SubRegion(FeatureLocation(start, end), tool=NAME, label=anchor)


def get_motif_summaries(report: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Keep the selected motif, its empirical p-value, and uncapped promoter counts."""
    summaries = {}
    for entry in report["inputs"]:
        if entry["status"] != "accepted":
            continue
        prediction = entry["prediction"]
        selected = next((candidate for candidate in entry.get("candidates", [])
                         if candidate["id"] == prediction.get("candidate_id")), {})
        summaries[entry["anchor"]] = {
            "repeat_type": prediction.get("repeat_type", ""),
            "motif": prediction.get("motif", ""),
            "p_empirical": selected.get("p_empirical"),
            "genes": [{"locus": gene["locus"], "total_motif_hits": gene["total_motif_hits"]}
                      for gene in prediction["genes"]],
        }
    return summaries


class FungalBoundaryResults(module_results.DetectionResults):
    """ Contains the subregions predicted by fungal_boundary """
    def __init__(self, record_id: str, backend_signature: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(record_id)
        self.subregions: List[SubRegion] = []
        self.motifs: Dict[str, Dict[str, Any]] = {}
        self.backend_signature = deepcopy(backend_signature)

    def to_json(self) -> Dict[str, Any]:
        return {
            "record_id": self.record_id,
            "subregions": [feature_to_json(sub.to_biopython()[0]) for sub in self.subregions],
            "promoters": [],
            "backend_signature": deepcopy(self.backend_signature),
            "motifs": deepcopy(self.motifs),
        }

    @staticmethod
    def from_json(data: Dict[str, Any], record: Record,
                  backend_signature: Optional[Dict[str, Any]] = None) -> Optional["FungalBoundaryResults"]:
        if (data.get("record_id") != record.id or not backend_signature
                or data.get("backend_signature") != backend_signature):
            return None
        results = FungalBoundaryResults(record.id, backend_signature)
        results.subregions = [SubRegion.from_biopython(feature_from_json(sub)) for sub in data["subregions"]]
        results.motifs = deepcopy(data.get("motifs", {}))
        # Include zero counts when restoring saved promoter tables.
        genes = record.get_genes()
        for subregion in results.subregions:
            summary = results.motifs.get(subregion.label)
            if not summary:
                continue
            contained = [gene for gene in genes if gene.is_contained_by(subregion)]
            if not contained:
                continue
            counts = {gene["locus"]: gene["total_motif_hits"] for gene in summary["genes"]}
            summary["genes"] = [{"locus": gene.get_name(), "total_motif_hits": counts.get(gene.get_name(), 0)}
                                for gene in contained]
        return results

    def get_predicted_subregions(self) -> List[SubRegion]:
        subregions = []
        for group in self.get_subregion_groups():
            if len(group) == 1:
                subregions.append(group[0])
            else:
                subregions.append(SubRegion(group[0].location, tool=group[0].tool,
                                            label=", ".join(sub.label for sub in group)))
        return subregions

    def get_subregion_groups(self) -> List[List[SubRegion]]:
        """Group matching boundaries and motifs while retaining each anchor's results."""
        groups: Dict[tuple[str, ...], List[SubRegion]] = {}
        for subregion in filter_subregions(self.subregions):
            summary = self.motifs.get(subregion.label, {})
            motif = summary.get("motif", "")
            # Without motif details, keep different anchors separate.
            key = (str(subregion.location), subregion.tool, summary.get("repeat_type", ""),
                   motif, "" if motif else subregion.label)
            groups.setdefault(key, []).append(subregion)
        return list(groups.values())


def filter_subregions(subregions: List[SubRegion]) -> List[SubRegion]:
    """Remove subregions contained by another prediction for the same anchor."""
    by_anchor: Dict[str, List[SubRegion]] = defaultdict(list)
    for sub in sorted(subregions, key=lambda x: x.location.end - x.location.start, reverse=True):
        if not any(sub.is_contained_by(other) for other in by_anchor[sub.label]):
            by_anchor[sub.label].append(sub)
    flattened = [sub for group in by_anchor.values() for sub in group]
    return sorted(flattened, key=lambda x: (x.location.start, x.location.end, x.label))


def regenerate_previous_results(previous: Dict[str, Any], record: Record,
                                options: ConfigType) -> Optional[FungalBoundaryResults]:
    """Reuse current saved results - older formats require a fresh analysis."""
    if not previous.get("backend_signature") or "motifs" not in previous:
        return None
    return FungalBoundaryResults.from_json(previous, record, get_backend_signature(options, record))


def get_backend_signature(options: ConfigType, record: Optional[Record] = None) -> Dict[str, Any]:
    """Identify the predictor files, arguments, whole-genome input, and anchors for reuse."""
    genome_path = _genome_path(options)
    if not genome_path.is_file():
        raise AntismashInputError("fungal_boundary requires a prepared whole-genome input -> call prepare_genome first")
    try:
        signature = {
            "backend": "palindromic_chri_jakob_v4_1",
            "adapter_version": 1,
            "predictor_sha256": _file_sha256(PREDICTOR_PATH),
            "utils_sha256": _file_sha256(PREDICTOR_PATH.with_name("utils.py")),
            "parameters": {"cli_arguments": _predictor_arguments(options), "use_cli_defaults": True},
            "boundary_policy": "expanded",
            "genome_sha256": _file_sha256(genome_path),
        }
        if record is not None:
            signature["anchors"] = sorted(set(get_anchor_gene_names(record)))
        return signature
    except OSError as error:
        raise AntismashInputError(f"Cannot read fungal_boundary predictor or genome input: {error}") from error


def _file_sha256(file_path: Path) -> str:
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def generate_html(region_layer: RegionLayer, results: FungalBoundaryResults,
                  _record_layer: RecordLayer, options_layer: OptionsLayer) -> HTMLSections:
    """Generate the fungal_boundary details tab through the standard module interface."""
    html = HTMLSections("fungal_boundary")
    if not options_layer.fungal_boundary:
        return html
    locations = [subregion.location for subregion in region_layer.subregions if subregion.tool == NAME]
    predictions = []
    for group in results.get_subregion_groups():
        if group[0].tool != NAME or group[0].location not in locations:
            continue
        anchors = [(subregion.label, results.motifs.get(subregion.label)) for subregion in group]
        predictions.append((anchors, results.motifs.get(group[0].label)))
    tooltip = (
        "Predictions with the same motif type, DNA motif, and boundaries share one bar and table. "
        "Empirical p-values are shown for each anchor. "
        "The table lists every gene within the displayed boundary and its total promoter matches, including zeros. "
        "N followed by a number denotes the number of spacer bases between repeat arms."
    )
    template = FileTemplate(path.get_full_path(__file__, "templates", "fungal_boundary.html"))
    content = template.render(predictions=predictions, tooltip=tooltip)
    html.add_detail_section("fungal_boundary", content, "fungal_boundary-details")
    return html


def will_handle(_products: List[str], _product_categories: set[str]) -> bool:
    """fungal_boundary details can apply to any region, regardless of its product type."""
    return True
