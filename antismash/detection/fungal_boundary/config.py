# License: GNU Affero General Public License v3 or later
# A copy of GNU AGPL v3 should have been included in this software
# package in LICENSE.txt.

"""Parameters and motif ID ranges for fungal boundary analysis."""

class FungalBoundariesConfig:
    """Settings and derived motif ID ranges for fungal boundary analysis."""

    MIN_GENES_PER_CONTIG = 1000     # minimum CDS features for boundary analysis
    MIN_GENES_WITH_MOTIF = 3        # minimum genes containing a motif for enrichment
    MAX_GENES_IN_BGC = 30          # maximum genes in a scored window
    UPSTREAM_BASES = 1000          # bases upstream of the CDS's 5' boundary
    DOWNSTREAM_BASES = 50          # bases downstream of the CDS's 5' boundary
    LINEAR_UPSTREAM_BASES = 300    # upstream bases used for linear motif searches

    BASES_PER_ARM = 3
    MAX_SPACER = 15                # max bases allowed between the two arms
    LINEAR_LENGTH = 6              # bases in a linear motif
    MAX_MOTIFS_PER_PROMOTER = 3
    MIN_RANDOM_ANCHORS = 10
    MAX_RANDOM_ANCHOR_OVERLAP = 28
    RANDOM_SEED = 1
    MIN_LLR = 20                   # minimum LLR required for empirical p-value testing
    P_VAL_MAX = 0.05               # empirical p-value a prediction must beat

    @property
    def SPACERS(self) -> int:
        """The number of allowed spacer BPs"""
        return self.MAX_SPACER + 1

    @property
    def MOTIF_IDS_PER_REPEAT_TYPE(self) -> int:
        """The number of motif IDs reserved for each repeat type."""
        return 4 ** self.BASES_PER_ARM * self.SPACERS

    @property
    def DIRECT_MOTIF_ID_START(self) -> int:
        """The first direct-repeat motif ID."""
        return self.MOTIF_IDS_PER_REPEAT_TYPE

    @property
    def LINEAR_MOTIF_ID_START(self) -> int:
        """The first linear motif ID."""
        return 2 * self.MOTIF_IDS_PER_REPEAT_TYPE

    @property
    def N_MOTIFS(self) -> int:
        """The total number of motif IDs across all motif types."""
        return self.LINEAR_MOTIF_ID_START + 4 ** self.LINEAR_LENGTH

    @property
    def MOTIF_TYPE_ID_BOUNDARIES(self) -> tuple[int, int]:
        """The first direct-repeat and linear IDs, ordered by motif type."""
        return self.DIRECT_MOTIF_ID_START, self.LINEAR_MOTIF_ID_START


config = FungalBoundariesConfig()
