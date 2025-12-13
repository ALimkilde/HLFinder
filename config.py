import math

# Original picture
PX_SIZE_M = 0.4

# Search parameters
px_size_m_goal = 5

MIN_HL_LENGTH = 100
MAX_HL_LENGTH = 400

PADDING_M = MAX_HL_LENGTH/2
PX_SIZE_M_SEARCH = int(px_size_m_goal / PX_SIZE_M) * PX_SIZE_M
print(f"PX_SIZE_M_SEARCH: {PX_SIZE_M_SEARCH}")
print(f"PADDING_M: {PADDING_M}")

# Tree settings
MAX_TREE_ANCHOR   = 5
MAX_TREE_FRACTION = 0.33
TREE_DIST         = 10
PRE_FILT_SIZE     = 2

# Clustering settings
CLUSTER_RADIUS    = 10
KEEP_METRICS      = ["walkable"]

