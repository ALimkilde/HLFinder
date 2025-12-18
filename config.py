import math

# REGION = "Denmark"
REGION = "Sweden"

# Original picture
if REGION == "Denmark":
    PX_SIZE_M = 0.4
    TILE_SIZE_M = 1000
    STEP_SIZE = 1
    COOR_SIZE = 1000
else:
    PX_SIZE_M = 1
    TILE_SIZE_M = 2500
    STEP_SIZE = 25
    COOR_SIZE = 100


print(f"PX_SIZE_M: {PX_SIZE_M}")
# Search parameters
px_size_m_goal = 5
print(f"px_size_m_goal: {px_size_m_goal}")

MIN_HL_LENGTH = 50
MAX_HL_LENGTH = 500

PADDING_M = MAX_HL_LENGTH/2
PX_SIZE_M_SEARCH = int(px_size_m_goal / PX_SIZE_M) * PX_SIZE_M
PADDING_PX = PADDING_M/PX_SIZE_M_SEARCH
print(f"PX_SIZE_M_SEARCH: {PX_SIZE_M_SEARCH}")

# Tree settings
MAX_TREE_ANCHOR   = 10
MAX_TREE_FRACTION = 0.5
TREE_DIST         = 5
PRE_FILT_SIZE     = 2

# Clustering settings
CLUSTER_RADIUS    = 10
KEEP_METRICS      = ["walkable", "hmean"]

