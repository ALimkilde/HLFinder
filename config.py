import math

# REGION = "Denmark"
REGION = "Denmark"

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
px_size_m_goal = 4.5
print(f"px_size_m_goal: {px_size_m_goal}")

MIN_HL_LENGTH = 80
MAX_HL_LENGTH = 300

PADDING_M = MAX_HL_LENGTH/2
PX_SIZE_M_SEARCH = int(px_size_m_goal / PX_SIZE_M) * PX_SIZE_M
PADDING_PX = PADDING_M/PX_SIZE_M_SEARCH
print(f"PX_SIZE_M_SEARCH: {PX_SIZE_M_SEARCH}")

# Tree settings
MAX_TREE_ANCHOR   = 15
MAX_TREE_FRACTION = 0.6
TREE_DIST         = 1
PRE_FILT_SIZE     = 2

# Clustering settings
CLUSTER_RADIUS    = 25
KEEP_METRICS      = ["walkable","rigging_height","hmean"]

