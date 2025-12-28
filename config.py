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
MAX_TREE_ANCHOR   = 15       # Maximum allowed anchor height in tree
MAX_TREE_FRACTION = 0.6      # Maximium fraction of tree height to put anchor 
                             # (0.6 is allowing 6m high anchors in a 10m tree)
TREE_DIST         = 1        # How many meters where we allow other trees being "in the way"
                             # (1: means that it is okey that there are trees in the way 1m infront of anchors. Hopefully we can the find a gap below the tree tops)
PRE_FILT_SIZE     = 2        

# Clustering settings
CLUSTER_RADIUS    = 25       # Only save the best line in this radius in meters
KEEP_METRICS      = ["walkable","rigging_height","hmean"] # What the best line means:
                                                          #       "height": keep highest line
                                                          #       "hmean": keep line that has the highest mean height when walking/falling
                                                          #       "walkable": keep the line with the longest walkable length
                                                          #       "score": keep the line with the best score (the fraction of the line that is walkable)
                                                          #       "rigging_height": keep the line with the lowest rigging height

