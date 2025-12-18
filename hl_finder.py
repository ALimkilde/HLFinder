import sys
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.ndimage import maximum_filter, minimum_filter
from search_picture import get_distance_px_to_m
from numba import njit
from numba.typed import List
from hl_plotter import get_score, hlheight

import numpy as np
from scipy.ndimage import shift
from skimage.measure import block_reduce

from config import MAX_HL_LENGTH, MIN_HL_LENGTH, PX_SIZE_M_SEARCH, PADDING_M, PADDING_PX

def clear_image_border(mask, n):
    # Top border
    mask[:n, :] = False
    # Bottom border
    mask[-n:, :] = False
    # Left border
    mask[:, :n] = False
    # Right border
    mask[:, -n:] = False

    return mask

def downsample(im, filt_size, fun="max"):
    if (fun == "max"):
        return block_reduce(im, filt_size, np.max)
    elif (fun == "min"):
        return block_reduce(im, filt_size, np.min)
    else:
        print(f"error in downsample")
        sys.exit()

def upsample(im, filt_size):
    return np.repeat(np.repeat(im, filt_size[0], axis=0), filt_size[1], axis=1)

def coarse_masks(im_mid, im_anc, lmin, lmax, deltal, scale):

    filt_size = (deltal, deltal)

    Cmin = downsample(im_mid, filt_size, "min")
    Cmax = downsample(im_anc, filt_size, "max")
    
    lmin_coarse = math.floor(lmin/deltal)
    lmax_coarse = math.ceil(lmax/deltal)

    n,m = Cmax.shape
    rts = math.ceil(lmax_coarse)
    rte = n - math.ceil(lmax_coarse)

    cts = math.ceil(lmax_coarse)
    cte = m - math.ceil(lmax_coarse)

    mask = np.full(Cmin.shape, False)

    mask = mask_core(Cmin, Cmax, lmin_coarse, lmax_coarse, mask, rts, rte, cts, cte, deltal*scale)


    return upsample(mask,filt_size)

# @njit
def mask_core(Cmin, Cmax, lmin_coarse, lmax_coarse, mask, rts, rte, cts, cte, scale):
    for R in range(0, lmax_coarse):
        for C in range(-lmax_coarse, lmax_coarse):
            l = math.sqrt(R**2 + C**2)
            if (l < lmin_coarse): continue
            if (l > lmax_coarse): continue
            hgoal = hlheight(l*scale)

            maskRC = np.full(Cmax.shape, False)

            M = np.logical_and( 
                               np.greater(Cmax[rts + R:rte + R, cts + C:cte + C], 
                                            Cmin[rts:rte, cts:cte] + hgoal),
                               np.greater(Cmax[rts - R:rte - R, cts - C:cte - C], 
                                            Cmin[rts:rte, cts:cte] + hgoal)
                               )

            maskRC[rts + R:rte + R, cts + C:cte + C] = M
            mask = np.logical_or(mask,maskRC)
            maskRC[rts - R:rte - R, cts - C:cte - C] = M
            mask = np.logical_or(mask,maskRC)

            # masks = {
            #     "Cmax": Cmax,
            #     "Cmin": Cmin,
            #     "M": M,
            #     "maskRC": maskRC
            #     }
            # plot_masks(masks)
            # sys.exit()

    return mask
            

def grid_masks(im, lmin, lmax, deltal):

    filt_size = (deltal+1, deltal+1)
    M = maximum_filter(im, size=filt_size, mode='constant')

    n,m = im.shape
    rts = math.ceil(lmax)
    rte = n - math.ceil(lmax)

    cts = math.ceil(lmax)
    cte = m - math.ceil(lmax)
    print(f"cts: {cts}")
    print(f"cte: {cte}")

    mask = np.full(im.shape, False)
    kernel = np.ones(filt_size, np.uint8)

    # TDAL: check what happens for lmax not divisible of deltal
    # TDAL: is it correct to add deltal here?
    for R in range(0, lmax-deltal+1, deltal):
        for C in range(-lmax+deltal, lmax-deltal+1, deltal):
            l = math.sqrt(R**2 + C**2)
            if (l < lmin): continue
            if (l > lmax): continue
            hgoal = hlheight(l)

            print(f"{R},{C}: l = {l}, h = {hgoal}")

            K = np.full(im.shape, False) # TDAL: doesn't have to be this big!
            maskRC = np.full(im.shape, False)


            K[rts:rte, cts:cte] = np.greater(M[rts + R:rte + R, cts + C:cte + C], 
                                             im[rts:rte, cts:cte] + hgoal)
            K[rts:rte, cts:cte] = np.logical_and(K[rts:rte, cts:cte],
                                             np.greater(M[rts - R:rte - R, cts - C:cte - C], 
                                             im[rts:rte, cts:cte] + hgoal))


            # maskRC[rts + R:rte + R, cts + C:cte + C] = maximum_filter(K[rts:rte, cts:cte], size=filt_size, mode='constant')



            out = cv2.dilate(
                K[rts:rte, cts:cte].astype(np.uint8),
                kernel,
                borderType=cv2.BORDER_CONSTANT
            )
            
            maskRC[rts + R:rte + R, cts + C:cte + C] = out.astype(bool)
            # maskRC[rts + R:rte + R, cts + C:cte + C] = binary_dilation(
            #         K[rts:rte, cts:cte],
            #         iterations = filt_size[0]-1)

            mask = np.logical_or(maskRC, mask)
            # tmp = maskRC.copy()

            # maskRC[rts - R:rte - R, cts - C:cte - C] = maximum_filter(K[rts:rte, cts:cte], size=filt_size, mode='constant')

            out = cv2.dilate(
                K[rts:rte, cts:cte].astype(np.uint8),
                kernel,
                borderType=cv2.BORDER_CONSTANT
            )
            
            maskRC[rts - R:rte - R, cts - C:cte - C] = out.astype(bool)
            # maskRC[rts - R:rte - R, cts - C:cte - C] = binary_dilation(
            #         K[rts:rte, cts:cte],
            #         iterations = filt_size[0]-1)

            mask = np.logical_or(maskRC, mask)
            # masks = {
            #     "maskRC": K + np.logical_or(maskRC, tmp),
            #     }
            # plot_masks(masks)

    return mask

         

def max_quadrants_1(im, dist):
    """
    Compute directional maximum filters:
      Q1 = up-right
      Q2 = up-left
      Q3 = down-left
      Q4 = down-right

    Returns a tuple: (Q1, Q2, Q3, Q4)
    """

    # --- Q1: Up + Right ---
    # Shift down & left → up-right neighborhood becomes rectangle
    pad = np.pad(im, ((dist, 0), (0, dist)), mode='edge')
    shifted = pad[:-dist, dist:]
    Q1 = maximum_filter(shifted, size=(dist+1, dist+1))

    # --- Q3: Down + Left ---
    # Shift up & right
    pad = np.pad(im, ((0, dist), (dist, 0)), mode='edge')
    shifted = pad[dist:, :-dist]
    Q3 = maximum_filter(shifted, size=(dist+1, dist+1))

    return Q1, Q3

def max_quadrants_2(im, dist):
    """
    Compute directional maximum filters:
      Q1 = up-right
      Q2 = up-left
      Q3 = down-left
      Q4 = down-right

    Returns a tuple: (Q1, Q2, Q3, Q4)
    """

    # --- Q2: Up + Left ---
    # Shift down & right
    pad = np.pad(im, ((dist, 0), (dist, 0)), mode='edge')
    shifted = pad[:-dist, :-dist]
    Q2 = maximum_filter(shifted, size=(dist+1, dist+1), mode='constant')


    # --- Q4: Down + Right ---
    # Shift up & left
    pad = np.pad(im, ((0, dist), (0, dist)), mode='edge')
    shifted = pad[dist:, dist:]
    Q4 = maximum_filter(shifted, size=(dist+1, dist+1), mode='constant')

    return Q2, Q4

def improved_max_masks(im_midpoint, im_anchor):

     num = 7

     lengths = np.linspace(MIN_HL_LENGTH, MAX_HL_LENGTH, num)

     mask = np.full(im_midpoint.shape, False)
     n_remove = math.floor(PADDING_M/PX_SIZE_M_SEARCH)
     # print(f"n_remove: {n_remove}")

     for i,l in enumerate(lengths[:-1]):
         lmin = l
         lmax = lengths[i+1]
         hgoal = hlheight(lmin)
         # print(f"lmin: {lmin}")
         # print(f"lmax: {lmax}")
         # print(f"hgoal: {hgoal}")
         n_extended = math.ceil(lmax/(2*PX_SIZE_M_SEARCH))
         Q1, Q3 = max_quadrants_1(im_anchor, n_extended)
         Q2, Q4 = max_quadrants_2(im_anchor, n_extended)

         extQ1 = np.greater(Q1, im_midpoint+hgoal)
         extQ2 = np.greater(Q2, im_midpoint+hgoal)
         extQ3 = np.greater(Q3, im_midpoint+hgoal)
         extQ4 = np.greater(Q4, im_midpoint+hgoal)

         midmask1 = np.logical_and(extQ1, extQ3)
         midmask2 = np.logical_and(extQ2, extQ4)

         midmask1 = clear_image_border(midmask1, n_remove)
         midmask2 = clear_image_border(midmask2, n_remove)

         mq1, mq3 = max_quadrants_1(midmask1, n_extended)
         mq2, mq4 = max_quadrants_2(midmask2, n_extended)

         mask1 = np.logical_or(mq1, mq3)
         mask2 = np.logical_or(mq2, mq4)

         mask = np.logical_or(mask, mask1)
         mask = np.logical_or(mask, mask2)

         # im_midpoint[0:10,0] = 0
         # im_midpoint[0:10,n_extended] = 0
         # masks = {
         #       "im_midpoint": im_midpoint,
         #     # "extQ1": extQ1,
         #     # "extQ2": extQ2,
         #     # "extQ3": extQ3,
         #     # "extQ4": extQ4,
         #     "midmask1": midmask1,
         #     "midmask2": midmask2,
         #     "mask1": mask1,
         #     "mask2": mask2,
         #     "mask": mask
         # }
         # plot_masks(masks)
         # sys.exit()

     
     # masks = {
     #        "im_midpoint": im_midpoint,
     #        "im_anchor": im_anchor,
     #        "mask final": mask
     # }
     # plot_masks(masks)
     # sys.exit()

     return mask



def get_extended_max_mask(im, H, filter_type):
     n_extended = math.ceil(MAX_HL_LENGTH/(2*PX_SIZE_M_SEARCH))
    
     if (filter_type == 'max'):
         extmax = maximum_filter(im, size=(n_extended, n_extended), mode='nearest')
         mask = np.greater(extmax, im + H)
     if (filter_type == 'min'):
         extmin = minimum_filter(im, size=(n_extended, n_extended), mode='nearest')
         mask = np.greater(im, extmin + H)

     mask = maximum_filter(mask, size=(n_extended, n_extended), mode='nearest')


     return mask

def get_slope_mask(im):
    gx,gy = np.gradient(im, 4)
    tmp = np.sqrt(gx**2 + gy**2)/PX_SIZE_M_SEARCH
    tmp = maximum_filter(tmp, size=(2, 2), mode='nearest')

    acc_slope = 0.01
    return np.greater(tmp, acc_slope)

def plot_masks(masks):

    n = len(masks)
    cols = 2
    rows = (n + 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(10, 10))
    axes = axes.ravel()

    for ax, (name, data) in zip(axes, masks.items()):
        ax.imshow(data)
        ax.set_title(name)
        # ax.axis("off")

    # Hide any unused subplots
    for ax in axes[n:]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()

def get_highline_mask(im_midpoint, im_anchor):
    # extmax_mask = improved_max_masks(im_midpoint, im_anchor)
    extmax_mask = coarse_masks(im_midpoint,
                        im_anchor,
                        math.floor(MIN_HL_LENGTH/(2*PX_SIZE_M_SEARCH)), 
                        math.ceil(MAX_HL_LENGTH/(2*PX_SIZE_M_SEARCH)),
                        math.floor(20/(2*PX_SIZE_M_SEARCH)),
                        PX_SIZE_M_SEARCH)

    slope_mask = get_slope_mask(im_anchor)
    extmax_mask = extmax_mask[:slope_mask.shape[0], :slope_mask.shape[1]]

    mask = np.logical_and(slope_mask, extmax_mask)

    # masks = {
    #     "im_midpoint": im_midpoint,
    #     "slope_mask": slope_mask,
    #     "extmax_mask": extmax_mask,
    #     "mask": mask,
    # }
    # plot_masks(masks)
    # sys.exit()

    return mask

@njit
def search_highline(im, im_min_surf, im_anchor, H, mask):

    result = List()

    nx, ny = im.shape
    
    n_extended = math.ceil(MAX_HL_LENGTH/(PX_SIZE_M_SEARCH))

    for r0 in range(0, nx):
        for c0 in range(0, ny):
            if mask[r0,c0]:
                h0 = im_anchor[r0, c0]
                
                if (h0 < H): 
                    continue
                 
                rmax = min(r0+n_extended, nx)
                cmin = max(0, c0 - n_extended)
                cmax = min(c0+n_extended, ny)

                for r in range(r0, rmax):
                    for c in range(cmin, cmax):
                        if mask[r,c]:
                            l = get_distance_px_to_m(PX_SIZE_M_SEARCH,r0,c0,r,c)
                            if (l<MIN_HL_LENGTH):
                                continue

                            if (l>MAX_HL_LENGTH):
                                continue

                            rm = (r + r0) // 2
                            cm = (c + c0) // 2
                            if (rm < PADDING_PX): continue
                            if (rm > nx - PADDING_PX): continue
                            if (cm < PADDING_PX): continue
                            if (cm > ny - PADDING_PX): continue

                            hgoal = hlheight(l)

                            h = im_anchor[r, c]
                            if (h < hgoal): 
                                continue


                            h_mid = im[rm, cm]
                            h_min = min(h, h0)

                            if(h_min > h_mid + hgoal):
                                score, do_not_hit_tree, hmean_terr, hmean_surf, walkable = get_score(im, im_min_surf, r0, c0, r, c, PX_SIZE_M_SEARCH, h_min, l)
                                
                                if (score>0.4 and do_not_hit_tree):
                                    result.append(( rm, cm, r0, c0, r, c, h_min, l, h_mid, h0, h, hgoal, score, hmean_terr, hmean_surf, walkable))
                 

    return result


if __name__ == "__main__":

    n = 3000
    r = 1500
    c = 1500

    lmin = 100
    deltal = 100
    lmax = 500

    im = np.full((n,n),100)

    s = 1000
    im[r-s:r,c-s//2:c+s//2] = 60

    mask = coarse_masks(im, lmin, lmax, deltal,10)

    masks = {
        "im": im,
        "mask": mask,
    }
    plot_masks(masks)
