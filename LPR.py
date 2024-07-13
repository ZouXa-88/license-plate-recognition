import os
import math
import shutil
import matplotlib.pyplot as plt
import matplotlib.lines as lines
import matplotlib.patches as mpatches
import numpy as np
from itertools import chain

from skimage import io
from skimage import color
from skimage import exposure
from skimage import measure
from skimage import filters
from skimage import morphology

# Global constants (parameters)
MAX_AREA = 300
MIN_AREA = 100
MAX_RATIO = 0.8
MIN_RATIO = 0.2
DIS_TOR = 4
MAX_ORIENT = 3
MAX_R = 170
OUTLIER_TOR = 110

INPUT_DIR = "./input/"
OUTPUT_DIR = "./410985049/"

# Global variables
width = 0
height = 0


def filterRegionsByShape(regions) -> list:
    returnRegions = []
    
    for region in regions:
        minr, minc, maxr, maxc = region.bbox
        ratio = (maxc - minc) / (maxr - minr)
        if (region.area > MIN_AREA and region.area < MAX_AREA) and (ratio > MIN_RATIO and ratio < MAX_RATIO):
            returnRegions.append(region)

    return returnRegions


def filterRegionsByLine(regions, showLine=False) -> list:
    returnRegions = []
    center_x, center_y = width / 2, height / 2
    optimal_radian, optimal_r = 0, 0
    
    for degree in chain(range(90 - MAX_ORIENT, 91 + MAX_ORIENT), range(270 - MAX_ORIENT, 271 + MAX_ORIENT)):
        radian = math.radians(degree)
        for r in range(MAX_R):
            normal = np.array([math.cos(radian), math.sin(radian)])
            
            buf = []
            for region in regions:
                testVector = np.array([region.centroid[1] - center_x, region.centroid[0] - center_y])
                distance = abs(r - np.dot(testVector, normal))
                if distance < DIS_TOR:
                   buf.append(region)

            if len(buf) > len(returnRegions):
                returnRegions = buf 
                optimal_radian, optimal_r = radian, r

    # Show the line.
    # It's not necessary to understand the code below, just know it will draw the optimal line.
    # =============================
    if showLine and optimal_radian != 0:
        tangent = [math.sin(optimal_radian), -math.cos(optimal_radian)]
        pivotPoint = [center_x + optimal_r * math.cos(optimal_radian), center_y + optimal_r * math.sin(optimal_radian)]
        ax = plt.subplot()
        line = lines.Line2D(
            [0, width], 
            [(-pivotPoint[0] / tangent[0]) * tangent[1] + pivotPoint[1], ((width - pivotPoint[0]) / tangent[0]) * tangent[1] + pivotPoint[1]], 
            color="blue",
            linewidth=2
        )
        ax.add_line(line)
    # =============================
    print(f"degree: {math.degrees(optimal_radian)}")
    print(f"radius: {optimal_r}")
    
    return returnRegions


def filterRegionsByOutlier(regions) -> list:
    if len(regions) < 2:
        return regions

    buf = sorted(regions, key=lambda region: region.centroid[1])
    center = width / 2
    start, end = 0, len(buf)

    for i in range(len(buf) - 1):
        if buf[i + 1].centroid[1] - buf[i].centroid[1] > OUTLIER_TOR:
            if abs(center - buf[i].centroid[1]) > abs(center - buf[i + 1].centroid[1]):
                start = i + 1
            else:
                end = i + 1
                break

    return buf[start:end]


def main():
    if os.path.exists(OUTPUT_DIR):
        # Remove former output.
        shutil.rmtree(OUTPUT_DIR)
    os.mkdir(OUTPUT_DIR)

    print("----------------")
    count = 0
    for fileName in os.listdir(INPUT_DIR):
        with open(f"{INPUT_DIR}{fileName}", "r") as imgFile:
            # Pre-process.
            # =============================
            count += 1
            print(f"Handing #{count}")
            
            plt.figure(count)
            plt.ion()
            plt.show()
            
            #imgName = imgFile.name.split("/")[-1].split(".")[0]
            print(f"image name: {fileName}")
            global width, height
            # =============================

            # Image processing.
            # =============================
            image = io.imread(imgFile.name)
            plt.imshow(image)

            height = image.shape[0]
            width = image.shape[1]

            image = color.rgb2gray(image)
            image = exposure.equalize_adapthist(image)
            image = filters.unsharp_mask(image)
            #plt.imshow(image, cmap="gray")

            threshold = filters.threshold_otsu(image)
            print(f"threshold: {threshold}")
            image = image < threshold # To binary & 0, 1 invert.

            image = morphology.binary_opening(image)
            #plt.imshow(image, cmap="binary")
            # =============================
            
            # Filter regions.
            # =============================
            image = measure.label(image, connectivity=2)
            regions = measure.regionprops(image)
            possibleRegions = filterRegionsByShape(regions)
            possibleRegions = filterRegionsByLine(possibleRegions)
            possibleRegions = filterRegionsByOutlier(possibleRegions)
            # =============================
            
            # Write text file & Show result.
            # =============================
            ax = plt.subplot()
            print(f"{len(possibleRegions)} region(s) found")

            with open(f"{OUTPUT_DIR}output.txt", "a") as txtFile:
                txtFile.write(f"{fileName}\n")
                txtFile.write(f"{len(possibleRegions)}\n")
                
                for region in possibleRegions:
                    minr, minc, maxr, maxc = region.bbox
                    print(f"area: {region.area}\tratio: {round((maxc - minc) / (maxr - minr), 3)}\tcenter: {region.centroid}")
                    
                    rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor="red", linewidth=1)
                    ax.add_patch(rect)
                    
                    txtFile.write(f"{minc} {minr} {maxc - minc} {maxr - minr}\n")
            
            plt.draw()
            plt.pause(0.001)
            # =============================
            

        print("----------------")

    input("===== Press 'Enter' to finish. =====")


if __name__ == "__main__":
    main()
