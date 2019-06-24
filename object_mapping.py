import sys
import os
import os.path
import time
import numpy as np
from math import radians, pi, cos, sin, asin, sqrt, atan2, tan, atan, exp, log
from scipy.cluster.hierarchy import linkage, fcluster
import csv
from statistics import mean

'''
------------------------------------------
This file contains Python (v2.7) implementation of the MRF-based tringulation procedure introduced in
"Automatic Discovery and Geotagging of Objects from Street View Imagery"
by V. A. Krylov, E. Kenny, R. Dahyot.
https://arxiv.org/abs/1708.08417
version 1.1
Copyright (c) ADAPT centre, Trinity College Dublin, 2018
------------------------------------------
The module takes the ouput of object detection and depth estimation deployed on the original image set.
Each line in the input CSV file defines a detected object with FOUR floating point values:
camera positions (GPS latitude and longitude), bearing from north clockwise in degrees towards the
object in the panoramic image and the depth estimate. The latter may be omitted or set to zero.
The module performs triangulation, MRF optimization to establish the optimal object configuration
and clustering.
The output CSV contains the list of GPS-coordinates (latitude and longitude) of identified objects
of interests and a score value for each of these. The score is the number of individual views
contributing to an object (greater or equal to 2).
------------------------------------------
'''

###########################################
###  I N P U T     P A R A M E T E R S  ###
###########################################

# Input CSV file
inputfilename = './detection_input_one.csv'
# Output CSV file
outputfilename = './water_tower_detection.csv'

# preset parameters
MaxObjectDstFromCam = 2500	# Max distance from camera to objects (in meters) (original: 25, used: 2500)
MaxDstInCluster = 500		# Maximal size of clusters employed (in meters) (original: 1, used: 500)

# MRF optimization parameters
ICMiterations = 15		# Number of iterations for ICM
DepthWeight = 0.005		# weight alpha in Eq.(4) (original: 0.2)
ObjectMultiView = 0.005		# weight beta in  Eq.(4) (original: 0.2)
StandAlonePrice = max(1 - DepthWeight - ObjectMultiView, 0) # weight (1-alpha-beta) in Eq. (4)
#values that reproduce the empty distance matrix error with paper's data: 0.5, 0.5

###########################################


# conversion from (lat,lon) to meters
def LatLonToMeters( lat, lon ):
    "Converts given lat/lon in WGS84 Datum to XY in Spherical Mercator EPSG:4326"
    originShift = 2 * pi * 6378137 / 2.0 #6378137 is radius of Earth at the equator (in meters)
    mx = lon * originShift / 180.0
    my = log( tan((90 + lat) * pi / 360.0 )) / (pi / 180.0)
    my = my * originShift / 180.0
    return mx, my

# conversion from meters to (lat,lon)
def MetersToLatLon( mx, my ):
    "Converts XY point from Spherical Mercator EPSG:4326 to lat/lon in WGS84 Datum"
    originShift = 2 * pi * 6378137 / 2.0
    lon = (mx / originShift) * 180.0
    lat = (my / originShift) * 180.0
    lat = 180 / pi * (2 * atan(exp(lat * pi / 180.0)) - pi / 2.0)
    return lat, lon


# haversine distance formula between two points specified by their GPS coordinates
def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    #lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    lon1, lat1, lon2, lat2 = list(map(radians, [lon1, lat1, lon2, lat2]))
    print("Haversine (lon1, lat1, lon2, lat2):",lon1, lat1, lon2, lat2)
    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    print("Haversine (dlon, dlat):",dlon,dlat)
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    m = 6367000. * c
    print("Haversine (a, c, m):",a, c, m)
    return m

# calculating the intersection  point between two rays (specified each by camera position
#and depth-estimated object location)
def Intersect(Object1, Object2, MaxObjectDstFromCam, mx_my_list) :
    latC1 = Object1[5]
    latC2 = Object2[5]
    lonC1 = Object1[6]
    lonC2 = Object2[6]

    latP1 = Object1[0]
    latP2 = Object2[0]
    lonP1 = Object1[1]
    lonP2 = Object2[1]

    a1 = latP1 - latC1
    b1 = latP2 - latC2
    c1 = latC2 - latC1

    a2 = lonP1 - lonC1
    b2 = lonP2 - lonC2
    c2 = lonC2 - lonC1

    if a2*b1-b2*a1 :
        y = (a1*c2 - a2*c1) / (a2*b1-b2*a1)
        print("y:",y)
    else :
        return -1, -1, 0 ,0
    if a1 != 0 :
        x = (b1*y+c1) / a1
    else :
        x = (b2*y+c2) / a2

    #print("x:",x)
    #print("mx:",a1*x+latC1,"my:",a2*x+lonC1)

    if (x < 0) or (y < 0) :
        #print("if (x < 0) or (y < 0) is TRUE")
        return -2, -2, 0, 0
        #if x < 0:
            #x = x*(-1.)
        #if y < 0:
            #y = y*(-1.)
    if (x > MaxObjectDstFromCam) or (y > MaxObjectDstFromCam) :
        #print("if (x > MaxObjectDstFromCam) or (y > MaxObjectDstFromCam) is TRUE")
        return -3, -3, 0, 0
    mx, my = a1*x+latC1, a2*x+lonC1 #original version
    #mx, my = a1*x+latC1, a2*y+lonC1
    #mx, my = a1*y+latC1, a2*x+lonC1 #seems to provide slighly better results overall

    mx_my_list.append([mx,my])

    print("x, y, mx, my:",x, y, mx, my)
    return x, y, mx, my

# calculate the MRF energy of an intersection
def CalcEnergyObject(ObjectsDst,ObjectsBase,ObjectsConnectivity,Object) :
    inters = np.count_nonzero(ObjectsConnectivity[Object,:])
    if inters == 0:
        #print("if inters == 0 IS TRUE")
        return StandAlonePrice
    Energy = 0
    dpthmin, dpthmax = 1000, 0
    for i in range(len(ObjectsBase)) :
        if ObjectsConnectivity[Object,i]:
            #THIS CONDITION IS TRUE WITH BOTH MY AND PAPER'S DATA
            #print("if ObjectsConnectivity[Object,i] IS TRUE")
            dpthPen = DepthWeight*abs(ObjectsDst[Object,i] - (ObjectsBase[Object])[3])
            #print("dpthPen:",dpthPen)
            Energy += dpthPen
            dpth = ObjectsDst[Object,i]
            #print("dpth:",dpth)
            if dpth<dpthmin:
                dpthmin = dpth
            if dpth>dpthmax:
                dpthmax = dpth
    #print("Energy + ObjectMultiView*(dpthmax-dpthmin):",Energy + ObjectMultiView*(dpthmax-dpthmin))
    return Energy + ObjectMultiView*(dpthmax-dpthmin)

# calculate the averaged object location (used after clustering)
def CalcAvrgObject(Intersects,ObjectsConnectivity,Object) :
    #print("Intersects:",Intersects,"\n","ObjectsConnectivity:",ObjectsConnectivity)
    #unique, counts = np.unique(ObjectsConnectivity, return_counts=True)
    #print(np.asarray((unique,counts)).T)
    res = np.zeros(2)
    cnt = 0
    for i in range(Intersects.shape[0]) :
        #THE CONDITION ON THE NEXT LINE DOES NOT EVALUATE TO TRUE WITH MY DATA
        #IT IS BECAUSE OBJECTSCONNECTIVITY ONLY HAS ZEROS IN IT
        if ObjectsConnectivity[Object,i]:
            res[:] += Intersects[Object,i,:]
            cnt += 1
        #else:
            #print("if ObjectsConnectivity[Object,i] is FALSE")
    if cnt :
        #print("if cnt is TRUE")
        return res/cnt
    #print("res:",res)
    return res


# hierarchical clustering
def MyClust(intersects,MaxIntraDegreeDst) :
    Z = linkage(np.asarray(intersects))
    clusters = fcluster(Z, MaxIntraDegreeDst, criterion='distance') - 1
    NumClusters =  max(clusters) + 1
    IntersectClusters = np.zeros((NumClusters,3))
    for i in range(len(intersects)) :
        IntersectClusters[clusters[i],0] += (intersects[i])[0]
        IntersectClusters[clusters[i],1] += (intersects[i])[1]
        IntersectClusters[clusters[i],2] += 1
    return IntersectClusters



# only PAIRWISE intersections
#def main( arguments ):
def main():

    start = time.time()
    ObjectsBase=[]

    if not os.path.isfile(inputfilename) :
        print('Input file not found. Aborting.')
        return

    if os.path.isfile(outputfilename) :
        os.remove(outputfilename)
        #print('A file with the specified ouput name already exists. Aborting.')
        #return

    try:
        f1=open(outputfilename,'w')
        f1.close()
    except:
        print('A file with the specified ouput name cannot be created. Aborting.')
        return

    ###############################
    #### A L L  O B J E C T S #####
    ###############################


    with open(inputfilename,'r') as f:
        next(f)	# skip the first line
        for line in f:
            nums = line.split(',')
            if len(nums)<3:
                print('Broken entry ignored')
            if len(nums)<4:	# if a depth estimate is not available
                lat, lon, bearing, depth = float(nums[0]), float(nums[1]), float(nums[2]), 5
            else:
                lat, lon, bearing, depth = float(nums[0]), float(nums[1]), float(nums[2]), float(nums[3])
            if depth<=0:
                depth =5

            # calculating the object positions from camera position + bearing + depth_estimate
            mx, my = LatLonToMeters(lat,lon)
            print("mx:",mx,"my:",my)
            #br1 = radians(bearing)
            br1 = radians(bearing + 90.)
            print("br1:",br1)
            #yCP = my + depth * cos(br1) * 640/256	# depth-based positions
            #xCP = mx + depth * sin(br1) * 640/256
            yCP = my + depth * cos(br1) * 640/256	# depth-based positions
            xCP = mx + depth * sin(br1) * 640/256
            print("yCP:",yCP,"xCP:",xCP)
            latp, lonp = MetersToLatLon(xCP, yCP)
            print("latp:",latp,"lonp:",lonp)
            #yCP = my + 1.0 * cos(br1) * 640/256	# normalized positions (at 1m distance from camera)
            #xCP = mx + 1.0 * sin(br1) * 640/256

            #yCP = my + 200.0 * cos(br1) * 640/256	# normalized positions (at 100m distance from camera)
            #xCP = mx + 200.0 * sin(br1) * 640/256

            yCP = my + 200.0 * cos(br1) * 640/256 # normalized positions (at 100m distance from camera)
            xCP = mx + 200.0 * sin(br1) * 640/256

            latp1, lonp1 = MetersToLatLon(xCP, yCP)

            ObjectsBase.append( (latp1,lonp1,bearing,depth,0,lat,lon,latp,lonp) )
            #print("ObjectsBase:",ObjectsBase)

    keys = ['latp1','lonp1','bearing','depth','zero','lat','lon','latp','lonp']
    with open('object_positions.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        for row in ObjectsBase:
            writer.writerow(row)

    print("All detected objects: {0:d}".format(len(ObjectsBase)))

    #############################
    #### A D M I S S I B L E ####
    #############################

    #the maximal distance between the two camera positions observing the same object
    MaxCamDst = 1.5 * MaxObjectDstFromCam

    #Intersects = []
    NumIntersects = 0
    ObjectsDst = np.zeros((len(ObjectsBase),len(ObjectsBase)))
    Intersects = np.zeros((len(ObjectsBase),len(ObjectsBase),2))
    #list added by me to hold intersection mx and my values:
    mx_my_list = [['lat', 'lon']]
    for i in range(len(ObjectsBase)) :
        if (i%1000 == 0) and (i>0):
            print('Parced {} object entries ({:.2f}%)'.format(i,100.*i/len(ObjectsBase)))
        ObjectsDst[i,i] = -5
        for j in range(i+1,len(ObjectsBase)) :
            print((ObjectsBase[i])[6],(ObjectsBase[i])[5],(ObjectsBase[j])[6],(ObjectsBase[j])[5])
            CamDstMtrs = haversine((ObjectsBase[i])[6],(ObjectsBase[i])[5],(ObjectsBase[j])[6],(ObjectsBase[j])[5])

            # cam_positions - same (less than 1m apart) or too far
            if (CamDstMtrs < 0.5) or (CamDstMtrs > MaxCamDst) :
                #print("if (CamDstMtrs < 0.5) or (CamDstMtrs > MaxCamDst) IS TRUE")
                if CamDstMtrs < 0.5:
                    #print("if (CamDstMtrs < 0.5) IS TRUE")
                    print("CamDstMtrs:",CamDstMtrs)
                else:
                    print("if (CamDstMtrs > MaxCamDst) IS TRUE")
                ObjectsDst[i,j] = -4
                ObjectsDst[j,i] = -4
                continue
            ObjectsDst[i,j], ObjectsDst[j,i], Intersects[i,j,0], Intersects[i,j,1] = Intersect(ObjectsBase[i], ObjectsBase[j], MaxObjectDstFromCam,
            mx_my_list)
            Intersects[j,i,0], Intersects[j,i,1] = Intersects[i,j,0], Intersects[i,j,1]
            if ObjectsDst[i,j] > 0 :
                NumIntersects += 1

    mean_lat = mean([lst[0] for lst in mx_my_list[1:]])
    mean_lon = mean([lst[1] for lst in mx_my_list[1:]])
    mx_my_list.append([mean_lat,mean_lon])
    with open('intersections.csv', 'w') as int_csv:
        writer = csv.writer(int_csv)
        writer.writerows(mx_my_list)

    print("All admissible intersections: {0:d}".format(NumIntersects))

    ObjectsConnectivity = np.zeros((len(ObjectsBase),len(ObjectsBase)),dtype=np.uint8)
    ObjectsConnectivityViableOptions = np.zeros(len(ObjectsBase),dtype=np.uint8)
    for i in range(len(ObjectsBase)) :
        ObjectsConnectivityViableOptions[i] = np.count_nonzero(ObjectsDst[i,:]>0)

    #############################
    ########### I C M ###########
    #############################

    np.random.seed(int(100000.0*time.time())%1000000000)
    chngcnt = 0
    for ICMiter in range(ICMiterations*len(ObjectsBase)) :
        if (ICMiter+1)%(len(ObjectsBase)) == 0:
            print('Iteration #{}: accepted {} changes'.format((ICMiter+1)/(len(ObjectsBase)),chngcnt))
            chngcnt = 0
        testObject = np.random.randint(0, len(ObjectsBase))
        if ObjectsConnectivityViableOptions[testObject] == 0 :	# no pairing possible (standalone - )
            #print("if ObjectsConnectivityViableOptions[testObject] == 0 is TRUE")
            #THIS CONDITION IS NEVER TRUE WITH PAPER'S DATA OR MY DATA
            continue

        randnum  = 1+np.random.randint(0, ObjectsConnectivityViableOptions[testObject])
        curcnt = 0
        for i in range(len(ObjectsBase)) :
            if (ObjectsDst[testObject,i]>0) :
                curcnt += 1
                #print("if (ObjectsDst[testObject,i]>0) IS TRUE")
                #THIS CONDITION IS TRUE WITH BOTH PAPER'S AND MY DATA
            if curcnt == randnum :
                testObjectPair = i
                #print("if curcnt == randnum IS TRUE")
                #THIS CONDITION IS TRUE WITH BOTH PAPER'S AND MY DATA
                break

        EnergyOld  = CalcEnergyObject(ObjectsDst,ObjectsBase,ObjectsConnectivity,testObject)
        #print("EnergyOld: ",EnergyOld)
        EnergyOld += CalcEnergyObject(ObjectsDst,ObjectsBase,ObjectsConnectivity,testObjectPair)
        #print("EnergyOld 2: ",EnergyOld)

        #print("testObject:",testObject,"; testObjectPair:",testObjectPair)
        ObjectsConnectivity[testObject, testObjectPair] = 1 - ObjectsConnectivity[testObject, testObjectPair]
        ObjectsConnectivity[testObjectPair, testObject] = 1 - ObjectsConnectivity[testObjectPair, testObject]

        #unique, counts = np.unique(ObjectsConnectivity, return_counts=True)
        #print(np.asarray((unique,counts)).T)
        #AT THIS STAGE, I GET 2 "1" VALUES IN OBJECTSCONNECTIVITY ARRAY (AND NO ACCEPTED CHANGES)
        #WITH PAPER'S DATA, THERE IS A LOT MORE "1" VALUES AND ACCEPTED CHANGES

        EnergyNew  = CalcEnergyObject(ObjectsDst,ObjectsBase,ObjectsConnectivity,testObject)
        #print("EnergyNew: ",EnergyNew)
        EnergyNew += CalcEnergyObject(ObjectsDst,ObjectsBase,ObjectsConnectivity,testObjectPair)
        #print("EnergyNew 2: ",EnergyNew)

        if EnergyNew<=EnergyOld:
            chngcnt += 1
            continue

        # revert to the old configuration
        ObjectsConnectivity[testObject, testObjectPair] = 1 - ObjectsConnectivity[testObject, testObjectPair]
        ObjectsConnectivity[testObjectPair, testObject] = 1 - ObjectsConnectivity[testObjectPair, testObject]

        #unique, counts = np.unique(ObjectsConnectivity, return_counts=True)
        #print(np.asarray((unique,counts)).T)
        #AT THIS STAGE, THOSE '1' VALUES ARE ALL CHANGED BACK TO '0'
        #WITH PAPER'S DATA, THERE ARE STILL '1'S IN OBJECTSCONNECTIVITY AT THIS STAGE

    #############################
    #### C L U S T E R I N G ####
    #############################

    mx, my = LatLonToMeters((ObjectsBase[0])[0], (ObjectsBase[0])[1])
    d45 = 0.707 * MaxDstInCluster * 640.0/256;
    ax, ay = MetersToLatLon(mx+d45, my+d45)
    ax1, ay1 =  MetersToLatLon(mx, my)
    MaxDegreeDstInCluster = ((ax-ax1)**2+(ay-ay1)**2)**0.5
    print(mx,my)
    print(ax,ax1,ay,ay1)
    print(MaxDegreeDstInCluster)

    ICMintersect = []
    ifObjectIntersects = np.zeros(len(ObjectsBase),dtype=np.uint8)
    for i in range(len(ObjectsBase)) :
        #unique, counts = np.unique(ObjectsConnectivity, return_counts=True)
        #print(np.asarray((unique,counts)).T)
        #MY DATA DOES NOT PRODUCE A NON-ZERO OBJECTSCONNECTIVITY ARRAY
        #if np.count_nonzero(ObjectsConnectivity):
            #print("ObjectsConnectivity IS NON ZERO")
        res = CalcAvrgObject(Intersects,ObjectsConnectivity,i)
        if res[0] :
            ifObjectIntersects[i] = 1
            ICMintersect.append((res[0], res[1]))

    print("ICM intersections: {0:d}".format(len(ICMintersect)))
    IntersectClusters = MyClust(ICMintersect,MaxDegreeDstInCluster)
    print("IntersectClusters:",IntersectClusters)

    NumClusters = IntersectClusters.shape[0]
    with open(outputfilename, "w") as inter:
        inter.write("lat,lon,score\n")
        for i in range(NumClusters) :
            inter.write("{0:f},{1:f},{2:d}\n".format(IntersectClusters[i,0]/IntersectClusters[i,2], \
            IntersectClusters[i,1]/IntersectClusters[i,2],int(IntersectClusters[i,2])))
    print("Number of output ICM clusters: {0:d}".format(NumClusters))

    print("Elapsed total time: {0:.2f} seconds.".format(time.time() - start))



if __name__ == '__main__':
    #main( sys.argv )
    main()
