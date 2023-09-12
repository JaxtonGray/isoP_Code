# Importing the needed packages that will be used throughout
import numpy as np
import pandas as pd
from scipy.spatial import Delaunay, distance
import netCDF4 as nc
import math

def readSHD_File(path, basinName):
    # JG -- Open SHD file, maybe we can make this more open to SHD files with different names
    # Converts file into a line by line list
    pathSHD = path + r"\basin" + "\\" + basinName + r"_shd.r2c"
    shdFile = open(pathSHD, "r")
    shdList = shdFile.readlines()

    # JG -- We create an index containing the line # of the SHD file that has 
    # the specific info we need
    info = []
    positionInfo = []
    index=[]
    for k in range(0,len(shdList)):
        if ":SourceFileName" in shdList[k]:
            index.append(k)
        elif ":Projection" in shdList[k]:
            index.append(k)
        elif ":xOrigin" in shdList[k]:
            index.append(k)
        elif ":yOrigin" in shdList[k]:
            index.append(k)
        elif ":xCount" in shdList[k]:
            index.append(k)
        elif ":yCount" in shdList[k]:
            index.append(k)
        elif ":xDelta" in shdList[k]:
            index.append(k)
        elif "yDelta" in shdList[k]:
            index.append(k)
            
    #JG -- Using the nexly created index, we the add the needed info inyo the info list
    for i in index:
        info.append(shdList[i].split())
    #JG -- Separating the actual info from the names we place it into a new list and then 
    # assign them variables
    for j in range(0, len(info)):
        positionInfo.append(info[j][1])
    mapName = positionInfo[0]
    projection = str(positionInfo[1])
    xOrigin = float(positionInfo[2])
    yOrigin = float(positionInfo[3])
    xCount =  int(positionInfo[4])
    yCount = int(positionInfo[5])
    xDelta = float(positionInfo[6])
    yDelta = float(positionInfo[7])

    #--------------------------------------------------------------------------
    #
    # Now that we have the information necessary to calculate the LAT LONG
    # pairs at each WATFLOOD grid. This code below does just that based on the
    # geographic information extracted from the SHP file for your basin.
    #
    #--------------------------------------------------------------------------
    WFcoord_X = np.zeros((yCount+1, xCount+1))
    WFcoord_Y = np.zeros((yCount+1, xCount+1))
    for m in range(yCount, 0, -1):
        for n in range(1, xCount+1, 1):
            if (m==yCount and n==1):
                WFcoord_X[m,n]=(xOrigin+xDelta/2)
                WFcoord_Y[m,n]=(yOrigin+yDelta/2)
            elif (n==1):
                WFcoord_Y[m,n]=WFcoord_Y[m+1, n]+yDelta
            elif (m==yCount):
                WFcoord_X[m,n]=WFcoord_X[m,n-1]+xDelta
            else:
                WFcoord_X[m,n]=WFcoord_X[yCount,n]
                WFcoord_Y[m,n]=WFcoord_Y[m,1]
    #JG -- I am unsure as to why I could not just make the initialiization array smaller
    # regradless, the below code erases the excess rows and columns!
    WFcoord_X = WFcoord_X[1:, 1:]
    WFcoord_Y = WFcoord_Y[1:, 1:]

    #JG -- TBH I am not sure why lines 76-78 exist, but I will copy them anyways
    WFcoord_Y[yCount-1: ] = WFcoord_Y[yCount-1, 0]
    WFcoord_X[:,0] = WFcoord_X[yCount-1, 0]
    numCells = xCount * yCount

    #JG -- These lines reshape the arrays into single column rows containing the information, then combines them
    newCoord_Y = np.reshape(WFcoord_Y, (numCells, 1), 'F')
    newCoord_X = np.reshape(WFcoord_X, (numCells, 1), 'F')
    WFcoords = np.column_stack((newCoord_Y, newCoord_X))
    #--------------------------------------------------------------------------
    #
    # Check if the basin is it LAT LONG coordinates. If it is not, then the LAT
    # LONGS of each grid need to be calculated based on the UTM coords. Prompt 
    # the user for which province the basin is in if this is the case. 
    #
    #--------------------------------------------------------------------------

    #if (projection == "UTM" or projection == "Cartesian"):
        # Will only activate if the projection is a UTM Zone
        #JG -- I can't finish this part of the code yet, MATLAB let you pick the zone like "17T" but python operates differently
        # in order for the process to be more automated it needs to pull the "T" part out on its own, but SHD file doesn't include it

    #JG -- Finally we must create an excel spreadsheet that can be converted into a shape file later
    pathCSV = path + "\\isoP\\" + basinName + "_coords.csv"
    np.savetxt(pathCSV, WFcoords, delimiter=",")

    # at this point, you must use the WF_coords excel file, convert it to a
    # shapefile, and then import that shapefile into ARC GIS software to obtain
    # the Koeppen cliamte classification for the WATFLOOD basin grids. 
    
    print(basinName + "_SHD.r2c file was successfully read in!")

    return WFcoords

def extract_NARR_timeseries(path,basinName):
    #################################################################
    # This program reads in an array of lat/long coordinates (I.e the
    # basin_coords.xls file previously output by read_SHD_file.m)
    # and extracts the NARR data at the corresponding grid locations.
    #
    #
    # ** IMPORTANT NOTE: If the output files already exist in the folder
    # directory, you must delete them before running this program (ie.
    # NARR_varname_mon_mean). Otherwise the program will just add the extracted
    # data onto the end of the previous dataset.
    #
    #################################################################
    #Extract all NARR data at the WATFLOOD grid coords.
    #This is accomplished through reading in an excel file with the names of
    #the NARR monthly mean files. These files names utilize the naming
    #convention of the NARR repository. Therefore, if the data files need to be
    #updated to include additional years, they can be directly downloaded form
    #the NARR server and the names should still be relevant to this program.
    #This requires reading in an xls file called "filename". This file must be
    #stored in the NARR directory. The PATH vairable below will need to be
    #changed depending on where the isoP file strcuture is stored.

    #From the path name input by the user, backtrack to the main SPL directory
    #to access the isoP_MAIN folder where the NARR netCDF files are stored.
        #JG -- is the main intent with this section of the code just to go back by one 
    splitPath = path.split("\\")
    for i in range(0, len(splitPath)-1):
        if i == 0:
            pathCharm = splitPath[0]
        else:
            pathCharm = pathCharm + "\\" + splitPath[i]
    
    #Read in the .xls file called 'NARR_var_names' which is stored in the
    #isoP_MAIN\Code folder. This file is a list of all the NARR climate
    #variables to read in. If you don't want to read all of the NARR variablesa
    #in, you can delete them from this list. Or, conversely, if you would like
    #to read more variables in, put the netCDF file for a monthly mean climate
    #varibale in this folder, add it's name to the .xls file and augment this
    #code to read it in.
    pathNARR = pathCharm + "\\isoP_Code\\NARR\\NARR_var_names.csv"
    varsNARR = np.genfromtxt(pathNARR, delimiter=',', encoding='utf-8-sig')
    numClimatePar = len(varsNARR)
    
    # Read in the WATFLOOD pairs of long/lats (Longitude is in the first column,
    # latitude is in the second column). These were output from read_SHD_file.m
    # # to the SPL\basin\isoP folder earlier.
    pathCoords=path + "\\isoP\\" + basinName + "_coords.csv"
    WFcoords = pd.read_csv(pathCoords, header=None)
    #JG -- Renames the columns to the respective Lat Long pairs
    WFcoords.rename(columns={0: 'LAT', 1: 'LONG'}, inplace=True)

    # Find the number of WATFLOOD points that NARR data needs to be
    # extracted at.
    numPts = len(WFcoords.index)

    # This next portion of code uses latitude and longitude pairs (from in the form [lat,long]) 
    # to find the corresponding NARR grid (xy) to extract the data from.
    # This is accomplished by finding the location of nearest four grid points corresponding to a certain 
    # location consist of latitude and longitude. 
    # NOTE: The NARRlatlon.mat file MUST be in the working directory! Otherwise the code will not run.
    #JG -- This is another instance where the MATLAB features allow this to be done easier, I will find a workaround
    #-- Loading in the NARR lat and lon files
    latPath = pathCharm + "\\isoP_Code\\NARR\\" + "NARRlat.csv"
    lonPath = pathCharm + "\\isoP_Code\\NARR\\" + "NARRlon.csv" 

    #JG -- Converting the csv tp arrays and resizing them so we can combine them
    lat = np.genfromtxt(latPath, delimiter=',', encoding='utf-8-sig')
    lon = np.genfromtxt(lonPath, delimiter=',', encoding='utf-8-sig')
    gridSize = lat.shape
    latr = np.reshape(lat, (np.size(lat), 1), 'F')
    lonr = np.reshape(lon, (np.size(lon), 1), 'F')
    X = np.column_stack((latr, lonr))

    #JG -- Begining nearest neighbour calculations
    delaunayTri = Delaunay(X)
    K = np.zeros((numPts, 1)) #Initialize empty array to hold indices
    xy_NARR = np.zeros((numPts, 2)) #Initialize empty array to hold NN lat lon pairs
    for j in range(numPts):
        coord = WFcoords.iloc[j, :]
        dist = distance.cdist(delaunayTri.points, np.array([coord]))
        K[j] = np.argmin(dist) + 1
        gridNo = [K[j] % gridSize[0], (K[j] // gridSize[0])+1]
        xy_NARR[j, 0] = gridNo[1]
        xy_NARR[j, 1] = gridNo[0] 
    # This loop will cycle through all of the NARR datafiles (ie. one loop for each climate variable)
    # whose names were in the spreadsheet previously read.
    print("Reading in NARR climate variables!")
    output = [[] for _ in range(numClimatePar)]
    for i in range(numClimatePar):
        filetmp = varsNARR.iloc[i, 0]
        ncid = nc.Dataset(filetmp, 'r')
        # Depending on which NARR climate variable is being extracted, time (as well as other parts of the data) will
        # be stored in a different dimension of the netCDF file. This portion of
        # code extracts the time variable from the netCDF files, specifying the
        # different locations depending on file name.
        time =  ncid.variables['time'][:]
        timeLength = len(time)

        # Create a matrix of dates for the NARR data (because the format they use is
        # weird and this is just easier to get month and yer next to the
        # corresponding NARR data. If the NARR files are updated to include 2013,
        # the end year will have to be updated to reflect that change as well.
        startYear = 1979
        numYears = timeLength / 12
        numFullYears=math.floor(numYears)
        numPartYears = numYears-numFullYears
        numMonths = round(numPartYears * 12)
        if (timeLength % 12) == 0:
            numCol = timeLength / 12
        else:
            rem = timeLength % 12
            numCol = (timeLength + (12 - rem))/12
        matrixSize = [12, int(numCol)]
        monthMatrix = np.zeros((matrixSize[0], matrixSize[1]))
        yearMatrix = np.zeros((matrixSize[0], matrixSize[1]))
        if (numPartYears == 0):
            numElements = numFullYears * 12
        else:
            numElements = (numFullYears+1)*12

        #JG -- Creating the date arrays to use
        for k in range(numFullYears+1):
            if (k == (numFullYears)):
                for j in range(numMonths):
                    monthMatrix[j,k] = j + 1
                    if k==0:
                        yearMatrix[j,k] = startYear
                    else:
                        yearMatrix[j,k] = (startYear + k)
            else:
                for j in range(12):
                    monthMatrix[j, k] = j +1 
                    if k==0:
                        yearMatrix[j,k] = startYear
                    else:
                        yearMatrix[j,k] = (startYear + k)
        
        #JG -- Fromatting the date arrays
        monthArray = np.reshape(monthMatrix, (numElements, 1), 'F')
        monthArray = monthArray[monthArray != 0]
        yearArray = np.reshape(yearMatrix, (numElements, 1), 'F')
        yearArray = yearArray[yearArray != 0]

        #JG -- Creating the date Matrix
        dateMatrix = np.column_stack((yearArray, monthArray))

        varID = filetmp.split('.')[0]
        #JG -- varID was 6 in matlab code, but I believe it is referring to variable name
        # Extract NARR climate variable time series for each WATFLOOD grid point.
        # NOTE: This data is extracted for the ENTIRE time series (ie. for NARR
        # this is from January 1, 1979 onwards).
        allData = ncid.variables[varID][:]
        out1 = np.zeros((timeLength, numPts))
        for j in range(numPts):
            data = allData[:timeLength, int(xy_NARR[j, 1]), int(xy_NARR[j, 0])]
            out1[:, j] = data
        output[i] = np.concatenate((dateMatrix, out1), axis=1)
    print("NARR climate variables successfully read in for specified WATFLOOD grids!")
    #JG --The final output is off by enough decimals that I am unconfident, however it runs so that will have to wait
    #I beleive the reason for this is likely that the information I am pulling from the netCDF file is different than the matlab version
    return output, pathNARR, pathCharm

def NARR_format_timeseries_basin(output, pathNARR, startYear, endYear):
    varsNARR = np.genfromtxt(pathNARR, delimiter=',', encoding='utf-sig-8')
    numClimatePar = len(varsNARR)
    print("Format NARR Climate Variables and crop dataset to specified year.")

    #Initializing some new parameters
    oldNARR= [[] for _ in range(numClimatePar)]
    month = [[] for _ in range(numClimatePar)]
    year = [[] for _ in range(numClimatePar)]
    newNARR = [[] for _ in range(numClimatePar)]

    for k in range(numClimatePar):
        filetmp = varsNARR[k, 1]

        #Read in data from the NARR file
        oldNARR[k] = output[k][:, 2:]
        length, numGrid = oldNARR[k].shape
        #Extracting date into columns of data
        month[k] = output[k][:, 1]
        year[k] = output[k][:, 0]

        #Reinitalizing the newNARR variable (may not keep this)
        newNARR[k] = np.zeros((length, numGrid))

        #Creating a list containg the cumulative variables that need special treatment
        cumVars = ['apcp_mon_mean', 'evap_mon_mean', 'acpcp_mon_mean', 'prwtr_mon_mean']
        if filetmp in cumVars:
            for j in range(numGrid):
                for i in range(length):
                    #Removing negatives from the dataset that may hav ebeen caused by errors in the NARR data
                    #Needed for these variables as negatives are not possible
                    if oldNARR[k][i, j] < 0:
                        oldNARR[k][i, j] = 0
                    
                    #Formating based on 30 day months
                    if (month[k][i] == 4) or (month[k][i] == 6) or (month[k][i] == 9) or (month[k][i] == 11):
                        newNARR[k][i, j] = oldNARR[k][i, j]  * 30
                    #Fromating them depending on February or not and also if a leap year or not
                    elif (month[k][i] == 2):
                        if (year[k][i] % 4 == 0):
                            newNARR[k][i, j] = oldNARR[k][i, j]  * 29
                        else:
                            newNARR[k][i, j] = oldNARR[k][i, j]  * 28
                    #For all other months
                    else:
                        newNARR[k][i, j] = oldNARR[k][i, j]  * 31
        elif filetmp == 'air2m_mon_mean':
            newNARR[k] = oldNARR[k]- 273.15 #Converting from kelvin into celsius
        else:
            newNARR[k] = oldNARR[k]
    #Initializing the rehaped grids
    gridNo = np.zeros((length))
    outputNARR_all = [np.zeros((length, numClimatePar)) for _ in range(numGrid)]
    reshapeNARR = [np.zeros((length, numClimatePar)) for _ in range(numGrid)]

    #Reshaping the grids
    for k in range(numGrid):
        for m in range(numClimatePar):
            reshapeNARR[k][:, m] = newNARR[m][:, k]
            gridNo[:] = k+1
        outputNARR_all[k] = np.column_stack((gridNo, month[0], year[0], reshapeNARR[k]))
    
    #Finding the start row and end row
    newLength = outputNARR_all[0].shape[0]
    #Usuing a for loop to find the start and end years and assign them to the indexs
    for i in range(newLength):
        if (outputNARR_all[0][i, 2] == startYear) and (outputNARR_all[0][i-1, 2] == startYear - 1):
            startIndex = i
        if (outputNARR_all[0][i, 2] == endYear) and (outputNARR_all[0][i+1, 2] == endYear + 1):
            endIndex = i

    #Creating a outputNARR that is cropped to the specified years
    outputNARR = []
    for i in range(len(outputNARR_all)):
        outputNARR.append(outputNARR_all[i][startIndex:endIndex+1, :])
    print("NARR files successfully formatted for specified WATFLOOD grids!")
    return outputNARR

def extract_GIS_info(startYear, endYear, WFcoords, pathCharm):
    #JG -- Extracting the Lat/Longs from the WFcoords
    numGrids = len(WFcoords)
    lat = WFcoords[:,0]
    lon = WFcoords[:,1]

    #Read in DEM for Canada/Northern Tier of the United States
    print("Extracting GIS information: Elevation, KPN zone indicator.")
    pathDEM = pathCharm + '\\isoP_Code\\DEM_CAD.csv'
    dem = np.genfromtxt(pathDEM, delimiter=',', skip_header=1)
    latDEM = dem[:,0]
    lonDEM = dem[:,1]
    altDEM = dem[:,2]

    #Using Delaunay triangulation and nearest neighbour interpolation, find the
    #DEM grid point closest to each WATFLOOD grid point. Extract the elevation
    #from that grid and assign to the corresponding WATFLOOD grid point.
    xDEM = np.concatenate((latDEM[:,None], lonDEM[:,None]), axis=1)

    #Calculate the Delaunay triangulation of the DEM grid points.
    #JG -- Do not ask me how this works, I have no idea. It is a black box.
    tri = Delaunay(xDEM)
    k = np.zeros(numGrids)
    demBasin = np.zeros((numGrids, 3))
    for j in range(numGrids):
        coord = WFcoords[j, :]
        dist = distance.cdist(tri.points, np.array([coord]))
        k[j] = np.argmin(dist)
        demBasin[j,:] = np.hstack((WFcoords[j,:], altDEM[int(k[j])]))
    
    #Based on the 1020 shapefile for Canada/northern US, read in Kpn Zone
    pathKPN = pathCharm + r'\isoP_Code\Kpn_zone.csv'
    kpnZone = np.genfromtxt(pathKPN, delimiter=',', skip_header=1)
    kpn = kpnZone[:,2]
    latKPN = kpnZone[:,0]
    lonKPN = kpnZone[:,1]

    #Same procedure as before using Delaunay triangulation and nearest neighbour interpolation
    # to find the KPN zone for each WATFLOOD grid point. Extract kpn zone from that grid and assign to the 
    # corresponding WATFLOOD grid.
    xKPN = np.concatenate((latKPN[:,None], lonKPN[:,None]), axis=1)
    tri = Delaunay(xKPN)
    k = np.zeros(numGrids)
    kpnBasin = np.zeros((numGrids, 3))
    for j in range(numGrids):
        coord = WFcoords[j, :]
        dist = distance.cdist(tri.points, np.array([coord]))
        k[j] = np.argmin(dist)
        kpnBasin[j,:] = np.hstack((WFcoords[j,:], kpn[int(k[j])]))

    #Combining the elevation and KPN zone data into one array
    length = (endYear-startYear+1)*12
    dataGEO = [np.zeros((length, 4)) for _ in range(numGrids)]

    for k in range(numGrids):
        dataGEO[k][:length, 0] = lat[k]
        dataGEO[k][:length, 1] = lon[k]
        dataGEO[k][:length, 2] = demBasin[k,2]
        dataGEO[k][:length, 3] = kpnBasin[k,2]


    print("GIS information successfully read in for specified WATFLOOD grids!")
    return dataGEO

def extract_tele_timeseries_basin(WFcoords, pathCharm, startYear, endYear):
    #Find the number of watflood points that TELE data needs to be extracted at
    numGrids = len(WFcoords)
    pathTele = pathCharm + r'\isoP_Code\index_files.csv'
    #Read in the teleconnection index data
    tele = np.genfromtxt(pathTele, delimiter=',', dtype='str', encoding='utf-8-sig')
    numTele = len(tele)
    print("Read in teleconncetion indices!")

    inTele = []
    for i in range(numTele):
        file = tele[i]
        path = pathCharm + r'\isoP_code\\' + file + '.csv'
        #Read in the teleconnection index data
        inTele.append(np.genfromtxt(path, delimiter=',', skip_header=1))
    
    data = np.stack((inTele[0][:,0], inTele[0][:,1], inTele[0][:,2], inTele[1][:, 2], inTele[2][:,2], inTele[3][:,2], inTele[4][:,2], inTele[5][:,2]), axis=1)

    cellTele = []
    for k in range(numGrids):
        cellTele.append(data)

    #Find start and end row for the specified years
    length = len(cellTele[0])
    startIndex = np.where(cellTele[0][:,0] == startYear)[0].min()
    endIndex = np.where(cellTele[0][:,0] == endYear)[0].max()

    #Trimming down the size of the teleconnection index data to the specified years
    tele = cellTele
    if endIndex < length:
        for i in range(numGrids):
            tele[i] = np.delete(tele[i], slice(endIndex+1, length), axis=0)
    if startIndex > 0:
        for i in range(numGrids):
            tele[i] = np.delete(tele[i], slice(0, startIndex), axis=0)
    
    print("Teleconncetion indices successfully read in for the specified WATFLOOD gird!")

    return tele

def all_data_format_condense(outputNARR, dataGEO, tele, pathCharm):
    #JG -- These arrays are all loaded atonce in matlab, a feature which does not translate to python
    #    I have split them up into their respective files and loaded them in individually.
    files_to_load = ['geoStatsA.csv', 'geoStatsB.csv', 'isotopeStatsA.csv', 'isotopeStatsB.csv', 'teleStatsA.csv', 'teleStatsB.csv', 'NARRStatsA.csv', 'NARRStatsB.csv']
    geoStats = []
    isotopeStats = []
    teleStats = []
    NARRStats = []
    for file in files_to_load:
        path = pathCharm + r'\isoP_Code\Stats\\' + file
        if 'geo' in file:
            geoStats.append(np.genfromtxt(path, delimiter=',', encoding='utf-8-sig'))
        elif 'iso' in file:
            isotopeStats.append(np.genfromtxt(path, delimiter=',', encoding='utf-8-sig'))
        elif 'tele' in file:
            teleStats.append(np.genfromtxt(path, delimiter=',', encoding='utf-8-sig'))
        elif 'NARR' in file:
            NARRStats.append(np.genfromtxt(path, delimiter=',', encoding='utf-8-sig'))
    dataStats = [geoStats, isotopeStats, teleStats, NARRStats]
    #JG -- These are the same as the matlab code, but I have changed the names to be more pythonic
    #    I have also changed the way the data is stored, instead of cell arrays, it is a list of arrays
    numGrids = len(outputNARR)

    print("Standardizing by Season!")

    for k in range(numGrids):
        #Transform necessary climate variables!
        #At this time, hgt_tropo, pres_tropo, PWAT and apcp all require natural log transformations
        outputNARR[k][:, 11] = np.log(outputNARR[k][:, 11])
        outputNARR[k][:, 12] = np.log(outputNARR[k][:, 12])
        outputNARR[k][:, 15] = np.log(outputNARR[k][:, 15])
        outputNARR[k][:, 16] = np.log(outputNARR[k][:, 16])
    
    #Initializing variables
    outputNARR_DJF = []
    outputNARR_MAM = []
    outputNARR_JJA = []
    outputNARR_SON = []
    tele_DJF = []
    geo_DJF = []
    tele_MAM = []
    geo_MAM = []
    tele_JJA = []
    geo_JJA = []
    tele_SON = []
    geo_SON = []

    #Separate the data into seasonal cells and standardize the data for each season
    #This is done through slicing rather than deleting as was the case in the Matlab code

    #Slicing the arrays into seasonal cells
    for k in range(numGrids):
        #Sorting the Output NARR data into seasonal cells
        indexCondition = np.logical_or.reduce([outputNARR[k][:, 1] == 1, outputNARR[k][:, 1] == 2, outputNARR[k][:, 1] == 12])
        arrayIndex = np.where(indexCondition[:, np.newaxis], outputNARR[k], np.nan)
        outputNARR_DJF.append(arrayIndex[~np.isnan(arrayIndex).any(axis=1)])
        indexCondition = np.logical_or.reduce([outputNARR[k][:, 1] == 3, outputNARR[k][:, 1] == 4, outputNARR[k][:, 1] == 5])
        arrayIndex = np.where(indexCondition[:, np.newaxis], outputNARR[k], np.nan)
        outputNARR_MAM.append(arrayIndex[~np.isnan(arrayIndex).any(axis=1)])
        indexCondition = np.logical_or.reduce([outputNARR[k][:, 1] == 6, outputNARR[k][:, 1] == 7, outputNARR[k][:, 1] == 8])
        arrayIndex = np.where(indexCondition[:, np.newaxis], outputNARR[k], np.nan)
        outputNARR_JJA.append(arrayIndex[~np.isnan(arrayIndex).any(axis=1)])
        indexCondition = np.logical_or.reduce([outputNARR[k][:, 1] == 9, outputNARR[k][:, 1] == 10, outputNARR[k][:, 1] == 11])
        arrayIndex = np.where(indexCondition[:, np.newaxis], outputNARR[k], np.nan)
        outputNARR_SON.append(arrayIndex[~np.isnan(arrayIndex).any(axis=1)])

        #Sorting the teleconnection data into seasonal cells
        indexCondition = np.logical_or.reduce([tele[k][:, 1] == 1, tele[k][:, 1] == 2, tele[k][:, 1] == 12])
        arrayIndex = np.where(indexCondition[:, np.newaxis], tele[k], np.nan)
        tele_DJF.append(arrayIndex[~np.isnan(arrayIndex).any(axis=1)])
        indexCondition = np.logical_or.reduce([tele[k][:, 1] == 3, tele[k][:, 1] == 4, tele[k][:, 1] == 5])
        arrayIndex = np.where(indexCondition[:, np.newaxis], tele[k], np.nan)
        tele_MAM.append(arrayIndex[~np.isnan(arrayIndex).any(axis=1)])
        indexCondition = np.logical_or.reduce([tele[k][:, 1] == 6, tele[k][:, 1] == 7, tele[k][:, 1] == 8])
        arrayIndex = np.where(indexCondition[:, np.newaxis], tele[k], np.nan)
        tele_JJA.append(arrayIndex[~np.isnan(arrayIndex).any(axis=1)])
        indexCondition = np.logical_or.reduce([tele[k][:, 1] == 9, tele[k][:, 1] == 10, tele[k][:, 1] == 11])
        arrayIndex = np.where(indexCondition[:, np.newaxis], tele[k], np.nan)
        tele_SON.append(arrayIndex[~np.isnan(arrayIndex).any(axis=1)])

        #Sorting the geo data into seasonal cells
        #JG -- given the different format of the geo data, I have to do this differently as it does not have a date associated with it
        #   I have to use the index of the outputNARR to sort the geo data
        indexCondition = np.logical_or.reduce([outputNARR[k][:, 1] == 1, outputNARR[k][:, 1] == 2, outputNARR[k][:, 1] == 12])
        arrayIndex = np.where(indexCondition[:, np.newaxis], dataGEO[k], np.nan)
        geo_DJF.append(arrayIndex[~np.isnan(arrayIndex).any(axis=1)])
        indexCondition = np.logical_or.reduce([outputNARR[k][:, 1] == 3, outputNARR[k][:, 1] == 4, outputNARR[k][:, 1] == 5])
        arrayIndex = np.where(indexCondition[:, np.newaxis], dataGEO[k], np.nan)
        geo_MAM.append(arrayIndex[~np.isnan(arrayIndex).any(axis=1)])
        indexCondition = np.logical_or.reduce([outputNARR[k][:, 1] == 6, outputNARR[k][:, 1] == 7, outputNARR[k][:, 1] == 8])
        arrayIndex = np.where(indexCondition[:, np.newaxis], dataGEO[k], np.nan)
        geo_JJA.append(arrayIndex[~np.isnan(arrayIndex).any(axis=1)])
        indexCondition = np.logical_or.reduce([outputNARR[k][:, 1] == 9, outputNARR[k][:, 1] == 10, outputNARR[k][:, 1] == 11])
        arrayIndex = np.where(indexCondition[:, np.newaxis], dataGEO[k], np.nan)
        geo_SON.append(arrayIndex[~np.isnan(arrayIndex).any(axis=1)])
    
    #Initializing the standardized seasonal cells
    allCellData_DJF = []
    allCellData_MAM = []
    allCellData_JJA = []
    allCellData_SON = []

    teleSTD_DJF = [np.zeros((tele_DJF[0].shape)) for _ in range(numGrids)]
    teleSTD_MAM = [np.zeros((tele_MAM[0].shape)) for _ in range(numGrids)]
    teleSTD_JJA = [np.zeros((tele_JJA[0].shape)) for _ in range(numGrids)]
    teleSTD_SON = [np.zeros((tele_SON[0].shape)) for _ in range(numGrids)]

    geoSTD_DJF = [np.zeros((geo_DJF[0].shape)) for _ in range(numGrids)]
    geoSTD_MAM = [np.zeros((geo_MAM[0].shape)) for _ in range(numGrids)]
    geoSTD_JJA = [np.zeros((geo_JJA[0].shape)) for _ in range(numGrids)]
    geoSTD_SON = [np.zeros((geo_SON[0].shape)) for _ in range(numGrids)]

    narrSTD_DJF = [np.zeros((outputNARR_DJF[0].shape)) for _ in range(numGrids)]
    narrSTD_MAM = [np.zeros((outputNARR_MAM[0].shape)) for _ in range(numGrids)]
    narrSTD_JJA = [np.zeros((outputNARR_JJA[0].shape)) for _ in range(numGrids)]
    narrSTD_SON = [np.zeros((outputNARR_SON[0].shape)) for _ in range(numGrids)]

    for k in range(numGrids):
        #Standardizing the NARR Data
        colsNARR = outputNARR_DJF[0].shape[1]
        narrSTD_DJF[k][:, 0:3] = outputNARR_DJF[k][:, 0:3]
        narrSTD_MAM[k][:, 0:3] = outputNARR_MAM[k][:, 0:3]
        narrSTD_JJA[k][:, 0:3] = outputNARR_JJA[k][:, 0:3]
        narrSTD_SON[k][:, 0:3] = outputNARR_SON[k][:, 0:3]
        for m in range(3, colsNARR):
            adjM = m-3
            narrSTD_DJF[k][:, m] = (outputNARR_DJF[k][:, m] - NARRStats[0][adjM, 0])/NARRStats[1][adjM, 0]
            narrSTD_MAM[k][:, m] = (outputNARR_MAM[k][:, m] - NARRStats[0][adjM, 1])/NARRStats[1][adjM, 1]
            narrSTD_JJA[k][:, m] = (outputNARR_JJA[k][:, m] - NARRStats[0][adjM, 2])/NARRStats[1][adjM, 2]
            narrSTD_SON[k][:, m] = (outputNARR_SON[k][:, m] - NARRStats[0][adjM, 3])/NARRStats[1][adjM, 3]
        #Standardizing the teleconnection data
        colsTele = tele_DJF[0].shape[1]
        teleSTD_DJF[k][:, 0:2] = tele_DJF[k][:, 0:2]
        teleSTD_MAM[k][:, 0:2] = tele_MAM[k][:, 0:2]
        teleSTD_JJA[k][:, 0:2] = tele_JJA[k][:, 0:2]
        teleSTD_SON[k][:, 0:2] = tele_SON[k][:, 0:2]
        for m in range(2, colsTele):
            adjM = m-2
            teleSTD_DJF[k][:, m] = (tele_DJF[k][:, m] - teleStats[0][adjM, 0])/teleStats[1][adjM, 0]
            teleSTD_MAM[k][:, m] = (tele_MAM[k][:, m] - teleStats[0][adjM, 1])/teleStats[1][adjM, 1]
            teleSTD_JJA[k][:, m] = (tele_JJA[k][:, m] - teleStats[0][adjM, 2])/teleStats[1][adjM, 2]
            teleSTD_SON[k][:, m] = (tele_SON[k][:, m] - teleStats[0][adjM, 3])/teleStats[1][adjM, 3]

        #Standardizing the geo data
        colsGeo = geo_DJF[0].shape[1]
        for m in range(colsGeo):
            geoSTD_DJF[k][:, m] = (geo_DJF[k][:, m] - geoStats[0][m, 0])/geoStats[1][m, 0]
            geoSTD_MAM[k][:, m] = (geo_MAM[k][:, m] - geoStats[0][m, 1])/geoStats[1][m, 1]
            geoSTD_JJA[k][:, m] = (geo_JJA[k][:, m] - geoStats[0][m, 2])/geoStats[1][m, 2]
            geoSTD_SON[k][:, m] = (geo_SON[k][:, m] - geoStats[0][m, 3])/geoStats[1][m, 3]
    
        #Combining the standardized seasonal cells into one cell
        allCellData_DJF.append(np.column_stack((narrSTD_DJF[k][:, 0:3], geoSTD_DJF[k][:, 0:3], narrSTD_DJF[k][:, 3:24],  teleSTD_DJF[k][:, 2:9], geoSTD_DJF[k][:, 3:5])))
        allCellData_MAM.append(np.column_stack((narrSTD_MAM[k][:, 0:3], geoSTD_MAM[k][:, 0:3], narrSTD_MAM[k][:, 3:24],  teleSTD_MAM[k][:, 2:9], geoSTD_MAM[k][:, 3:5])))
        allCellData_JJA.append(np.column_stack((narrSTD_JJA[k][:, 0:3], geoSTD_JJA[k][:, 0:3], narrSTD_JJA[k][:, 3:24],  teleSTD_JJA[k][:, 2:9], geoSTD_JJA[k][:, 3:5])))
        allCellData_SON.append(np.column_stack((narrSTD_SON[k][:, 0:3], geoSTD_SON[k][:, 0:3], narrSTD_SON[k][:, 3:24],  teleSTD_SON[k][:, 2:9], geoSTD_SON[k][:, 3:5])))

    #Combining the cell data into one array for each season
    length, width = allCellData_DJF[0].shape
    allData_DJF = np.zeros((length*numGrids, width))
    allData_MAM = np.zeros((length*numGrids, width))
    allData_JJA = np.zeros((length*numGrids, width))
    allData_SON = np.zeros((length*numGrids, width))

    #Stacking the data into one array
    for k in range(numGrids):
        allData_DJF[length*k:length*(k+1), :] = allCellData_DJF[k]
        allData_MAM[length*k:length*(k+1), :] = allCellData_MAM[k]
        allData_JJA[length*k:length*(k+1), :] = allCellData_JJA[k]
        allData_SON[length*k:length*(k+1), :] = allCellData_SON[k]
    print("NARR, geographic, and teleconnection data standardized by season!")
    
    #Sort Data and put into corresponding KPN zone
    #DJF
    sortKPN_DJF = allData_DJF[allData_DJF[:, 32].argsort(kind='mergesort')]
    insertKPN_DJF = np.array([allData_DJF[:, 32].min()])
    colKPN_DJF = np.append(insertKPN_DJF, sortKPN_DJF[:, 32])
    lengthKPN = len(colKPN_DJF)
    diffKPN = sortKPN_DJF[:, 32]-colKPN_DJF[0:lengthKPN-1]
    indexKPN_DJF = np.nonzero(diffKPN) 
    numPts = np.array(len(sortKPN_DJF))
    indexKPN_DJF = np.append(indexKPN_DJF, numPts)
    numIndex = len(indexKPN_DJF)
    allDataKPN_DJF = [[] for _ in range(5)]

    for i in range(numIndex):
        if i == 0 and i != numIndex-1:
            if np.any(sortKPN_DJF[0:indexKPN_DJF[i]-1, 32] == 35):
                allDataKPN_DJF[0] = sortKPN_DJF[0:indexKPN_DJF[0]-1, :]
            elif np.any(sortKPN_DJF[0:indexKPN_DJF[i]-1, 32] == 42):
                allDataKPN_DJF[1] = sortKPN_DJF[0:indexKPN_DJF[0]-1, :]
            elif np.any(sortKPN_DJF[0:indexKPN_DJF[i]-1, 32] == 43):
                allDataKPN_DJF[2] = sortKPN_DJF[0:indexKPN_DJF[0]-1, :]
            elif np.any(sortKPN_DJF[0:indexKPN_DJF[i]-1, 32] == 47):
                allDataKPN_DJF[3] = sortKPN_DJF[0:indexKPN_DJF[0]-1, :]
            elif np.any(sortKPN_DJF[0:indexKPN_DJF[i]-1, 32] == 62):
                allDataKPN_DJF[4] = sortKPN_DJF[0:indexKPN_DJF[0]-1, :]
        elif i == 0 and i == numIndex-1:
            if np.any(sortKPN_DJF[0:indexKPN_DJF[i], 32] == 35):
                allDataKPN_DJF[0] = sortKPN_DJF[0:indexKPN_DJF[0], :]
            elif np.any(sortKPN_DJF[0:indexKPN_DJF[i], 32] == 42):
                allDataKPN_DJF[1] = sortKPN_DJF[0:indexKPN_DJF[0], :]
            elif np.any(sortKPN_DJF[0:indexKPN_DJF[i], 32] == 43):
                allDataKPN_DJF[2] = sortKPN_DJF[0:indexKPN_DJF[0], :]
            elif np.any(sortKPN_DJF[0:indexKPN_DJF[i], 32] == 47):
                allDataKPN_DJF[3] = sortKPN_DJF[0:indexKPN_DJF[0], :]
            elif np.any(sortKPN_DJF[0:indexKPN_DJF[i], 32] == 62):
                allDataKPN_DJF[4] = sortKPN_DJF[0:indexKPN_DJF[0], :]
        elif i >= 1 and i != numIndex-1:
            if np.any(sortKPN_DJF[indexKPN_DJF[i-1]:indexKPN_DJF[i]-1, 32] == 35):
                allDataKPN_DJF[0] = sortKPN_DJF[indexKPN_DJF[i-1]:indexKPN_DJF[i]-1, :]
            elif np.any(sortKPN_DJF[indexKPN_DJF[i-1]:indexKPN_DJF[i]-1, 32] == 42):
                allDataKPN_DJF[1] = sortKPN_DJF[indexKPN_DJF[i-1]:indexKPN_DJF[i]-1, :]
            elif np.any(sortKPN_DJF[indexKPN_DJF[i-1]:indexKPN_DJF[i]-1, 32] == 43):
                allDataKPN_DJF[2] = sortKPN_DJF[indexKPN_DJF[i-1]:indexKPN_DJF[i]-1, :]
            elif np.any(sortKPN_DJF[indexKPN_DJF[i-1]:indexKPN_DJF[i]-1, 32] == 47):
                allDataKPN_DJF[3] = sortKPN_DJF[indexKPN_DJF[i-1]:indexKPN_DJF[i]-1, :]
            elif np.any(sortKPN_DJF[indexKPN_DJF[i-1]:indexKPN_DJF[i]-1, 32] == 62):
                allDataKPN_DJF[4] = sortKPN_DJF[indexKPN_DJF[i-1]:indexKPN_DJF[i]-1, :]
        elif i >= 1 and i == numIndex-1:
            if np.any(sortKPN_DJF[indexKPN_DJF[i-1]:indexKPN_DJF[i], 32] == 35):
                allDataKPN_DJF[0] = sortKPN_DJF[indexKPN_DJF[i-1]:indexKPN_DJF[i], :]
            elif np.any(sortKPN_DJF[indexKPN_DJF[i-1]:indexKPN_DJF[i], 32] == 42):
                allDataKPN_DJF[1] = sortKPN_DJF[indexKPN_DJF[i-1]:indexKPN_DJF[i], :]
            elif np.any(sortKPN_DJF[indexKPN_DJF[i-1]:indexKPN_DJF[i], 32] == 43):
                allDataKPN_DJF[2] = sortKPN_DJF[indexKPN_DJF[i-1]:indexKPN_DJF[i], :]
            elif np.any(sortKPN_DJF[indexKPN_DJF[i-1]:indexKPN_DJF[i], 32] == 47):
                allDataKPN_DJF[3] = sortKPN_DJF[indexKPN_DJF[i-1]:indexKPN_DJF[i], :]
            elif np.any(sortKPN_DJF[indexKPN_DJF[i-1]:indexKPN_DJF[i], 32] == 62):
                allDataKPN_DJF[4] = sortKPN_DJF[indexKPN_DJF[i-1]:indexKPN_DJF[i], :]
    
    #MAM
    sortKPN_MAM = allData_MAM[allData_MAM[:, 32].argsort(kind='mergesort')]
    insertKPN_MAM = np.array([allData_MAM[:, 32].min()])
    colKPN_MAM = np.append(insertKPN_MAM, sortKPN_MAM[:, 32])
    lengthKPN = len(colKPN_MAM)
    diffKPN = sortKPN_MAM[:, 32]-colKPN_MAM[0:lengthKPN-1]
    indexKPN_MAM = np.nonzero(diffKPN)
    numPts = np.array(len(sortKPN_MAM))
    indexKPN_MAM = np.append(indexKPN_MAM, numPts)
    numIndex = len(indexKPN_MAM)
    allDataKPN_MAM = [[] for _ in range(5)]

    for i in range(numIndex):
        if i == 0 and i!= numIndex-1:
            if np.any(sortKPN_MAM[0:indexKPN_MAM[i]-1, 32]) == 35:
                allDataKPN_MAM[0] = sortKPN_MAM[0:indexKPN_MAM[0]-1, :]
            elif np.any(sortKPN_MAM[0:indexKPN_MAM[i]-1, 32]) == 42:
                allDataKPN_MAM[1] = sortKPN_MAM[0:indexKPN_MAM[0]-1, :]
            elif np.any(sortKPN_MAM[0:indexKPN_MAM[i]-1, 32]) == 43:
                allDataKPN_MAM[2] = sortKPN_MAM[0:indexKPN_MAM[0]-1, :]
            elif np.any(sortKPN_MAM[0:indexKPN_MAM[i]-1, 32]) == 47:
                allDataKPN_MAM[3] = sortKPN_MAM[0:indexKPN_MAM[0]-1, :]
            elif np.any(sortKPN_MAM[0:indexKPN_MAM[i]-1, 32]) == 62:
                allDataKPN_MAM[4] = sortKPN_MAM[0:indexKPN_MAM[0]-1, :]
        elif i == 0 and i == numIndex-1:
            if np.any(sortKPN_MAM[0:indexKPN_MAM[i], 32] == 35):
                allDataKPN_MAM[0] = sortKPN_MAM[0:indexKPN_MAM[0], :]
            elif np.any(sortKPN_MAM[0:indexKPN_MAM[i], 32] == 42):
                allDataKPN_MAM[1] = sortKPN_MAM[0:indexKPN_MAM[0], :]
            elif np.any(sortKPN_MAM[0:indexKPN_MAM[i], 32] == 43):
                allDataKPN_MAM[2] = sortKPN_MAM[0:indexKPN_MAM[0], :]
            elif np.any(sortKPN_MAM[0:indexKPN_MAM[i], 32] == 47):
                allDataKPN_MAM[3] = sortKPN_MAM[0:indexKPN_MAM[0], :]
            elif np.any(sortKPN_MAM[0:indexKPN_MAM[i], 32] == 62):
                allDataKPN_MAM[4] = sortKPN_MAM[0:indexKPN_MAM[0], :]
        elif i >= 1 and i != numIndex-1:
            if np.any(sortKPN_MAM[indexKPN_MAM[i-1]:indexKPN_MAM[i]-1, 32] == 35):
                allDataKPN_MAM[0] = sortKPN_MAM[indexKPN_MAM[i-1]:indexKPN_MAM[i]-1, :]
            elif np.any(sortKPN_MAM[indexKPN_MAM[i-1]:indexKPN_MAM[i]-1, 32] == 42):
                allDataKPN_MAM[1] = sortKPN_MAM[indexKPN_MAM[i-1]:indexKPN_MAM[i]-1, :]
            elif np.any(sortKPN_MAM[indexKPN_MAM[i-1]:indexKPN_MAM[i]-1, 32] == 43):
                allDataKPN_MAM[2] = sortKPN_MAM[indexKPN_MAM[i-1]:indexKPN_MAM[i]-1, :]
            elif np.any(sortKPN_MAM[indexKPN_MAM[i-1]:indexKPN_MAM[i]-1, 32] == 47):
                allDataKPN_MAM[3] = sortKPN_MAM[indexKPN_MAM[i-1]:indexKPN_MAM[i]-1, :]
            elif np.any(sortKPN_MAM[indexKPN_MAM[i-1]:indexKPN_MAM[i]-1, 32] == 62):
                allDataKPN_MAM[4] = sortKPN_MAM[indexKPN_MAM[i-1]:indexKPN_MAM[i]-1, :]
        elif i >= 1 and i == numIndex-1:
            if np.any(sortKPN_MAM[indexKPN_MAM[i-1]:indexKPN_MAM[i], 32] == 35):
                allDataKPN_MAM[0] = sortKPN_MAM[indexKPN_MAM[i-1]:indexKPN_MAM[i], :]
            elif np.any(sortKPN_MAM[indexKPN_MAM[i-1]:indexKPN_MAM[i], 32] == 42):
                allDataKPN_MAM[1] = sortKPN_MAM[indexKPN_MAM[i-1]:indexKPN_MAM[i], :]
            elif np.any(sortKPN_MAM[indexKPN_MAM[i-1]:indexKPN_MAM[i], 32] == 43):
                allDataKPN_MAM[2] = sortKPN_MAM[indexKPN_MAM[i-1]:indexKPN_MAM[i], :]
            elif np.any(sortKPN_MAM[indexKPN_MAM[i-1]:indexKPN_MAM[i], 32] == 47):
                allDataKPN_MAM[3] = sortKPN_MAM[indexKPN_MAM[i-1]:indexKPN_MAM[i], :]
            elif np.any(sortKPN_MAM[indexKPN_MAM[i-1]:indexKPN_MAM[i], 32] == 62):
                allDataKPN_MAM[4] = sortKPN_MAM[indexKPN_MAM[i-1]:indexKPN_MAM[i], :]
    
    #JJA
    sortKPN_JJA = allData_JJA[allData_JJA[:, 32].argsort(kind='mergesort')]
    insertKPN_JJA = np.array([allData_JJA[:, 32].min()])
    colKPN_JJA = np.append(insertKPN_JJA, sortKPN_JJA[:, 32])
    lengthKPN = len(colKPN_JJA)
    diffKPN = sortKPN_JJA[:, 32]-colKPN_JJA[0:lengthKPN-1]
    indexKPN_JJA = np.nonzero(diffKPN)
    numPts = np.array(len(sortKPN_JJA))
    indexKPN_JJA = np.append(indexKPN_JJA, numPts)
    numIndex = len(indexKPN_JJA)
    allDataKPN_JJA = [[] for _ in range(5)]

    for i in range(numIndex):
        if i == 0 and i!= numIndex-1:
            if np.any(sortKPN_JJA[0:indexKPN_JJA[i]-1, 32]) == 35:
                allDataKPN_JJA[0] = sortKPN_JJA[0:indexKPN_JJA[0]-1, :]
            elif np.any(sortKPN_JJA[0:indexKPN_JJA[i]-1, 32]) == 42:
                allDataKPN_JJA[1] = sortKPN_JJA[0:indexKPN_JJA[0]-1, :]
            elif np.any(sortKPN_JJA[0:indexKPN_JJA[i]-1, 32]) == 43:
                allDataKPN_JJA[2] = sortKPN_JJA[0:indexKPN_JJA[0]-1, :]
            elif np.any(sortKPN_JJA[0:indexKPN_JJA[i]-1, 32]) == 47:
                allDataKPN_JJA[3] = sortKPN_JJA[0:indexKPN_JJA[0]-1, :]
            elif np.any(sortKPN_JJA[0:indexKPN_JJA[i]-1, 32]) == 62:
                allDataKPN_JJA[4] = sortKPN_JJA[0:indexKPN_JJA[0]-1, :]
        elif i == 0 and i == numIndex-1:
            if np.any(sortKPN_JJA[0:indexKPN_JJA[i], 32] == 35):
                allDataKPN_JJA[0] = sortKPN_JJA[0:indexKPN_JJA[0], :]
            elif np.any(sortKPN_JJA[0:indexKPN_JJA[i], 32] == 42):
                allDataKPN_JJA[1] = sortKPN_JJA[0:indexKPN_JJA[0], :]
            elif np.any(sortKPN_JJA[0:indexKPN_JJA[i], 32] == 43):
                allDataKPN_JJA[2] = sortKPN_JJA[0:indexKPN_JJA[0], :]
            elif np.any(sortKPN_JJA[0:indexKPN_JJA[i], 32] == 47):
                allDataKPN_JJA[3] = sortKPN_JJA[0:indexKPN_JJA[0], :]
            elif np.any(sortKPN_JJA[0:indexKPN_JJA[i], 32] == 62):
                allDataKPN_JJA[4] = sortKPN_JJA[0:indexKPN_JJA[0], :]
        elif i >= 1 and i != numIndex-1:
            if np.any(sortKPN_JJA[indexKPN_JJA[i-1]:indexKPN_JJA[i]-1, 32] == 35):
                allDataKPN_JJA[0] = sortKPN_JJA[indexKPN_JJA[i-1]:indexKPN_JJA[i]-1, :]
            elif np.any(sortKPN_JJA[indexKPN_JJA[i-1]:indexKPN_JJA[i]-1, 32] == 42):
                allDataKPN_JJA[1] = sortKPN_JJA[indexKPN_JJA[i-1]:indexKPN_JJA[i]-1, :]
            elif np.any(sortKPN_JJA[indexKPN_JJA[i-1]:indexKPN_JJA[i]-1, 32] == 43):
                allDataKPN_JJA[2] = sortKPN_JJA[indexKPN_JJA[i-1]:indexKPN_JJA[i]-1, :]
            elif np.any(sortKPN_JJA[indexKPN_JJA[i-1]:indexKPN_JJA[i]-1, 32] == 47):
                allDataKPN_JJA[3] = sortKPN_JJA[indexKPN_JJA[i-1]:indexKPN_JJA[i]-1, :]
            elif np.any(sortKPN_JJA[indexKPN_JJA[i-1]:indexKPN_JJA[i]-1, 32] == 62):
                allDataKPN_JJA[4] = sortKPN_JJA[indexKPN_JJA[i-1]:indexKPN_JJA[i]-1, :]
        elif i >= 1 and i == numIndex-1:
            if np.any(sortKPN_JJA[indexKPN_JJA[i-1]:indexKPN_JJA[i], 32] == 35):
                allDataKPN_JJA[0] = sortKPN_JJA[indexKPN_JJA[i-1]:indexKPN_JJA[i], :]
            elif np.any(sortKPN_JJA[indexKPN_JJA[i-1]:indexKPN_JJA[i], 32] == 42):
                allDataKPN_JJA[1] = sortKPN_JJA[indexKPN_JJA[i-1]:indexKPN_JJA[i], :]
            elif np.any(sortKPN_JJA[indexKPN_JJA[i-1]:indexKPN_JJA[i], 32] == 43):
                allDataKPN_JJA[2] = sortKPN_JJA[indexKPN_JJA[i-1]:indexKPN_JJA[i], :]
            elif np.any(sortKPN_JJA[indexKPN_JJA[i-1]:indexKPN_JJA[i], 32] == 47):
                allDataKPN_JJA[3] = sortKPN_JJA[indexKPN_JJA[i-1]:indexKPN_JJA[i], :]
            elif np.any(sortKPN_JJA[indexKPN_JJA[i-1]:indexKPN_JJA[i], 32] == 62):
                allDataKPN_JJA[4] = sortKPN_JJA[indexKPN_JJA[i-1]:indexKPN_JJA[i], :]
    
    #SON
    sortKPN_SON = allData_SON[allData_SON[:, 32].argsort(kind='mergesort')]
    insertKPN_SON = np.array([allData_SON[:, 32].min()])
    colKPN_SON = np.append(insertKPN_SON, sortKPN_SON[:, 32])
    lengthKPN = len(colKPN_SON)
    diffKPN = sortKPN_SON[:, 32]-colKPN_SON[0:lengthKPN-1]
    indexKPN_SON = np.nonzero(diffKPN)
    numPts = np.array(len(sortKPN_SON))
    indexKPN_SON = np.append(indexKPN_SON, numPts)
    numIndex = len(indexKPN_SON)
    allDataKPN_SON = [[] for _ in range(5)]

    for i in range(numIndex):
        if i == 0 and i!= numIndex-1:
            if np.any(sortKPN_SON[0:indexKPN_SON[i]-1, 32]) == 35:
                allDataKPN_SON[0] = sortKPN_SON[0:indexKPN_SON[0]-1, :]
            elif np.any(sortKPN_SON[0:indexKPN_SON[i]-1, 32]) == 42:
                allDataKPN_SON[1] = sortKPN_SON[0:indexKPN_SON[0]-1, :]
            elif np.any(sortKPN_SON[0:indexKPN_SON[i]-1, 32]) == 43:
                allDataKPN_SON[2] = sortKPN_SON[0:indexKPN_SON[0]-1, :]
            elif np.any(sortKPN_SON[0:indexKPN_SON[i]-1, 32]) == 47:
                allDataKPN_SON[3] = sortKPN_SON[0:indexKPN_SON[0]-1, :]
            elif np.any(sortKPN_SON[0:indexKPN_SON[i]-1, 32]) == 62:
                allDataKPN_SON[4] = sortKPN_SON[0:indexKPN_SON[0]-1, :]
        elif i == 0 and i == numIndex-1:
            if np.any(sortKPN_SON[0:indexKPN_SON[i], 32] == 35):
                allDataKPN_SON[0] = sortKPN_SON[0:indexKPN_SON[0], :]
            elif np.any(sortKPN_SON[0:indexKPN_SON[i], 32] == 42):
                allDataKPN_SON[1] = sortKPN_SON[0:indexKPN_SON[0], :]
            elif np.any(sortKPN_SON[0:indexKPN_SON[i], 32] == 43):
                allDataKPN_SON[2] = sortKPN_SON[0:indexKPN_SON[0], :]
            elif np.any(sortKPN_SON[0:indexKPN_SON[i], 32] == 47):
                allDataKPN_SON[3] = sortKPN_SON[0:indexKPN_SON[0], :]
            elif np.any(sortKPN_SON[0:indexKPN_SON[i], 32] == 62):
                allDataKPN_SON[4] = sortKPN_SON[0:indexKPN_SON[0], :]
        elif i >= 1 and i != numIndex-1:
            if np.any(sortKPN_SON[indexKPN_SON[i-1]:indexKPN_SON[i]-1, 32] == 35):
                allDataKPN_SON[0] = sortKPN_SON[indexKPN_SON[i-1]:indexKPN_SON[i]-1, :]
            elif np.any(sortKPN_SON[indexKPN_SON[i-1]:indexKPN_SON[i]-1, 32] == 42):
                allDataKPN_SON[1] = sortKPN_SON[indexKPN_SON[i-1]:indexKPN_SON[i]-1, :]
            elif np.any(sortKPN_SON[indexKPN_SON[i-1]:indexKPN_SON[i]-1, 32] == 43):
                allDataKPN_SON[2] = sortKPN_SON[indexKPN_SON[i-1]:indexKPN_SON[i]-1, :]
            elif np.any(sortKPN_SON[indexKPN_SON[i-1]:indexKPN_SON[i]-1, 32] == 47):
                allDataKPN_SON[3] = sortKPN_SON[indexKPN_SON[i-1]:indexKPN_SON[i]-1, :]
            elif np.any(sortKPN_SON[indexKPN_SON[i-1]:indexKPN_SON[i]-1, 32] == 62):
                allDataKPN_SON[4] = sortKPN_SON[indexKPN_SON[i-1]:indexKPN_SON[i]-1, :]
        elif i >= 1 and i == numIndex-1:
            if np.any(sortKPN_SON[indexKPN_SON[i-1]:indexKPN_SON[i], 32] == 35):
                allDataKPN_SON[0] = sortKPN_SON[indexKPN_SON[i-1]:indexKPN_SON[i], :]
            elif np.any(sortKPN_SON[indexKPN_SON[i-1]:indexKPN_SON[i], 32] == 42):
                allDataKPN_SON[1] = sortKPN_SON[indexKPN_SON[i-1]:indexKPN_SON[i], :]
            elif np.any(sortKPN_SON[indexKPN_SON[i-1]:indexKPN_SON[i], 32] == 43):
                allDataKPN_SON[2] = sortKPN_SON[indexKPN_SON[i-1]:indexKPN_SON[i], :]
            elif np.any(sortKPN_SON[indexKPN_SON[i-1]:indexKPN_SON[i], 32] == 47):
                allDataKPN_SON[3] = sortKPN_SON[indexKPN_SON[i-1]:indexKPN_SON[i], :]
            elif np.any(sortKPN_SON[indexKPN_SON[i-1]:indexKPN_SON[i], 32] == 62):
                allDataKPN_SON[4] = sortKPN_SON[indexKPN_SON[i-1]:indexKPN_SON[i], :]
    
    #Combing all that data into one list
    dataAllKPN_seas = [allDataKPN_DJF, allDataKPN_MAM, allDataKPN_JJA, allDataKPN_SON]
    
    print("Data has been standardized and condensed into seasons!")
    return dataAllKPN_seas, dataStats

def isoP():
    path = input("What is the main directory for the model? ")
    startYear = input("What is the start year for the model? ")
    endYear = input("What is the end year for the model? ")
    #Extract the neame of the basin from the path name provided
    #JG - Struggled with the raw string input, needed this way to list correctly
    #path = input("What is the main directory for the model? ")
    splitPath = path.split("\\")
    basinName = splitPath[-1]

    #Read in the SHD file and create a grid of LAT LONGs for the basin
    #JG -- Used to create an XLS file in the isoP folder, now creates a CSV
    WFcoords = readSHD_File(path, basinName)

    #For the specified lat/long pairs, extract the full NARR time series
    #(currently 1979-2012) at each grid point.
    output, pathNARR, pathCharm = extract_NARR_timeseries(path, basinName)

    #Format the NARR data to be in the same format as the crop data
    outputNARR = NARR_format_timeseries_basin(output, pathNARR, startYear, endYear)

    #Extract the elevation and  specify the zones for the KPN, 2Zone, and 1Zone
    #regionalizations for the specified WATFLOOD grid
    dataGEO = extract_GIS_info(startYear, endYear, WFcoords, pathCharm)

    #Extract teleconnection indices for the specified period of years
    tele = extract_tele_timeseries_basin(WFcoords, pathCharm, startYear, endYear)

    #Standaize the data (for model stability), and condense data sources into one cell
    # for each of the three regionalization
    dataAllKPN_seas, dataStats = all_data_format_condense(outputNARR, dataGEO, tele, pathCharm)

    #Simulate th kpn data
    #stackPI, binaryPI = Simulate_Kpn(dataAllKPN_seas, dataStats, pathCharm)

isoP()