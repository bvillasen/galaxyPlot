import sys, time, os
import numpy as np
import matplotlib.pylab as plt
import h5py as h5
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray

#Add Modules from other directories
currentDirectory = os.getcwd()
parentDirectory = currentDirectory[:currentDirectory.rfind("/")]
toolsDirectory = parentDirectory + "/tools"
haloFinderDirectory = parentDirectory + '/haloFinder'
dataDirectory = currentDirectory[:currentDirectory.find("pyCUDA")] + "extras/haloFinder/data/"
volumeRenderDirectory = parentDirectory + "/volumeRender"
sys.path.extend( [toolsDirectory, haloFinderDirectory, dataDirectory, volumeRenderDirectory] )
from cudaTools import setCudaDevice, getFreeMemory, gpuArray3DtocudaArray
from tools import *
import volumeRender

nPoints = 128
useDevice = None
for option in sys.argv:
  if option == "128" or option == "256": nPoints = int(option)
  if option.find("device=") != -1: useDevice = int(option[-1]) 
 

import readsnapHDF5
snapname = dataDirectory + 'L25n128/output/snapdir_135/snap_135'
box_size = 25000.0  # kpc
particle_resolution = 128
linking_length = 0.2 * box_size / particle_resolution

# Read data
start = time.time()
print 'Reading data...'
#0 = gas particle, 1 = dark matter (DM), 4 = star particles
posParticles_dm = readsnapHDF5.read_block(snapname, "POS ", parttype=1)
posParticles_gas = readsnapHDF5.read_block(snapname, "POS ", parttype=0)
posParticles_star = readsnapHDF5.read_block(snapname, "POS ", parttype=4)
nParticles = len(posParticles_dm)
massParticles_dm = readsnapHDF5.read_block(snapname, "MASS", parttype=1)
massParticles_gas = readsnapHDF5.read_block(snapname, "MASS", parttype=0)
massParticles_star = readsnapHDF5.read_block(snapname, "MASS", parttype=4)
dm_particle_mass = massParticles_dm[0]
print 'Time:', time.time() - start, "\n"

posAll = { "dm":posParticles_dm, "gas":posParticles_gas, "star":posParticles_star }
massAll = { "dm":massParticles_dm, "gas":massParticles_gas, "star":massParticles_star }


nWidth = nPoints
nHeight = nPoints
nDepth = nPoints
nData = nWidth*nHeight*nDepth

def createData( particleType, nPoints ):
  nWidth = nPoints
  nHeight = nPoints
  nDepth = nPoints
  nData = nWidth*nHeight*nDepth

  dx = box_size/nWidth
  dy = box_size/nHeight
  dz = box_size/nDepth
  delta = np.array([ dx, dy, dz ])
  nParticles = len(posAll[particleType])
  #####################################################################
  #Satart Simulation
  print ""
  print "Starting simulation"
  #if cudaP == "double": print "Using double precision"
  print " nParticles: ", nParticles
  print " nPoints: ", nPoints
  print " type: ", particleType
  start = time.time()
  density_h = np.zeros( nData, dtype=np.float32 )
  
  idx3D = ( posAll[particleType] // delta ).astype(np.int32)
  for i in range( nParticles ):
    density_h[idx3D[i][0] + idx3D[i][1]*nWidth + idx3D[i][2]*nWidth*nHeight] += massAll[particleType][i]
  #density_h = np.log10( density_h +1 )
  #density_h = np.log10( density_h +1 )
  #maxDensity = density_h.max()
  #density_h /= maxDensity
  
  dataFile.create_dataset( particleType + "_density_" + str(nPoints), data=density_h)  

  #density_d.set( density_h )
  print 'Time:', time.time() - start, "\n"
def saveDensity():
  dataFile = h5.File("density.hdf5",'w')
  for particleType in ["dm", "gas", "star"]:
    for nPoints in [128, 256]:
      createData( particleType, nPoints )
  dataFile.close()

def loadDensity():
  start = time.time()
  print 'Loading density...'
  dataFile = h5.File( 'density.hdf5' ,'r')
  density_dm_128 = dataFile.get("dm_density_128")[...]
  density_dm_256 = dataFile.get("dm_density_256")[...]
  density_gas_128 = dataFile.get("gas_density_128")[...]
  density_gas_256 = dataFile.get("gas_density_256")[...]
  density_star_128 = dataFile.get("star_density_128")[...]
  density_star_256 = dataFile.get("star_density_256")[...]
  dataFile.close()
  densityAll = {128:{"dm": density_dm_128, "gas": density_gas_128, "star":density_star_128},
		256:{"dm": density_dm_256, "gas": density_gas_256, "star":density_star_256} }
  print 'Time:', time.time() - start, "\n"
  return densityAll

if nPoints == 128: densityAll = loadDensity()[128]
if nPoints == 256: densityAll = loadDensity()[256]


























#set thread grid for CUDA kernels
block_size_x, block_size_y, block_size_z = 8,8,8   #hardcoded, tune to your needs
gridx = nWidth // block_size_x + 1 * ( nWidth % block_size_x != 0 )
gridy = nHeight // block_size_y + 1 * ( nHeight % block_size_y != 0 )
gridz = nDepth // block_size_z + 1 * ( nDepth % block_size_z != 0 )
grid3D = (gridx, gridy, gridz)
block3D = (block_size_x, block_size_y, block_size_z)




#Initialize openGL
volumeRender.nWidth = nPoints
volumeRender.nHeight = nPoints
volumeRender.nDepth = nPoints
volumeRender.initGL()    
#initialize pyCUDA context 
cudaDevice = setCudaDevice(devN=useDevice, usingAnimation=True )

#Read and compile CUDA code
print "\nCompiling CUDA code"
cudaCodeString_raw = open("CUDAdensity3D.cu", "r").read()
cudaCodeString = cudaCodeString_raw % { "THREADS_PER_BLOCK":block3D[0]*block3D[1]*block3D[2] }
cudaCode = SourceModule(cudaCodeString)
densityKernel = cudaCode.get_function("density_kernel" )
########################################################################
from pycuda.elementwise import ElementwiseKernel
########################################################################
floatToUchar = ElementwiseKernel(arguments="float *input, unsigned char *output",
				operation = "output[i] = (unsigned char) ( -255*(input[i]-1));")
########################################################################
def preparePlotData( inputData ):
  plotData = inputData
  plotData = np.log( plotData +1 )
  #plotData = np.log10( plotData +1 )
  #plotData /= plotData.max()
  return plotData
########################################################################
def sendToScreen( plotData ):
  floatToUchar( plotData, plotData_d )
  copyToScreenArray()
########################################################################
def stepFunction():
  sendToScreen( density_d )

########################################################################
  
  
#Initialize all gpu data
print "\nInitializing Data"
initialMemory = getFreeMemory( show=True )  
density_h = densityAll["dm"]
density_d = gpuarray.to_gpu( density_h )
#memory for plotting
plotData_d = gpuarray.to_gpu(np.zeros([nDepth, nHeight, nWidth], dtype = np.uint8))
volumeRender.plotData_dArray, copyToScreenArray = gpuArray3DtocudaArray( plotData_d )
finalMemory = getFreeMemory( show=False )
print " Total Global Memory Used: {0} Mbytes".format(float(initialMemory-finalMemory)/1e6) 


density_h = preparePlotData( density_h )
density_d.set(density_h)
#change volumeRender default step function for heat3D step function
volumeRender.stepFunc = stepFunction
#run volumeRender animation
volumeRender.animate()



