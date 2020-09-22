#include "vars.h"
#include "vars_gpu.h"

extern int raynuminc;
extern int raynumout;
extern int mynumray;
extern int mynumpix;

extern int *raysendstart;
extern int *rayrecvstart;
extern int *raysendcount;
extern int *rayrecvcount;

extern int *rayraystart;
extern int *rayrayind;
extern int *rayrecvlist;

extern double ftime;
extern double btime;
extern double fktime;
extern double frtime;
extern double bktime;
extern double brtime;
extern double aftime;
extern double abtime;

extern int numproj;
extern int numback;

extern int proj_rownztot;
extern int *proj_rowdispl;
extern int *proj_rowindex;
extern float *proj_rowvalue;
extern int proj_blocksize;
extern int proj_numblocks;
extern int proj_blocknztot;
extern int *proj_blockdispl;
extern int *proj_blockindex;
extern float *proj_blockvalue;
extern int proj_buffsize;
extern int *proj_buffdispl;
extern int proj_buffnztot;
extern int *proj_buffmap;
extern short *proj_buffindex;
extern float *proj_buffvalue;
extern int back_rownztot;
extern int *back_rowdispl;
extern int *back_rowindex;
extern float *back_rowvalue;
extern int back_blocksize;
extern int back_numblocks;
extern int back_blocknztot;
extern int *back_blockdispl;
extern int *back_blockindex;
extern float *back_blockvalue;
extern int back_buffsize;
extern int *back_buffdispl;
extern int back_buffnztot;
extern int *back_buffmap;
extern short *back_buffindex;
extern float *back_buffvalue;

int *proj_blockdispl_d;
int *proj_buffdispl_d;
int *proj_buffmap_d;
short *proj_buffindex_d;
float *proj_buffvalue_d;
int *back_blockdispl_d;
int *back_buffdispl_d;
int *back_buffmap_d;
short *back_buffindex_d;
float *back_buffvalue_d;

int *rayraystart_d;
int *rayrayind_d;
int *rayindray_d;

float *tomogram_d;
float *sinogram_d;
float *raypart_d;
float *raybuff_d;

extern float *raypart;
extern float *raybuff;

void setup_gpu(float **obj,float **gra, float **dir,float **mes,float **res,float **ray){

  int numproc;
  int myid;
  MPI_Comm_size(MPI_COMM_WORLD,&numproc);
  MPI_Comm_rank(MPI_COMM_WORLD,&myid);

  int device = myid;///8;
  printf("myid: %d device: %d\n",myid,device);
  cudaSetDevice(device);
  if(myid==0){
  int deviceCount;
  cudaGetDeviceCount(&deviceCount);
  printf("\n");
  printf("Device Count: %d\n",deviceCount);
  //for (int dev = 0; dev < deviceCount; dev++) {
    int dev = myid;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("Device %d name: %d\n",dev,deviceProp.name);
    printf("Computational Capabilities: %d, %d\n",deviceProp.major,deviceProp.minor);
    printf("Maximum global memory size: %lu\n",deviceProp.totalGlobalMem);
    printf("Maximum constant memory size: %d\n",deviceProp.totalConstMem);
    printf("Maximum shared memory size per block: %d\n",deviceProp.sharedMemPerBlock);
    printf("Maximum block dimensions: %dx%dx%d\n",deviceProp.maxThreadsDim[0],deviceProp.maxThreadsDim[1],deviceProp.maxThreadsDim[2]);
    printf("Maximum grid dimensions: %dx%dx%d\n",deviceProp.maxGridSize[0],deviceProp.maxGridSize[1],deviceProp.maxGridSize[2]);
    printf("Maximum threads per block: %d\n",deviceProp.maxThreadsPerBlock);
    printf("Warp size: %d\n",deviceProp.warpSize);
    printf("\n");
  //}
  }

  cudaMallocHost((void**)obj,sizeof(float)*mynumpix);
  cudaMallocHost((void**)gra,sizeof(float)*mynumpix);
  cudaMallocHost((void**)dir,sizeof(float)*mynumpix);
  cudaMallocHost((void**)mes,sizeof(float)*mynumray);
  cudaMallocHost((void**)res,sizeof(float)*mynumray);
  cudaMallocHost((void**)ray,sizeof(float)*mynumray);
  //raypart = new float[raynumout];
  //raybuff = new float[raynuminc];
  cudaMallocHost((void**)&raypart,sizeof(float)*raynumout);
  cudaMallocHost((void**)&raybuff,sizeof(float)*raynuminc);

  float projmem = 0;
  projmem = projmem + sizeof(int)/1e9*(proj_numblocks+1);
  projmem = projmem + sizeof(int)/1e9*(proj_blocknztot+1);
  projmem = projmem + sizeof(int)/1e9*(proj_blocknztot*proj_buffsize);
  projmem = projmem + sizeof(int)/1e9*(proj_buffnztot*proj_blocksize);
  projmem = projmem + sizeof(float)/1e9*(proj_buffnztot*proj_blocksize);
  //printf("PROC %d FORWARD PROJECTION MEMORY: %f GB\n",myid,projmem);

  cudaMalloc((void**)&proj_blockdispl_d,sizeof(int)*(proj_numblocks+1));
  cudaMalloc((void**)&proj_buffdispl_d,sizeof(int)*(proj_blocknztot+1));
  cudaMalloc((void**)&proj_buffmap_d,sizeof(int)*proj_blocknztot*proj_buffsize);
  cudaMalloc((void**)&proj_buffindex_d,sizeof(int)*proj_buffnztot*proj_blocksize);
  cudaMalloc((void**)&proj_buffvalue_d,sizeof(float)*proj_buffnztot*proj_blocksize);
  cudaMemcpy(proj_blockdispl_d,proj_blockdispl,sizeof(int)*(proj_numblocks+1),cudaMemcpyHostToDevice);
  cudaMemcpy(proj_buffdispl_d,proj_buffdispl,sizeof(int)*(proj_blocknztot+1),cudaMemcpyHostToDevice);
  cudaMemcpy(proj_buffmap_d,proj_buffmap,sizeof(int)*proj_blocknztot*proj_buffsize,cudaMemcpyHostToDevice);
  cudaMemcpy(proj_buffindex_d,proj_buffindex,sizeof(short)*proj_buffnztot*proj_blocksize,cudaMemcpyHostToDevice);
  cudaMemcpy(proj_buffvalue_d,proj_buffvalue,sizeof(float)*proj_buffnztot*proj_blocksize,cudaMemcpyHostToDevice);

  /*for(int block = 0; block < proj_numblocks; block++)
    for(int buff = proj_blockdispl[block]; buff < proj_blockdispl[block+1]; buff++){
      printf("                                              block %d buff %d\n",block,buff);
      for(int m = proj_buffdispl[buff]; m < proj_buffdispl[buff+1]; m++){
        for(int n = 0; n < proj_blocksize; n++)
          printf("%d ",proj_buffindex[m*proj_blocksize+n]);
        printf("\n");
      }
    }
  for(int block = 0; block < proj_numblocks; block++)
    for(int buff = proj_blockdispl[block]; buff < proj_blockdispl[block+1]; buff++){
      printf("                                              block %d buff %d\n",block,buff);
      for(int m = proj_buffdispl[buff]; m < proj_buffdispl[buff+1]; m++){
        for(int n = 0; n < proj_blocksize; n++)
          printf("%0.1f ",proj_buffvalue[m*proj_blocksize+n]);
        printf("\n");
      }
    }
  printf("buffnztot: %d blocksize: %d %d\n",proj_buffnztot,proj_blocksize,proj_buffnztot*proj_blocksize);
  printf("blocknztot: %d buffsize: %d %d\n",proj_blocknztot,proj_buffsize,proj_blocknztot*proj_buffsize);*/

  cudaMalloc((void**)&back_blockdispl_d,sizeof(int)*(back_numblocks+1));
  cudaMalloc((void**)&back_buffdispl_d,sizeof(int)*(back_blocknztot+1));
  cudaMalloc((void**)&back_buffmap_d,sizeof(int)*back_blocknztot*back_buffsize);
  cudaMalloc((void**)&back_buffindex_d,sizeof(int)*back_buffnztot*back_blocksize);
  cudaMalloc((void**)&back_buffvalue_d,sizeof(float)*back_buffnztot*back_blocksize);
  cudaMemcpy(back_blockdispl_d,back_blockdispl,sizeof(int)*(back_numblocks+1),cudaMemcpyHostToDevice);
  cudaMemcpy(back_buffdispl_d,back_buffdispl,sizeof(int)*(back_blocknztot+1),cudaMemcpyHostToDevice);
  cudaMemcpy(back_buffmap_d,back_buffmap,sizeof(int)*back_blocknztot*back_buffsize,cudaMemcpyHostToDevice);
  cudaMemcpy(back_buffindex_d,back_buffindex,sizeof(short)*back_buffnztot*back_blocksize,cudaMemcpyHostToDevice);
  cudaMemcpy(back_buffvalue_d,back_buffvalue,sizeof(float)*back_buffnztot*back_blocksize,cudaMemcpyHostToDevice);

  float backmem = 0;
  backmem = backmem + sizeof(int)/1e9*(back_numblocks+1);
  backmem = backmem + sizeof(int)/1e9*(back_blocknztot+1);
  backmem = backmem + sizeof(int)/1e9*(back_blocknztot*back_buffsize);
  backmem = backmem + sizeof(int)/1e9*(back_buffnztot*back_blocksize);
  backmem = backmem + sizeof(float)/1e9*(back_buffnztot*back_blocksize);
  //printf("PROC %d BACKPROJECTION MEMORY: %f GB\n",myid,backmem);

  printf("PROC %d TOTAL GPU MEMORY: %f GB\n",myid,projmem+backmem);

  cudaMalloc((void**)&rayraystart_d,sizeof(int)*(mynumray+1));
  cudaMalloc((void**)&rayrayind_d,sizeof(int)*raynuminc);
  cudaMalloc((void**)&rayindray_d,sizeof(int)*raynuminc);
  cudaMemcpy(rayraystart_d,rayraystart,sizeof(int)*(mynumray+1),cudaMemcpyHostToDevice);
  cudaMemcpy(rayrayind_d,rayrayind,sizeof(int)*raynuminc,cudaMemcpyHostToDevice);
  cudaMemcpy(rayindray_d,rayrecvlist,sizeof(int)*raynuminc,cudaMemcpyHostToDevice);

  cudaMalloc((void**)&tomogram_d,sizeof(float)*mynumpix);
  cudaMalloc((void**)&sinogram_d,sizeof(float)*mynumray);
  cudaMalloc((void**)&raypart_d,sizeof(float)*raynumout);
  cudaMalloc((void**)&raybuff_d,sizeof(float)*raynuminc);

  //cudaFuncSetAttribute(kernel_SpMV_buffered,cudaFuncAttributeMaxDynamicSharedMemorySize,98304);
}

void projection(float *mes, float *obj){
  MPI_Barrier(MPI_COMM_WORLD);
  double timef = MPI_Wtime();
  {
    double time = timef;
    int blocksize = proj_blocksize;
    int numblocks = proj_numblocks;
    int buffsize = proj_buffsize;
    int *blockdispl = proj_blockdispl_d;
    int *buffdispl = proj_buffdispl_d;
    int *buffmap = proj_buffmap_d;
    short *buffindex = proj_buffindex_d;
    float *buffvalue = proj_buffvalue_d;
    cudaMemcpy(tomogram_d,obj,sizeof(float)*mynumpix,cudaMemcpyHostToDevice);
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
    kernel_SpMV_buffered<<<numblocks,blocksize,sizeof(float)*buffsize>>>(raypart_d,tomogram_d,buffindex,buffvalue,raynumout,blockdispl,buffdispl,buffmap,buffsize);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  fktime = fktime + milliseconds/1000;
    //cudaDeviceSynchronize();
    //MPI_Barrier(MPI_COMM_WORLD);
    //fktime = fktime + MPI_Wtime()-time;
  }
  {
    double time = MPI_Wtime();
    cudaMemcpy(raypart,raypart_d,sizeof(float)*raynumout,cudaMemcpyDeviceToHost);
    MPI_Alltoallv(raypart,raysendcount,raysendstart,MPI_FLOAT,raybuff,rayrecvcount,rayrecvstart,MPI_FLOAT,MPI_COMM_WORLD);
    cudaMemcpy(raybuff_d,raybuff,sizeof(float)*raynuminc,cudaMemcpyHostToDevice);
    MPI_Barrier(MPI_COMM_WORLD);
    aftime = aftime + MPI_Wtime()-time;
  }
  {
    double time = MPI_Wtime();
    kernel_SpReduce<<<(mynumray+255)/256,256>>>(sinogram_d,raybuff_d,rayraystart_d,rayrayind_d,mynumray);
    cudaMemcpy(mes,sinogram_d,sizeof(float)*mynumray,cudaMemcpyDeviceToHost);
    MPI_Barrier(MPI_COMM_WORLD);
    frtime = frtime + MPI_Wtime()-time;
  }
  ftime = ftime + MPI_Wtime()-timef;
  numproj++;
}

void backprojection(float *gra, float *res){
  MPI_Barrier(MPI_COMM_WORLD);
  double timeb = MPI_Wtime();
  {
    double time = timeb;
    cudaMemcpy(sinogram_d,res,sizeof(float)*mynumray,cudaMemcpyHostToDevice);
    kernel_SpGather<<<(raynuminc+255)/256,256>>>(raybuff_d,sinogram_d,rayindray_d,raynuminc);
    cudaDeviceSynchronize();
    MPI_Barrier(MPI_COMM_WORLD);
    brtime = brtime + MPI_Wtime()-time;
  }
  {
    double time = MPI_Wtime();
    cudaMemcpy(raybuff,raybuff_d,sizeof(float)*raynuminc,cudaMemcpyDeviceToHost);
    MPI_Alltoallv(raybuff,rayrecvcount,rayrecvstart,MPI_FLOAT,raypart,raysendcount,raysendstart,MPI_FLOAT,MPI_COMM_WORLD);
    cudaMemcpy(raypart_d,raypart,sizeof(float)*raynumout,cudaMemcpyHostToDevice);
    MPI_Barrier(MPI_COMM_WORLD);
    abtime = abtime + MPI_Wtime()-time;
  }
  {
    double time = MPI_Wtime();
    int blocksize = back_blocksize;
    int numblocks = back_numblocks;
    int buffsize = back_buffsize;
    int *blockdispl = back_blockdispl_d;
    int *buffdispl = back_buffdispl_d;
    int *buffmap = back_buffmap_d;
    short *buffindex = back_buffindex_d;
    float *buffvalue = back_buffvalue_d;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
    kernel_SpMV_buffered<<<numblocks,blocksize,sizeof(float)*buffsize>>>(tomogram_d,raypart_d,buffindex,buffvalue,mynumpix,blockdispl,buffdispl,buffmap,buffsize);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  bktime = bktime + milliseconds/1000;
    cudaMemcpy(gra,tomogram_d,sizeof(float)*mynumpix,cudaMemcpyDeviceToHost);
    //MPI_Barrier(MPI_COMM_WORLD);
    //bktime = bktime + MPI_Wtime()-time;
  }
  btime = btime + MPI_Wtime()-timeb;
  numback++;
}

__global__ void kernel_SpMV_buffered(float *y, float *x, short *index, float *value, int numrow, int *blockdispl, int *buffdispl, int *buffmap, int buffsize){
  extern __shared__ float shared[];
  float reduce = 0;
  int ind;
  for(int buff = blockdispl[blockIdx.x]; buff < blockdispl[blockIdx.x+1]; buff++){
    for(int i = threadIdx.x; i < buffsize; i += blockDim.x)
      shared[i] = x[buffmap[buff*buffsize+i]];
    __syncthreads();
    for(int n = buffdispl[buff]; n < buffdispl[buff+1]; n++){
      ind = n*blockDim.x+threadIdx.x;
      reduce = reduce + shared[index[ind]]*value[ind];
    }
    __syncthreads();
  }
  ind = blockIdx.x*blockDim.x+threadIdx.x;
  if(ind < numrow)
    y[ind] = reduce;
}
__global__ void kernel_SpReduce(float *y, float *x, int *displ, int *index, int numrow){
  int row = blockIdx.x*blockDim.x+threadIdx.x;
  float reduce = 0;
  if(row < numrow){
    for(int n = displ[row]; n < displ[row+1]; n++)
      reduce = reduce + x[index[n]];
    y[row] = reduce;
  }
}
__global__ void kernel_SpGather(float *y, float *x, int *index, int numrow){
  int row = blockIdx.x*blockDim.x+threadIdx.x;
  if(row < numrow)
    y[row] = x[index[row]];
}
