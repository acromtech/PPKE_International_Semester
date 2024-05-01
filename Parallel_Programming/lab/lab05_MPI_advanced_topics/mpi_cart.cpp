#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#define SIZE 16
#define UP    0
#define DOWN  1
#define LEFT  2
#define RIGHT 3

int main(int argc, char *argv[])  {
  int numtasks, rank, source, dest, outbuf, i, tag=1, 
      inbuf[4]={MPI_PROC_NULL,MPI_PROC_NULL,MPI_PROC_NULL,MPI_PROC_NULL}, 
      nbrs[4], dims[2]={4,4}, 
      periods[2]={0,0}, reorder=0, coords[2];

  MPI_Request reqs[8];
  MPI_Status stats[8];
  MPI_Comm cartcomm;

  MPI_Init(&argc,&argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numtasks);

  //TODO: make this work for any number of processes, not just 16:
  // look up and use MPI_Dims_create

  if (numtasks == SIZE) {
    //Create cartesian communicator for 2D, dims[0]*dims[1] processes,
    //without periodicity and reordering
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &cartcomm);
    //Get my rank in the new communicator
    MPI_Comm_rank(cartcomm, &rank);
    //Get my coordinates coords[0] and coords[1]
    MPI_Cart_coords(cartcomm, rank, 2, coords);
    //Get my neighbours in dimension 0
    MPI_Cart_shift(cartcomm, 0, 1, &nbrs[LEFT], &nbrs[RIGHT]);
    //Get my neighbours in dirmension 1
    MPI_Cart_shift(cartcomm, 1, 1, &nbrs[UP], &nbrs[DOWN]);

    printf("rank= %d coords= %d %d  neighbors(u,d,l,r)= %d %d %d %d\n",
        rank,coords[0],coords[1],nbrs[UP],nbrs[DOWN],nbrs[LEFT],
        nbrs[RIGHT]);

    outbuf = rank;

    //Send my rank to all four neighbours, and receive message from them
    

    printf("rank= %d inbuf(u,d,l,r)= %d %d %d %d\n",
        rank,inbuf[UP],inbuf[DOWN],inbuf[LEFT],inbuf[RIGHT]);  }
  else
    printf("Must specify %d processors. Terminating.\n",SIZE);

  MPI_Finalize();
}
