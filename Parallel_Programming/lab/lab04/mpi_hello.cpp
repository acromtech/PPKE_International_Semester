#include <mpi.h>
#include <iostream>

int main(int argc, char *argv[]) {
 int  numtasks, rank, len, rc; 
 char hostname[MPI_MAX_PROCESSOR_NAME];

 rc = MPI_Init(&argc,&argv);
 if (rc != MPI_SUCCESS) {
   std::cout << "Error starting MPI program. Terminating.\n";
   MPI_Abort(MPI_COMM_WORLD, rc);
 }

 MPI_Comm_size(MPI_COMM_WORLD,&numtasks);
 MPI_Comm_rank(MPI_COMM_WORLD,&rank);
 MPI_Get_processor_name(hostname, &len);
 std::cout << "Number of tasks=" << numtasks << " My rank= "<< rank <<" Running on " << hostname << std::endl;

 MPI_Finalize();
}
