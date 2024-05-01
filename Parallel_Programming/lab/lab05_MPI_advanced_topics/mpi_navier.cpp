#include <mpi.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <iostream>

int my_rank;
int nprocs;
int nprocs_y;
int nprocs_x;
int my_rank_x;
int my_rank_y;
int prev_y;
int next_y;
int next_x;
int prev_x;
int imax_full;
int jmax_full;
int gbl_i_begin;
int gbl_j_begin;
MPI_Comm cartcomm;
MPI_Datatype row, column;

double* dat_ptrs[6];
int dat_dirty[6] = {1,1,1,1,1,1};

void mpi_setup(int *imax, int *jmax) {
	int periods[2] = {0,0};
	int reorder = 0;
	//Initialise: get #of processes and process id
	MPI_Init(NULL, NULL);
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

	//Check for compatible number of processes
	int dims[2] = {0,0};
	MPI_Dims_create(nprocs, 2, &dims[0]);
	if (my_rank==0) std::cout << "(" << dims[0]<<","<<dims[1]<<")\n";

	//Create cartesian communicator for 2D, dims[0]*dims[1] processes,
	//without periodicity and reordering
	MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &cartcomm);
	//Get my rank in the new communicator
	int coords[2];
	//Get my coordinates coords[0] and coords[1]
	MPI_Cart_coords(cartcomm, my_rank, 2, coords);
	my_rank_x = coords[0];
	my_rank_y = coords[1];
	//Get my neighbours in dimension 0
	MPI_Cart_shift(cartcomm, 0, 1, &prev_x, &next_x);
	//Get my neighbours in dirmension 1
	MPI_Cart_shift(cartcomm, 1, 1, &prev_y, &next_y);

	//Save original full sizes in x and y directions
	imax_full = *imax;
	jmax_full = *jmax;

	//Modify imax and jmax (pay attention to integer divisions's rounding issues!)
	*imax = *imax / dims[0];
	gbl_i_begin = my_rank_x * *imax;
	if (my_rank_x + 1 == dims[0] && *imax * dims[0] != imax_full)
		*imax += (imax_full - *imax * dims[0]);
	*jmax = *jmax / dims[1];
	gbl_j_begin = my_rank_y * *jmax;
	if (my_rank_y + 1 == dims[1] && *jmax * dims[1] != jmax_full)
		*jmax += (jmax_full - *jmax * dims[1]);

	//Let's set up MPI Datatypes
	MPI_Type_vector((*imax + 2), 1, 1, MPI_DOUBLE, &row);
	MPI_Type_vector((*jmax + 2), 1, (*imax+2), MPI_DOUBLE, &column);
	MPI_Type_commit(&row);
	MPI_Type_commit(&column);


}

void exchange_halo(int imax, int jmax, double *arr) {
	MPI_Sendrecv(&arr[(jmax)*(imax+2)], 1, row, next_y, 0,
			&arr[0]              , 1, row, prev_y, 0,
			cartcomm, MPI_STATUS_IGNORE);
	MPI_Sendrecv(&arr[(imax+2)],          1, row, prev_y, 0,
			&arr[(jmax+1)*(imax+2)], 1, row, next_y, 0,
			cartcomm, MPI_STATUS_IGNORE);
	MPI_Sendrecv(&arr[imax],            1, column, next_x, 0,
			&arr[0]              , 1, column, prev_x, 0,
			cartcomm, MPI_STATUS_IGNORE);
	MPI_Sendrecv(&arr[1],                 1, column, prev_x, 0,
			&arr[(imax+1)],          1, column, next_x, 0,
			cartcomm, MPI_STATUS_IGNORE);
}

void set_dirty(int jmax, double *arr) {
}
