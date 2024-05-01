/* Copyright (c) 2012, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <omp.h>
#include <mpi.h>
#include <stdio.h>
#include "mpi_navier.h"


int main(int argc, char** argv)
{
    //TODO initialize MPI
    
    //Size along y
    int jmax = 4094;
    //Size along x
    int imax = 4094;
    int iter_max = 1000;
    mpi_setup(&imax, &jmax);
    const double pi  = 2.0 * asin(1.0);
    const double tol = 1.0e-5;
    double error     = 1.0;

    double *A;
    double *Anew;
    double *y0;

    A    = (double *)malloc((imax+2) * (jmax+2) * sizeof(double));
    Anew = (double *)malloc((imax+2) * (jmax+2) * sizeof(double));
    y0   = (double *)malloc((imax+2) * sizeof(double));

    memset(A, 0, (imax+2) * (jmax+2) * sizeof(double));
    
    // set boundary conditions
    for (int i = 0; i < imax+2; i++)
      A[(0)*(imax+2)+i]   = 0.0;

    for (int i = 0; i < imax+2; i++)
      A[(jmax+1)*(imax+2)+i] = 0.0;
    
    for (int j = 0; j < jmax+2; j++)
    {
        y0[j] = sin(pi * (gbl_j_begin+j) / (jmax_full+1)); //TODO: within sin(), j is a global index and jmax is the global size
                                          //need to offset j by the beginning j index of this partition, and use the full jmax size
        A[(j)*(imax+2)+0] = y0[j];
    }

    for (int j = 0; j < imax+2; j++)
    {
        y0[j] = sin(pi * (gbl_j_begin+j)/ (jmax_full+1)); //TODO: within sin(), j is a global index and jmax is the global size
                                          //need to offset j by the beginning j index of this partition, and use the full jmax size
        A[(j)*(imax+2)+imax+1] = y0[j]*exp(-pi);
    }
    
    //TODO: Only process 0 should print this, and with the full imax and jmax sizes
    //printf("Jacobi relaxation Calculation: %d x %d mesh\n", imax+2, jmax+2);
    
    double t1 = omp_get_wtime();
    int iter = 0;
    
    for (int i = 1; i < imax+2; i++)
       Anew[(0)*(imax+2)+i]   = 0.0;

    for (int i = 1; i < imax+2; i++)
       Anew[(jmax+1)*(imax+2)+i] = 0.0;

    for (int j = 1; j < jmax+2; j++)
        Anew[(j)*(imax+2)+0]   = y0[j];

    for (int j = 1; j < jmax+2; j++)
        Anew[(j)*(imax+2)+jmax+1] = y0[j]*expf(-pi);
    

    while ( error > tol && iter < iter_max )
    {
        error = 0.0;
        //TODO: need to make sure stencil accesses to A read correct data
	exchange_halo(imax, jmax, A);
#pragma omp parallel for reduction(max:error)
        for( int j = 1; j < jmax+1; j++ )
        {
            for( int i = 1; i < imax+1; i++)
            {
                Anew[(j)*(imax+2)+i] = 0.25f * ( A[(j)*(imax+2)+i+1] + A[(j)*(imax+2)+i-1]
                                     + A[(j-1)*(imax+2)+i] + A[(j+1)*(imax+2)+i]);
                error = fmax( error, fabs(Anew[(j)*(imax+2)+i]-A[(j)*(imax+2)+i]));
            }
        }
        //TODO: need global reduction for error
	double error_gbl;
	MPI_Allreduce(&error, &error_gbl, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
	error = error_gbl;

        //No stencil accesses to Anew, no halo exchange necessary
#pragma omp parallel for 
        for( int j = 1; j < jmax+1; j++ )
        {
            for( int i = 1; i < imax+1; i++)
            {
                A[(j)*(imax+2)+i] = Anew[(j)*(imax+2)+i];    
            }
        }
        //TODO: Need to set A to dirty
        //TODO: Only process 0 should print this
        if(iter % 100 == 0) printf("%5d, %0.6f\n", iter, error);
        
        iter++;
    }

    double runtime = omp_get_wtime()-t1;
 
    printf(" total: %f s\n", runtime);

    //TODO finalize MPI
    MPI_Finalize();
}
