/*************************************************
 * Laplace Serial C Version
 *
 * Temperature is initially 0.0
 * Boundaries are as follows:
 *
 *      0         T         0
 *   0  +-------------------+  0
 *      |                   |
 *      |                   |
 *      |                   |
 *   T  |                   |  T
 *      |                   |
 *      |                   |
 *      |                   |
 *   0  +-------------------+ 100
 *      0         T        100
 *
 *  John Urbanic, PSC 2014
 *
 ************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include "mpi.h"

// size of plate
#define COLUMNS    1000
#define ROWS       1000

// largest permitted change in temp (This value takes about 3400 steps)
#define MAX_TEMP_ERROR 0.01

// 250 + upper and lower ghost zones
double Temperature[ROWS/4 + 2][COLUMNS+2];      // temperature grid
double Temperature_last[ROWS/4 + 2][COLUMNS+2]; // temperature grid from last iteration

//   helper routines
void initialize(int my_PE);
void track_progress(int iter);


int main(int argc, char *argv[]) {

    int i, j;                                            // grid indexes
    int max_iterations;                                  // number of iterations
    int iteration=1;                                     // current iteration
    double dt=100;                                       // largest change in t
    struct timeval start_time, stop_time, elapsed_time;  // timers

    // MPI initialization stuffs
    MPI_Status status;
    MPI_Init(&argc, &argv);
    int my_PE, num_PEs;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_PE);
    MPI_Comm_size(MPI_COMM_WORLD, &num_PEs); 
    if(my_PE == 0)
    {
       if(num_PEs != 4) // must have 4 workers
       {
             printf("This program is written for 4 PEs only");
             return 1; // return error
       }
        printf("Maximum iterations [100-4000]?\n");
        scanf("%d", &max_iterations);

        gettimeofday(&start_time,NULL); // Unix timer
    }

      // split up plate into 4 parts
    initialize(my_PE);       // initialize Temp_last including boundary conditions

    // Need to broadcast the max_iterations to all other PE's
    MPI_Bcast(&max_iterations, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // do until error is minimal or until max steps
    while ( dt > MAX_TEMP_ERROR && iteration <= max_iterations ) {

        // main calculation: average my four neighbors
        for(i = 1; i <= ROWS/4; i++) {
            for(j = 1; j <= COLUMNS; j++) {
                Temperature[i][j] = 0.25 * (Temperature_last[i+1][j] + Temperature_last[i-1][j] +
                                            Temperature_last[i][j+1] + Temperature_last[i][j-1]);
            }
        }

	// Send Ghost Zones to neighbors
	if(my_PE == 0)
	{
	   MPI_Send(&Temperature[ROWS/4][0], COLUMNS, MPI_FLOAT, 1, 10, MPI_COMM_WORLD);
           MPI_Recv(&Temperature[ROWS/4+1][0], COLUMNS, MPI_FLOAT,1, 10, MPI_COMM_WORLD, &status); 
	}
	else if(my_PE == 1 || my_PE == 2)
	{
	   MPI_Send(&Temperature[1][0], COLUMNS, MPI_FLOAT, my_PE-1, 10, MPI_COMM_WORLD);
	   MPI_Send(&Temperature[ROWS/4][0], COLUMNS, MPI_FLOAT, my_PE+1,10,  MPI_COMM_WORLD);
	   MPI_Recv(&Temperature[0][0], COLUMNS, MPI_FLOAT, my_PE-1, 10, MPI_COMM_WORLD, &status);
	   MPI_Recv(&Temperature[ROWS/4+1][0], COLUMNS, MPI_FLOAT, my_PE+1, 10, MPI_COMM_WORLD, &status);
	}
	else if(my_PE == 3)
	{
	   MPI_Send(&Temperature[1][0], COLUMNS, MPI_FLOAT, 2, 10, MPI_COMM_WORLD);
           MPI_Recv(&Temperature[0][0], COLUMNS, MPI_FLOAT, 2, 10, MPI_COMM_WORLD, &status);
	}

        dt = 0.0; // reset largest temperature change

        // copy grid to old grid for next iteration and find latest dt
        for(i = 1; i <= ROWS/4+1; i++){
//printf("[");
            for(j = 1; j <= COLUMNS; j++){
	      dt = fmax( fabs(Temperature[i][j]-Temperature_last[i][j]), dt);
	      Temperature_last[i][j] = Temperature[i][j];
            }
        }

	// need to find global dt. Going to send all the local dt to PE 0 and then find max
	if(my_PE == 0)
	{
	   double  dt0 = dt;
	   double  dt1, dt2, dt3;
	   MPI_Recv(&dt1, 1, MPI_FLOAT, 1, iteration, MPI_COMM_WORLD, &status);
	   MPI_Recv(&dt2, 1, MPI_FLOAT, 2, iteration, MPI_COMM_WORLD, &status);
	   MPI_Recv(&dt3, 1, MPI_FLOAT, 3, iteration, MPI_COMM_WORLD, &status);
	   //printf("Iteration %d: dt0 %f, dt1 %f, dt2 %f, dt3 %f\n",iteration, dt0, dt1, dt2, dt3);
	    dt1 = fmax(dt0,dt1);
	   dt2 = fmax( dt2, dt3);
	   dt = fmax(dt1, dt2);
	}
	else
	{
	   MPI_Send(&dt, 1, MPI_FLOAT, 0, iteration, MPI_COMM_WORLD);
	}

	//Broadcast dt to PEs
	MPI_Bcast(&dt, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);


        // periodically print test values
        if((iteration % 100) == 0 && my_PE == 3) {
 	    track_progress(iteration);
        }
	//printf("End of iteration %d, from %d\n", iteration, my_PE);
	iteration++;
    }

    if(my_PE == 0)
   {
       gettimeofday(&stop_time,NULL);
       timersub(&stop_time, &start_time, &elapsed_time); // Unix time subtract routine

       printf("\nMax error at iteration %d was %f\n", iteration-1, dt);
       printf("Total time was %f seconds.\n", elapsed_time.tv_sec+elapsed_time.tv_usec/1000000.0);
   }

   MPI_Finalize();
}


// initialize plate and boundary conditions
// Temp_last is used to to start first iteration
void initialize(int my_PE){

    int i,j;

    for(i = 0; i < ROWS/4+2; i++){
        for (j = 0; j <= COLUMNS+1; j++){
            Temperature_last[i][j] = 0.0;
        }
    }

    // these boundary conditions never change throughout run

    // set left side to 0 and right to a linear increase
   int row;
   if(my_PE == 0)
   {
      row=0;
      for(i = 0; i < ROWS/4+1; i++) {
           Temperature_last[row][0] = 0.0;
           Temperature_last[row][COLUMNS+1] = (100.0/ROWS)*i;
           row++;
       }
   }
   else if(my_PE == 1)
   {
      row=1;// ghost line is row 0
      for(i = ROWS/4+1; i < ROWS/4 *2 + 1; i++) {
           Temperature_last[row][0] = 0.0;
           Temperature_last[row][COLUMNS+1] = (100.0/ROWS)*i;
           row++;
       }
   }
   else if(my_PE == 2)
   {
      row=1;// ghost line is row 0
      for(i = ROWS/4 * 2 +1; i < ROWS/4 * 3 + 1; i++) {
           Temperature_last[row][0] = 0.0;
           Temperature_last[row][COLUMNS+1] = (100.0/ROWS)*i;
           row++;
       }
   }
   else if(my_PE == 3)
   {
      row=1;// ghost line is row 0
      for(i = ROWS/4 *3 +1; i <= ROWS + 1; i++) {
           Temperature_last[row][0] = 0.0;
           Temperature_last[row][COLUMNS+1] = (100.0/ROWS)*i;
           row++;
       }
       // set top to 0 and bottom to linear increase
       for(j = 0; j <= COLUMNS+1; j++) {
           Temperature_last[0][j] = 0.0;
           Temperature_last[ROWS/4+1][j] = (100.0/COLUMNS)*j;
       }
    }
}


// print diagonal in bottom right corner where most action is
void track_progress(int iteration) {

    int i;

    printf("---------- Iteration number: %d ------------\n", iteration);
    for(i = ROWS/4+1-5; i <= ROWS/4; i++) {
        printf("[%d,%d]: %5.2f  ", i, i, Temperature[i][i]);
    }
    printf("\n");
}
