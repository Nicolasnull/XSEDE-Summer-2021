#include <stdio.h>
#include "mpi.h"

int main(int argc, char** argv)
{
	MPI_Init(&argc, &argv);
	MPI_Status status;
	int message_recieved, my_PE;
	MPI_Comm_rank(MPI_COMM_WORLD, &my_PE);

	int number_to_send = my_PE;

	if(my_PE == 7)
		MPI_Send(&number_to_send, 1, MPI_INT, 0, 10, MPI_COMM_WORLD);
	else
		MPI_Send(&number_to_send, 1, MPI_INT, my_PE+1, 10, MPI_COMM_WORLD);

	MPI_Recv(&message_recieved, 1, MPI_INT, MPI_ANY_SOURCE, 10, MPI_COMM_WORLD, &status);
	printf("PE %d recieved %d.\n", my_PE, message_recieved);
	MPI_Finalize();
	return 0;
}
