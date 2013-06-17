#include <Teuchos_RCP.hpp>

#ifdef HAVE_MPI
#include <Epetra_MpiComm.h>
#else
#include <Epetra_SerialComm.h>
#endif

#include "Epetra_Map.h"
#include "Epetra_Vector.h"
#include "Epetra_Version.h"

using Teuchos::rcp;
using Teuchos::RCP;

int main(int argc, char *argv[])
{
  // Create a communicator for Epetra objects.
#ifdef HAVE_MPI
  MPI_Init( &argc, &argv );
  RCP<const Epetra_MpiComm> Comm =
    rcp<Epetra_MpiComm>(new Epetra_MpiComm(MPI_COMM_WORLD));
#else
  RCP<const Epetra_SerialComm> Comm =
    rcp<Epetra_SerialComm>(new Epetra_SerialComm());
#endif

  int NumElements = 1000;

  // Construct a Map with NumElements and index base of 0
  Epetra_Map Map(NumElements, 0, *Comm);

  // Create x and b vectors
  Epetra_Vector x(Map);
  Epetra_Vector b(Map);

  b.Random();

  x.Update(2.0, b, 0.0); // x = 2*b

  double bnorm, xnorm;

  x.Norm2(&xnorm);
  b.Norm2(&bnorm);

  cout << "2 norm of x = " << xnorm << endl
       << "2 norm of b = " << bnorm << endl;

#ifdef HAVE_MPI
    MPI_Finalize();
#endif

  return EXIT_SUCCESS;
}
