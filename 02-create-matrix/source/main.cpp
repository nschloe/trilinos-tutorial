#include <Teuchos_RCP.hpp>

#ifdef HAVE_MPI
#include <Epetra_MpiComm.h>
#else
#include <Epetra_SerialComm.h>
#endif

#include "Epetra_Map.h"
#include "Epetra_Vector.h"
#include "Epetra_CrsMatrix.h"

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

  const int NumGlobalElements = 10;
  int ierr;

  // Construct a Map with NumElements and index base of 0
  Epetra_Map Map(NumGlobalElements, 0, *Comm);

  // Get update list and number of local equations from newly created Map.
  int NumMyElements = Map.NumMyElements();

  std::vector<int> MyGlobalElements(NumMyElements);
  Map.MyGlobalElements(&MyGlobalElements[0]);

  // NumNz[i] is the number of nonzero elements in row i of the sparse
  // matrix on this MPI process.  Epetra_CrsMatrix uses this to figure
  // out how much space to allocate.
  std::vector<int> NumNz (NumMyElements);

  // We are building a tridiagonal matrix where each row contains the
  // nonzero elements (-1 2 -1).  Thus, we need 2 off-diagonal terms,
  // except for the first and last row of the matrix.
  for (int i = 0; i < NumMyElements; ++i)
    if (MyGlobalElements[i] == 0 || MyGlobalElements[i] == NumGlobalElements-1)
      NumNz[i] = 2; // First or last row
    else
      NumNz[i] = 3; // Not the (first or last row)

  // Create the Epetra_CrsMatrix.
  Epetra_CrsMatrix A (Copy, Map, &NumNz[0]);


  //
  // Add rows to the sparse matrix one at a time.
  //
  std::vector<double> Values(2);
  Values[0] = -1.0; Values[1] = -1.0;
  std::vector<int> Indices(2);
  const double two = 2.0;
  int NumEntries;
  for (int i = 0; i < NumMyElements; ++i)
  {
    if (MyGlobalElements[i] == 0)
    { // The first row of the matrix.
      Indices[0] = 1;
      NumEntries = 1;
    }
    else if (MyGlobalElements[i] == NumGlobalElements - 1)
    { // The last row of the matrix.
      Indices[0] = NumGlobalElements-2;
      NumEntries = 1;
    }
    else
    { // Any row of the matrix other than the first or last.
      Indices[0] = MyGlobalElements[i]-1;
      Indices[1] = MyGlobalElements[i]+1;
      NumEntries = 2;
    }
    ierr = A.InsertGlobalValues(MyGlobalElements[i], NumEntries, &Values[0], &Indices[0]);
    assert (ierr==0);
    // Insert the diagonal entry.
    ierr = A.InsertGlobalValues(MyGlobalElements[i], 1, &two, &MyGlobalElements[i]);
    assert(ierr==0);
  }
  // Finish up.  We can call FillComplete() with no arguments, because
  // the matrix is square.
  ierr = A.FillComplete();
  assert (ierr==0);

  std::cout << A << std::endl;

#ifdef HAVE_MPI
    MPI_Finalize();
#endif

  return EXIT_SUCCESS;
}
