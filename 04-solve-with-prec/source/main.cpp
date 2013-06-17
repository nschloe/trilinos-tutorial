#include <Teuchos_RCP.hpp>

#ifdef HAVE_MPI
#include <Epetra_MpiComm.h>
#else
#include <Epetra_SerialComm.h>
#endif

#include <BelosLinearProblem.hpp>
#include <BelosEpetraAdapter.hpp>
#include <BelosPseudoBlockCGSolMgr.hpp>
#include <ml_epetra_preconditioner.h>

#include <MueLu.hpp>
#include <BelosXpetraAdapter.hpp>
#include <BelosMueLuAdapter.hpp>
#include <MueLu_UseDefaultTypes.hpp>  // => Scalar=double, LocalOrdinal=int, GlobalOrdinal=int
#include <MueLu_UseShortNames.hpp>

#include "Galeri_Maps.h"
#include "Galeri_CrsMatrices.h"
#include "Galeri_Utils.h"

#include "Epetra_Map.h"
#include "Epetra_Vector.h"
#include "Epetra_CrsMatrix.h"

#include "Ifpack.h"
#include "Ifpack_AdditiveSchwarz.h"

//typedef double ST;
typedef Epetra_MultiVector MV;
typedef Epetra_Operator OP;

using Teuchos::rcp;
using Teuchos::RCP;

int main(int argc, char *argv[])
{
  // Create a communicator for Epetra objects.
#ifdef HAVE_MPI
  MPI_Init( &argc, &argv );
  RCP<Epetra_MpiComm> Comm =
    rcp<Epetra_MpiComm>(new Epetra_MpiComm(MPI_COMM_WORLD));
#else
  RCP<Epetra_SerialComm> Comm =
    rcp<Epetra_SerialComm>(new Epetra_SerialComm());
#endif

  const int NumGlobalElements = 10;
  // ---------------------------------------------------------------------------
  // Construct a Map with NumElements and index base of 0
  Teuchos::ParameterList GaleriList;

  // Set the number of discretization points in the x and y direction.
  GaleriList.set ("nx", 100 * Comm->NumProc());
  GaleriList.set ("ny", 100);

  // Create the map and matrix using the parameter list for a 2D Laplacian.
  RCP<Epetra_Map> Map = rcp(Galeri::CreateMap ("Cartesian2D", *Comm, GaleriList));
  RCP<Epetra_CrsMatrix> A = rcp(Galeri::CreateCrsMatrix ("Laplace2D", &*Map, GaleriList));
  // ---------------------------------------------------------------------------
  // Now setup the Belos solver.
  Teuchos::ParameterList belosList;
  belosList.set("Convergence Tolerance", 1.0e-10);
  belosList.set("Maximum Iterations", 1000);
  belosList.set("Output Frequency", 1);
  belosList.set("Output Style", (int) Belos::Brief);
  belosList.set("Verbosity", Belos::Errors+Belos::StatusTestDetails+Belos::Warnings+Belos::TimingDetails+Belos::IterationDetails+Belos::FinalSummary );


  RCP<Epetra_Vector> x = rcp(new Epetra_Vector(*Map));
  TEUCHOS_ASSERT_EQUALITY(0, x->PutScalar(0.0)); // not strictly necessary

  RCP<Epetra_Vector> b = rcp(new Epetra_Vector(*Map));
  TEUCHOS_ASSERT_EQUALITY(0, b->Random());

  Belos::LinearProblem<double,MV,OP> problem(A, x, b);
  TEUCHOS_ASSERT(problem.setProblem());

  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  // Create ML preconditioner.
  //MlPrec = rcp(new ML_Epetra::MultiLevelPreconditioner(A, MLList));
  Teuchos::ParameterList MLList;
  ML_Epetra::SetDefaults("SA", MLList);
  // ... set more parameters if you like.
  RCP<ML_Epetra::MultiLevelPreconditioner> MlPrec =
    rcp(new ML_Epetra::MultiLevelPreconditioner(*A, MLList));
  RCP<Belos::EpetraPrecOp> mlPrec = rcp(new Belos::EpetraPrecOp(MlPrec));
  problem.setLeftPrec(mlPrec);
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  //typedef Belos::OperatorTraits<SC, MV, OP> OPT;
  //typedef Belos::MultiVecTraits<SC, MV>     MVT;
  //RCP<Hierarchy> H = rcp(new Hierarchy(A));
  //H->setVerbLevel(Teuchos::VERB_HIGH);
  //// Multigrid setup phase (using default parameters)
  //H->Setup();
  //RCP<OP> belosOp   = rcp(new Belos::XpetraOp<SC, LO, GO, NO, LMO>(A));
  //RCP<OP> mueluPrec = rcp(new Belos::MueLuOp<SC, LO, GO, NO, LMO>(H));
  //Belos::LinearProblem<SC, MV, OP> problem(belosOp, x, b);
  //problem.setLeftPrec(mueluPrec);
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  //// Ifpack preconditioner.
  //Teuchos::ParameterList List;

  //Ifpack Factory;
  //string PrecType = "ILU";
  //int OverlapLevel = 1;
  //RCP<Ifpack_Preconditioner> Prec =
  //  rcp(Factory.Create(PrecType, &*A, OverlapLevel));
  //TEUCHOS_TEST_FOR_EXCEPTION(Prec == Teuchos::null, std::runtime_error,
  //                           "IFPACK failed to create a preconditioner of type \""
  //                           << PrecType << "\" with overlap level "
  //                           << OverlapLevel << ".");

  //List.set("fact: drop tolerance", 1e-9);
  //List.set("fact: level-of-fill", 1);
  //List.set("schwarz: combine mode", "Add");
  //IFPACK_CHK_ERR(Prec->SetParameters(List));
  //IFPACK_CHK_ERR(Prec->Initialize());
  //IFPACK_CHK_ERR(Prec->Compute());
  //RCP<Belos::EpetraPrecOp> ifPrec = rcp (new Belos::EpetraPrecOp(Prec));
  //problem.setLeftPrec(ifPrec);
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

  // Create an iterative solver manager.
  RCP<Belos::SolverManager<double,MV,OP> > newSolver =
    rcp(new Belos::PseudoBlockCGSolMgr<double,MV,OP>(rcp(&problem, false), rcp(&belosList, false)));

  // Perform solve.
  Belos::ReturnType ret = newSolver->solve();

  if (ret==Belos::Converged)
    std::cout << "Success!" << std::endl;
  // ---------------------------------------------------------------------------

#ifdef HAVE_MPI
    MPI_Finalize();
#endif

  return EXIT_SUCCESS;
}
