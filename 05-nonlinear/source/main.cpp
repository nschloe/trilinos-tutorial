//
// Simple example of solving the following nonlinear partial
// differential equation (PDE):
//
// -\Delta u + \lambda e^u = 0  in \Omega = (0,1) \times (0,1)
//                       u = 0  on \partial \Omega
//
// using NOX (Trilinos' Nonlinear Object-Oriented Solutions package).
// For more details and documentation, see the NOX web site:
//
// http://trilinos.sandia.gov/packages/nox/
//
#ifdef HAVE_MPI
#  include "mpi.h"
#  include "Epetra_MpiComm.h"
#else
#  include "Epetra_SerialComm.h"
#endif
#include "Epetra_Map.h"
#include "Epetra_CrsMatrix.h"
#include "Epetra_RowMatrix.h"
#include "Epetra_Vector.h"

#include <Teuchos_StandardCatchMacros.hpp>
#include <Teuchos_VerboseObject.hpp>

#include "Galeri_Maps.h"
#include "Galeri_CrsMatrices.h"
#include "Galeri_Utils.h"

#include "NOX.H"
#include "NOX_Epetra_Interface_Required.H"
#include "NOX_Epetra_Interface_Jacobian.H"
#include "NOX_Epetra_Interface_Preconditioner.H"
#include "NOX_Epetra_LinearSystem_AztecOO.H"
#include "NOX_Epetra_Group.H"

using Teuchos::ParameterList;
using Teuchos::parameterList;
using Teuchos::RCP;
using Teuchos::rcp;

// ==========================================================================
class MyProblem:
  public NOX::Epetra::Interface::Required,
  public NOX::Epetra::Interface::Jacobian,
  public NOX::Epetra::Interface::Preconditioner
{
public:
  // --------------------------------------------------------------------------
  MyProblem(const int nx,
            const int ny,
            const double lambda,
            Epetra_Comm& Comm):
  nx_(nx),
  ny_(ny),
  hx_(1.0/(nx_-1)),
  hy_(1.0/(ny_-1)),
  lambda_(lambda)
  {
    TEUCHOS_ASSERT_EQUALITY(nx, ny);
    // Construct the Laplacian.
    Teuchos::ParameterList GaleriList;
    GaleriList.set("nx", nx);
    GaleriList.set("ny", ny);
    RCP<Epetra_Map> Map = rcp(Galeri::CreateMap("Cartesian2D", Comm, GaleriList));
    Matrix_ = rcp(Galeri::CreateCrsMatrix("Laplace2D", &*Map, GaleriList));
    Matrix_->Scale(1.0/(hx_*hx_));
  }
  // --------------------------------------------------------------------------
  // The destructor doesn't need to do anything, because RCPs are
  // smart pointers; they handle deallocation automatically.
  ~MyProblem() {}
  // --------------------------------------------------------------------------
  RCP<Epetra_CrsMatrix>
  GetMatrix()
  {
    return Matrix_;
  }
  // --------------------------------------------------------------------------
  bool
  computeF (const Epetra_Vector& x,
            Epetra_Vector& f,
            NOX::Epetra::Interface::Required::FillType F)
  {
    // Reset the diagonal entries.
    double diag = 2.0/(hx_*hx_) + 2.0/(hy_*hy_);

    int NumMyElements = Matrix_->Map().NumMyElements();

    // Get the list of the global elements that belong to my MPI process.
    int* MyGlobalElements = Matrix_->Map ().MyGlobalElements ();

    // Update the diagonal entry of the matrix.
    for (int i = 0; i < NumMyElements; ++i)
      Matrix_->ReplaceGlobalValues (MyGlobalElements[i], 1, &diag, MyGlobalElements+i);

    // Sparse matrix-vector product.
    // Interprocess communication happens here.
    Matrix_->Multiply(false, x, f);

    // Include the contribution from the current diagonal entry.
    for (int i = 0; i < NumMyElements; ++i)
      f[i] += lambda_*exp(x[i]);

    return true;
  }
  // --------------------------------------------------------------------------
  bool
  computeJacobian(const Epetra_Vector& x,
                  Epetra_Operator& Jac
                  )
  {
    double diag = 2.0/(hx_*hx_) + 2.0/(hy_*hy_);
    int NumMyElements = Matrix_->Map ().NumMyElements ();
    // Get the list of the global elements that belong to my MPI process.
    int* MyGlobalElements = Matrix_->Map ().MyGlobalElements ();
    for (int i = 0; i < NumMyElements; ++i)
    {
      // Update the current diagonal entry.
      double newdiag = diag + lambda_*exp(x[i]);
      Matrix_->ReplaceGlobalValues (MyGlobalElements[i], 1,
            &newdiag, MyGlobalElements+i);
    }
    return true;
  }
  // --------------------------------------------------------------------------
  bool
  computePreconditioner(const Epetra_Vector & x,
                        Epetra_Operator & M,
                        Teuchos::ParameterList* params
                        )
  {
    throw std::runtime_error ("*** MyProblem does not implement "
            "computing an explicit preconditioner from an "
            "Epetra_Operator ***");
  }
  // --------------------------------------------------------------------------
protected:
private:
  const int nx_, ny_;
  const double hx_, hy_;
  Teuchos::RCP<Epetra_CrsMatrix> Matrix_;
  const double lambda_;
};
// ==========================================================================
// Test driver routine.
int
main (int argc, char **argv)
{

#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
  Epetra_MpiComm Comm(MPI_COMM_WORLD);
#else
  Epetra_SerialComm Comm;
#endif

  const RCP<Teuchos::FancyOStream> out =
      Teuchos::VerboseObjectBase::getDefaultOStream();

  bool success = false;

  try {
    // Parameters for setting up the nonlinear PDE.  The 2-D regular
    // mesh on which the PDE's discretization is defined is nx by ny
    // (internal nodes; we assume Dirichlet boundary conditions have
    // been condensed out).
    const int nx = 200;
    const int ny = 200;
    const double lambda = 1.0;

    RCP<MyProblem> problem = rcp(new MyProblem(nx, ny, lambda, Comm));

    // Prepare the initial guess vector.  It should be a vector in the
    // domain of the nonlinear problem's matrix.
    Epetra_Vector InitialGuess(problem->GetMatrix()->OperatorDomainMap());

    // Make the starting solution a zero vector.
    InitialGuess.PutScalar(0.0);

    // Create the top-level parameter list to control NOX.
    //
    // "parameterList" (lowercase initial "p") is a "nonmember
    // constructor" that returns an RCP<ParameterList> with the
    // given name.
    RCP<ParameterList> params = parameterList ("NOX");

    // Tell the nonlinear solver to use line search.
    params->set ("Nonlinear Solver", "Line Search Based");

    // Set the printing parameters in the "Printing" sublist.
    ParameterList& printParams = params->sublist("Printing");
    printParams.set ("MyPID", Comm.MyPID());
    printParams.set ("Output Precision", 3);
    printParams.set ("Output Processor", 0);
    printParams.set ("Output Information", NOX::Utils::OuterIteration);
    //printParams.set ("Output Information",
    //             NOX::Utils::OuterIteration +
    //             NOX::Utils::OuterIterationStatusTest +
    //             NOX::Utils::InnerIteration +
    //             NOX::Utils::LinearSolverDetails +
    //             NOX::Utils::Parameters +
    //             NOX::Utils::Details +
    //             NOX::Utils::Warning +
    //                         NOX::Utils::Debug +
    //             NOX::Utils::TestDetails +
    //             NOX::Utils::Error);

    // Line search parameters.
    ParameterList& searchParams = params->sublist ("Line Search");
    searchParams.set ("Method", "More'-Thuente");

    // Parameters for picking the search direction.
    Teuchos::ParameterList& dirParams = params->sublist ("Direction");
    // Use Newton's method to pick the search direction.
    dirParams.set("Method", "Newton");

    // Parameters for Newton's method.
    ParameterList& newtonParams = dirParams.sublist ("Newton");
    newtonParams.set("Forcing Term Method", "Constant");

    // Newton's method invokes a linear solver repeatedly.
    // Set the parameters for the linear solver.
    ParameterList& lsParams = newtonParams.sublist ("Linear Solver");
    lsParams.set("Aztec Solver", "CG");
    lsParams.set("Max Iterations", 800);
    lsParams.set("Tolerance", 1e-10);
    lsParams.set("Output Frequency", 10);
    lsParams.set("Aztec Preconditioner", "ml");

    RCP<Epetra_CrsMatrix> A = problem->GetMatrix();

    // Our MyProblem implements both Required and
    // Jacobian, so we can use the same object for each.
    RCP<NOX::Epetra::Interface::Required> iReq = problem;
    RCP<NOX::Epetra::Interface::Jacobian> iJac = problem;
    RCP<NOX::Epetra::Interface::Preconditioner> iPrec = problem;

    RCP<NOX::Epetra::LinearSystemAztecOO> linSys =
      rcp (new NOX::Epetra::LinearSystemAztecOO (printParams, lsParams,
                   iReq, iJac, A, InitialGuess));

    // Need a NOX::Epetra::Vector for constructor.
    NOX::Epetra::Vector noxInitGuess (InitialGuess, NOX::DeepCopy);
    RCP<NOX::Epetra::Group> group =
      rcp (new NOX::Epetra::Group (printParams, iReq, noxInitGuess, linSys));

    //
    // Set up NOX's iteration stopping criteria ("status tests").
    //

    // ||F(X)||_2 / N < 1.0e-10, where N is the length of F(X).
    //
    // NormF has many options for setting up absolute vs. relative
    // (scaled by the norm of the initial guess) tolerances, scaling or
    // not scaling by the length of F(X), and choosing a different norm
    // (we use the 2-norm here).
    RCP<NOX::StatusTest::NormF> testNormF =
      rcp (new NOX::StatusTest::NormF (1.0e-10));

    // At most 20 (nonlinear) iterations.
    RCP<NOX::StatusTest::MaxIters> testMaxIters =
      rcp (new NOX::StatusTest::MaxIters (20));

    // Combine the above two stopping criteria (normwise convergence,
    // and maximum number of nonlinear iterations).  The result tells
    // NOX to stop if at least one of them is satisfied.
    RCP<NOX::StatusTest::Combo> combo =
      rcp (new NOX::StatusTest::Combo (NOX::StatusTest::Combo::OR,
               testNormF, testMaxIters));

    // Create the NOX nonlinear solver.
    RCP<NOX::Solver::Generic> solver =
      NOX::Solver::buildSolver (group, combo, params);

    // Solve the nonlinear system.
    NOX::StatusTest::StatusType status = solver->solve();

    // Print the result.
    if (Comm.MyPID() == 0) {
      *out << endl << "-- Parameter List From Solver --" << endl;
      solver->getList ().print (cout);
    }

    // Get the Epetra_Vector with the final solution from the solver.
    const NOX::Epetra::Group& finalGroup =
      dynamic_cast<const NOX::Epetra::Group&> (solver->getSolutionGroup ());

    const Epetra_Vector& finalSolution =
      dynamic_cast<const NOX::Epetra::Vector&>(finalGroup.getX()).getEpetraVector();

    //// Add a barrier and flush cout, just to make it less likely that
    //// the output will get mixed up when running with multiple MPI
    //// processes.
    //Comm.Barrier ();
    //cout.flush ();
    //// Add a barrier and flush cout, so that the above header line
    //// appears before the rest of the vector data.
    //*out << "Computed solution : " << endl;
    //Comm.Barrier ();
    //cout.flush ();
    //*out << finalSolution;

    success = true;
  }
  TEUCHOS_STANDARD_CATCH_STATEMENTS(true, *out, success);

#ifdef HAVE_MPI
  MPI_Finalize();
#endif
  return EXIT_SUCCESS;
}
// ==========================================================================
