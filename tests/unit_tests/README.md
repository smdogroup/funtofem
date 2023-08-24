### Unit Tests ###
All tests under the `unit_tests` directory are included in the github workflow and are ran for each PR to the main repo. The tests fall under the following main categories.
* `basic_functionality` - Construct classes and write input/output files.
* `framework` - Test steady-state derivatives for fully-coupled and oneway drivers using TACS + test solvers. Considers multiple scenarios and composite functions too.
* `framework_unsteady` - Test unsteady derivatives with test solvers and the `TacsUnsteadyInterface`. Also considers shape derivatives of the unsteady analysis.
* `shape` - Test coordinate derivatives of the fully coupled and oneway drivers for steady-state analysis. Also considers ESP/CAPS shape derivatives with the `TacsAim`.
* `transfer_scheme` - Test the partial derivatives and adjoint products of each transfer scheme, including MELD. Single proc and multi-proc subcases are considered.