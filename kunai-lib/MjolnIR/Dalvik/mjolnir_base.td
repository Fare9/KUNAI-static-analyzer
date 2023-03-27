//------------------------------------------------------------------- -*- tablegen -*-
// Kunai-static-analyzer: library for doing analysis of dalvik files
// @author Farenain <kunai.static.analysis@gmail.com>
// @author Ernesto Java <javaernesto@gmail.com>
//
// @file mjolnir.td
// @brief Specification of tablegen for the IR, here all the operations
// will be declared as well of all necessary types for this IR. This time
// we will use the framework MLIR for creating the IR.

#ifndef KUNAI_MJOLNIR_MLIR_MJOLNIR_TD
#define KUNAI_MJOLNIR_MLIR_MJOLNIR_TD

include "mlir/IP/OpBase.td"

// Provide first a definition for the dialect in the ODS
// framework where we will add all the operations.
def MjolnIR_Dialect : Dialect
{
    let name = "MjolnIR";
    let summary = "This IR is planned to cover instructions from Dalvik VM";
    let cppNamespace = "::KUNAI::MjolnIR";
}



// Base class for the operations of MjolnIR. Operations will
// inherit from base `Op` class from MLIR. It provides:
//      * Parent dialect of operation
//      * Mnemonic for the operation, or name without dialect prefix
//      * List of traits for the operation
class MjolnIR_Op<string mnemonic, list<Traits> traits = []> :
    Op<MjolnIR_Dialect, mnemonic, traits>;

#endif