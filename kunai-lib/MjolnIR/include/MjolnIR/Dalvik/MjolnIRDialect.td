//------------------------------------------------------------------- -*- tablegen -*-
// Kunai-static-analyzer: library for doing analysis of dalvik files
// @author Farenain <kunai.static.analysis@gmail.com>
//
// @file MjolnIRDialect.td
// @brief Main file for the Dialect of Dalvik with MLIR, here we will find basic
// definitions of main components, for example the dialect itself.

#ifndef DALVIK_MJOLNIRDIALECT_TD
#define DALVIK_MJOLNIRDIALECT_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"


//===----------------------------------------------------------------------===//
// MjolnIR dialect definition.
//===----------------------------------------------------------------------===//

def MjolnIR_Dialect : Dialect
{
    let name = "MjolnIR";
    let summary = "Dialect for the instructions of Dalvik Virtual Machine";
    let description = [{
      This dialect tries to represent the instructions from the Dalvik Virtual Machine
      together with the instruction we will find also the different types found on
      this bytecode (Fields, Methods, Classes).
    }];

    let cppNamespace = "::mlir::KUNAI::MjolnIR";

    let useDefaultTypePrinterParser = 1;
    let extraClassDeclaration = [{
        void registerTypes();
    }];
}

// Base class for the operations of MjolnIR. Operations will
// inherit from base `Op` class from MLIR. It provides:
//      * Parent dialect of operation
//      * Mnemonic for the operation, or name without dialect prefix
//      * List of traits for the operation
class MjolnIR_Op<string mnemonic, list<Trait> traits = []> :
    Op<MjolnIR_Dialect, mnemonic, traits>;


//===----------------------------------------------------------------------===//
// MjolnIR_FieldType
//===----------------------------------------------------------------------===//
def MjolnIR_FieldType : I32EnumAttr<
    "FieldType", "",
    [
        I32EnumAttrCase<"STATIC", 0, "static">,
        I32EnumAttrCase<"INSTANCE", 1, "instance">
    ]
> {
    let cppNamespace = "::mlir::KUNAI::MjolnIR";
}


#endif // DALVIK_MJOLNIRDIALECT_TD