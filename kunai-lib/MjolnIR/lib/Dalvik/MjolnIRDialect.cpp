//------------------------------------------------------------------- -*- cpp -*-
// Kunai-static-analyzer: library for doing analysis of dalvik files
// @author Farenain <kunai.static.analysis@gmail.com>
//
// @file MjolnIRDialect.cpp

#include "Dalvik/MjolnIRDialect.hpp"
#include "Dalvik/MjolnIROps.hpp"
#include "Dalvik/MjolnIRTypes.hpp"

using namespace mlir;
using namespace ::mlir::KUNAI::MjolnIR;

// import the cpp generated from tablegen
#include "Dalvik/MjolnIROpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// MjolnIR Dialect
//===----------------------------------------------------------------------===//

// initialize the operations from those generated
// with tablegen
void MjolnIRDialect::initialize()
{
    addOperations<
#define GET_OP_LIST
#include "Dalvik/MjolnIROps.cpp.inc"
        >();
    registerTypes();
}