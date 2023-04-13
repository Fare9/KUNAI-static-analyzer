//------------------------------------------------------------------- -*- cpp -*-
// Kunai-static-analyzer: library for doing analysis of dalvik files
// @author Farenain <kunai.static.analysis@gmail.com>
//
// @file MjolnIRDialect.cpp


#include "Dalvik/MjolnIRTypes.hpp"

#include "Dalvik/MjolnIRDialect.hpp"
#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>
#include <llvm/ADT/TypeSwitch.h>

using namespace ::mlir::KUNAI::MjolnIR;

#define GET_TYPEDEF_CLASSES
#include "Dalvik/MjolnIROpsTypes.cpp.inc"

void MjolnIRDialect::registerTypes()
{
    addTypes<
#define GET_TYPEDEF_LIST
#include "Dalvik/MjolnIROpsTypes.cpp.inc"
        >();
}
