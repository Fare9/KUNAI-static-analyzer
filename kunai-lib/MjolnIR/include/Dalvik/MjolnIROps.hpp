//------------------------------------------------------------------- -*- cpp -*-
// Kunai-static-analyzer: library for doing analysis of dalvik files
// @author Farenain <kunai.static.analysis@gmail.com>
//
// @file MjolnIROps.hpp

#ifndef DALVIK_MJOLNIROPS_HPP
#define DALVIK_MJOLNIROPS_HPP

#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/FunctionInterfaces.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/Interfaces/CallInterfaces.h>
#include <mlir/Interfaces/CastInterfaces.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>


#define GET_OP_CLASSES
#include "Dalvik/MjolnIROps.h.inc"


#endif // DALVIK_MJOLNIROPS_HPP