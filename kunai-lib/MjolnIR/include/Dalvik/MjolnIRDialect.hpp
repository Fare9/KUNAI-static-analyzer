//------------------------------------------------------------------- -*- cpp -*-
// Kunai-static-analyzer: library for doing analysis of dalvik files
// @author Farenain <kunai.static.analysis@gmail.com>
//
// @file MjolnIRDialect.hpp
// @brief Definition of the dialect for Dalvik

#ifndef DALVIK_MJOLNIRDIALECT_HPP
#define DALVIK_MJOLNIRDIALECT_HPP

#include <mlir/IR/Dialect.h>

#include "Dalvik/MjolnIROpsDialect.h.inc"

#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/FunctionInterfaces.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/Interfaces/CallInterfaces.h>
#include <mlir/Interfaces/CastInterfaces.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

#define GET_TYPEDEF_CLASSES
#include "Dalvik/MjolnIROpsTypes.h.inc"


#define GET_OP_CLASSES
#include "Dalvik/MjolnIROps.h.inc"

#endif // DALVIK_MJOLNIRDIALECT_HPP