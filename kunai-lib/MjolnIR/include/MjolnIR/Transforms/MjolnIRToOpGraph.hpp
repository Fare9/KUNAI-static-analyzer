//------------------------------------------------------------------- -*- cpp -*-
// Kunai-static-analyzer: library for doing analysis of dalvik files
// @author Farenain <kunai.static.analysis@gmail.com>
//
// @file MjolnIRToOpGraph.hpp
// @brief Pass for creating a Dot file from a Module of MjolnIR, this will be a version
// of the one from MLIR source code.

#ifndef TRANSFORMS_MJOLNIRTOOPGRAPH_HPP
#define TRANSFORMS_MJOLNIRTOOPGRAPH_HPP

#include <memory>

#include <mlir/Support/LLVM.h>
#include <llvm/Support/raw_ostream.h>

namespace mlir
{
class Pass;
} // namespace mlir

namespace KUNAI
{
namespace MjolnIR
{
std::unique_ptr<mlir::Pass> createMjolnIROpGraphPass(mlir::raw_ostream &os = llvm::errs());
} // namespace MjolnIR
} // namespace KUNAI



#endif