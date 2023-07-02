//------------------------------------------------------------------- -*- cpp -*-
// Kunai-static-analyzer: library for doing analysis of dalvik files
// @author Farenain <kunai.static.analysis@gmail.com>
//
// @file CfgToScf.hpp
// @brief Implementations for Kunai from the CfgToScf pass from numba project.

#ifndef MJOLNIR_TRANSFORMS_CFGTOSCF_HPP
#define MJOLNIR_TRANSFORMS_CFGTOSCF_HPP

#include <memory>

namespace mlir
{
class Pass;
} // namespace mlir

namespace KUNAI
{
namespace MjolnIR
{
std::unique_ptr<mlir::Pass> createMjolnIRCfgToScfgPass();
} // namespace MjolnIR
} // namespace KUNAI


#endif