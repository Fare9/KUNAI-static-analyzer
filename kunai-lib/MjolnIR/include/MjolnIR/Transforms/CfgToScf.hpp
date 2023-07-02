//------------------------------------------------------------------- -*- cpp -*-
// Kunai-static-analyzer: library for doing analysis of dalvik files
// @author Farenain <kunai.static.analysis@gmail.com>
//
// @file CfgToScf.hpp
// @brief Implementations for Kunai from the CfgToScf pass from numba project.
// Numba project contains various presentations and slides:
//  * Slides 2021: https://drive.google.com/file/d/114r8KHlPf1eyZiIX5ce8-ckm40xXzUzP/view
//  * Presentation 2021: https://drive.google.com/file/d/1C6ecGtSK9-c_LuIT7CdHp-BQl3IBqpmV/view
//  * Slides 2023: https://discourse.llvm.org/uploads/short-url/1EqK3iG8k3icbrHwHSZQA0sNQir.pdf
//  * Repository: https://github.com/numba/numba-mlir

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