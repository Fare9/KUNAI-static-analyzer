//------------------------------------------------------------------- -*- cpp -*-
// Kunai-static-analyzer: library for doing analysis of dalvik files
// @author Farenain <kunai.static.analysis@gmail.com>
//
// @file mjolnir_lifter.hpp
// @brief Lifter for MjolnIR, we will use the types from
// the MLIR

#ifndef LIFTER_MJOLNIR_LIFTER_HPP
#define LIFTER_MJOLNIR_LIFTER_HPP

#include "Dalvik/MjolnIRDialect.hpp"

#include "Kunai/DEX/analysis/analysis.hpp"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/ScopedHashTable.h"

#include <utility>

namespace KUNAI
{
    namespace MjolnIR
    {
        class MjolnIRLifter
        {
            /// @brief A Context from MLIR
            mlir::MLIRContext &context;

            /// @brief A mlir::ModuleOp from the Dalvik file, it also stores all the Methods
            mlir::ModuleOp Module;

            /// @brief The builder is a helper class to create IR inside a function. The builder
            /// is stateful, in particular it keeps an "insertion point": this is where
            /// the next operations will be introduced.
            mlir::OpBuilder builder;

            /// @brief DialectRegistry
            mlir::DialectRegistry registry;

            /// @brief ScopedHashTable for registers
            llvm::ScopedHashTable<std::uint32_t, std::pair<mlir::Value, KUNAI::DEX::EncodedMethod*>>
                registerTable;

            using RegisterTableScopeT = 
                llvm::ScopedHashTable<std::uint32_t, std::pair<mlir::Value, KUNAI::DEX::EncodedMethod*>>;

            /// @brief A mapping for the functions that have been code generated to MLIR.
            llvm::StringMap<mlir::KUNAI::MjolnIR::MethodOp> methodMap;

            /// @brief Declare a register in an scope for the SSA
            /// @param reg register to declare in the scope
            /// @param MA encoded method to set the scope
            /// @param value value of the register
            /// @return failure or success
            mlir::LogicalResult declareReg(std::uint32_t reg, KUNAI::DEX::EncodedMethod* EM, mlir::Value value);

            /// @brief Return an mlir::Type from a Fundamental type of Dalvik
            /// @param fundamental fundamental type of Dalvik
            /// @return different type depending on input
            mlir::Type get_type(KUNAI::DEX::DVMFundamental *fundamental);

            /// @brief Return an mlir::Type for any of the types from Dalvik
            /// @param type type from Dalvik
            /// @return mlir::Type for the different DVMTypes
            mlir::Type get_type(KUNAI::DEX::DVMType *type);

            /// @brief Generate a small vector of Types from the prototypes of
            /// a Method
            /// @param proto prototype to generate the small vector
            /// @return vector of types.
            llvm::SmallVector<mlir::Type> gen_prototype(KUNAI::DEX::ProtoID *proto);

            /// @brief Generate a MethodOp from a EncodedMethod given
            /// @param method pointer to an encoded method to generate a MethodOp
            /// @return method operation from Dalvik
            ::mlir::KUNAI::MjolnIR::MethodOp get_method(KUNAI::DEX::EncodedMethod *encoded_method);

            /// @brief Generate the IR from an instruction
            /// @param instr instruction from Dalvik to generate the IR
            void gen_instruction(KUNAI::DEX::Instruction * instr);

        public:
            /// @brief Constructor of MjolnIRLifter
            MjolnIRLifter(mlir::MLIRContext &context) : context(context), builder(&context)
            {
                registry.insert<::mlir::KUNAI::MjolnIR::MjolnIRDialect>();
            }

            /// @brief Generate a ModuleOp with the lifted instructions from a MethodAnalysis
            /// @param methodAnalysis method analysis to lift
            /// @return reference to ModuleOp with the lifted instructions
            mlir::OwningOpRef<mlir::ModuleOp> mlirGen(KUNAI::DEX::MethodAnalysis *methodAnalysis);
        };
    }
}

#endif