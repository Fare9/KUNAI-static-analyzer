//------------------------------------------------------------------- -*- cpp -*-
// Kunai-static-analyzer: library for doing analysis of dalvik files
// @author Farenain <kunai.static.analysis@gmail.com>
//
// @file mjolnir_lifter.hpp
// @brief Lifter for MjolnIR, we will use the types from
// the MLIR

#ifndef LIFTER_MJOLNIR_LIFTER_HPP
#define LIFTER_MJOLNIR_LIFTER_HPP

/// MjolnIR Includes
#include "Dalvik/MjolnIRDialect.hpp"
#include "Dalvik/MjolnIRTypes.hpp"
#include "Dalvik/MjolnIROps.hpp"

/// KUNAI includes
#include "Kunai/DEX/analysis/analysis.hpp"

/// MLIR includes
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/MLIRContext.h>

/// LLVM includes
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/ScopedHashTable.h>


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

            /// @brief ScopedHashTable for registers
            llvm::ScopedHashTable<std::uint32_t, std::pair<mlir::Value, KUNAI::DEX::EncodedMethod *>>
                registerTable;

            using RegisterTableScopeT =
                llvm::ScopedHashTableScope<std::uint32_t, std::pair<mlir::Value, KUNAI::DEX::EncodedMethod *>>;

            /// @brief A mapping for the functions that have been code generated to MLIR.
            llvm::StringMap<mlir::KUNAI::MjolnIR::MethodOp> methodMap;

            /// @brief File name for location
            llvm::StringRef file_name;

            /// @brief Declare a register in an scope for the SSA
            /// @param reg register to declare in the scope
            /// @param MA encoded method to set the scope
            /// @param value value of the register
            /// @return failure or success
            mlir::LogicalResult declareReg(std::uint32_t reg, KUNAI::DEX::EncodedMethod *EM, mlir::Value value);

            /// @brief Update a register value to keep the SSA form of the IR
            /// @param reg register to update
            /// @param value new value
            /// @return failure or success
            mlir::LogicalResult updateReg(std::uint32_t reg, mlir::Value value);

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

            //===----------------------------------------------------------------------===//
            // Lifting instructions, these class functions will be specialized for the
            // different function types.
            //===----------------------------------------------------------------------===//

            /// @brief Lift an instruction of the type Instruction23x
            /// @param instr instruction to lift
            void gen_instruction(KUNAI::DEX::Instruction23x *instr);

            /// @brief Lift an instruction of the type Instruction12x
            /// @param instr instruction to lift
            void gen_instruction(KUNAI::DEX::Instruction12x *instr);

            /// @brief Lift an instruction of type Instruction22s
            /// @param instr instruction to lift
            void gen_instruction(KUNAI::DEX::Instruction22s *instr);

            /// @brief Lift an instruction of type Instruction22b
            /// @param instr instruction to lift
            void gen_instruction(KUNAI::DEX::Instruction22b *instr);

            /// @brief Lift an instruction of type Instruction22t
            /// @param instr instruction to lift
            void gen_instruction(KUNAI::DEX::Instruction22t *instr);

            /// @brief Lift an instruction of type Instruction21t
            /// @param instr instruction to lift
            void gen_instruction(KUNAI::DEX::Instruction21t *instr);

            /// @brief Lift an instruction of type Instruction11x
            /// @param instr instruction to lift
            void gen_instruction(KUNAI::DEX::Instruction11x * instr);

            /// @brief Generate the IR from an instruction
            /// @param instr instruction from Dalvik to generate the IR
            void gen_instruction(KUNAI::DEX::Instruction *instr);

            /// @brief generate all the instructions from a block
            /// @param bb
            void gen_block(KUNAI::DEX::DVMBasicBlock *bb);

            /// @brief Bool indicating if generate exception or
            /// just create a Nop instruction
            bool gen_exception;

        public:
            /// @brief Constructor of MjolnIRLifter
            MjolnIRLifter(mlir::MLIRContext &context, bool gen_exception) : context(context),
                                                                            builder(&context),
                                                                            gen_exception(gen_exception)
            {
                // Load our Dialect in this MLIR Context
                context.getOrLoadDialect<::mlir::KUNAI::MjolnIR::MjolnIRDialect>();
            }

            /// @brief Generate a ModuleOp with the lifted instructions from a MethodAnalysis
            /// @param methodAnalysis method analysis to lift
            /// @return reference to ModuleOp with the lifted instructions
            mlir::OwningOpRef<mlir::ModuleOp> mlirGen(KUNAI::DEX::MethodAnalysis *methodAnalysis);
        };
    }
}

#endif