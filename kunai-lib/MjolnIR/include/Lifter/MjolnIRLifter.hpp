//------------------------------------------------------------------- -*- cpp -*-
// Kunai-static-analyzer: library for doing analysis of dalvik files
// @author Farenain <kunai.static.analysis@gmail.com>
//
// @file MjolnIRLifter.hpp
// @brief Lifter for MjolnIR Dialect. The purpose of this class is
// from a MethodAnalysis or ClassAnalysis object from Kunai, generate
// a module from MLIR.

#ifndef LIFTER_MJOLNIRLIFTER_HPP
#define LIFTER_MJOLNIRLIFTER_HPP

/// MjolnIR includes
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
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>

#include <unordered_map>
#include <utility>
#include <set>

namespace KUNAI
{
namespace MjolnIR
{
    class Lifter
    {
    public:
        struct BasicBlockDef
        {
            /// Map a register to its definition in IR
            mlir::DenseMap<std::uint32_t, mlir::Value> Defs;

            std::set<std::uint32_t> required;

            /// Block is sealed, means no more predecessors will be
            /// added nor analyzed.
            unsigned Analyzed : 1;

            BasicBlockDef() : Analyzed(0) {}
        };

    private:
        /// @brief A map to keep the definitions of variables, and
        /// know if a basic block is completely analyzed
        mlir::DenseMap<KUNAI::DEX::DVMBasicBlock*, BasicBlockDef> CurrentDef;

        /// @brief Map for the Kunai basic blocks and the
        /// mlir blocks
        std::unordered_map<KUNAI::DEX::DVMBasicBlock*,
                           mlir::Block*> map_blocks;
        
        
        /// @brief Write a declaration of a local register, this will be
        /// used for local value analysis
        /// @param BB block where we find the assignment
        /// @param Reg register written
        /// @param Val 
        void writeLocalVariable(KUNAI::DEX::DVMBasicBlock* BB,
                                std::uint32_t Reg,
                                mlir::Value Val)
        {
            assert(BB && "Basic Block does not exist");
            assert(Val && "Value does not exist");
            CurrentDef[BB].Defs[Reg] = Val;
        }

        /// @brief Read a local variable from the current basic block
        /// @param BB basic block where to retrieve the data
        /// @param BBs basic blocks to retrieve the predecessors and successors
        /// @param Reg register to retrieve its Value
        /// @return value generated from an instruction.
        mlir::Value readLocalVariable(KUNAI::DEX::DVMBasicBlock* BB,
                                        KUNAI::DEX::BasicBlocks& BBs,
                                        std::uint32_t Reg)
        {
            assert(BB && "Basic Block does not exist");
            auto Val = CurrentDef[BB].Defs.find(Reg);
            /// if the block has the value, return it
            if (Val != CurrentDef[BB].Defs.end())
                return Val->second;
            /// if it doesn't have the value, it becomes required for
            /// us too
            return readLocalVariableRecursive(BB, BBs, Reg);
        }

        mlir::Value readLocalVariableRecursive(KUNAI::DEX::DVMBasicBlock* BB,
                                                 KUNAI::DEX::BasicBlocks& BBs,
                                                 std::uint32_t Reg);
        


        /// @brief Reference to an MLIR Context
        mlir::MLIRContext & context;

        /// @brief Module to return on lifting process
        mlir::ModuleOp Module;

        /// @brief The builder is a helper class to create IR inside a function. The builder
        /// is stateful, in particular it keeps an "insertion point": this is where
        /// the next operations will be introduced.
        mlir::OpBuilder builder;

        /// @brief In case an instruction has some error while lifting it
        /// generate an exception or generate a NOP instruction
        bool gen_exception;

        /// @brief Method currently analyzed, must be updated for each analyzed method
        KUNAI::DEX::MethodAnalysis * current_method;
        /// @brief Basic block currently analyzed, must be updated for each basic block analyzed
        KUNAI::DEX::DVMBasicBlock * current_basic_block;

        std::string module_name;

        //===----------------------------------------------------------------------===//
        // Some generators methods
        //===----------------------------------------------------------------------===//

        /// @brief Return an mlir::Type from a Fundamental type of Dalvik
        /// @param fundamental fundamental type of Dalvik
        /// @return different type depending on input
        mlir::Type get_type(KUNAI::DEX::DVMFundamental *fundamental);

        /// @brief Generic generator method for all DVMType
        /// @param type type to obtain the mlir::Type
        /// @return an mlir::Type from the dalvik type
        mlir::Type get_type(KUNAI::DEX::DVMType * type);

        /// @brief Given a prototype generate the types in MLIR
        /// @param proto prototype of the method
        /// @return vector with the generated types from the parameters
        llvm::SmallVector<mlir::Type> gen_prototype(KUNAI::DEX::ProtoID * proto);

        /// @brief Generate a MethodOp from a EncodedMethod given
        /// @param method pointer to an encoded method to generate a MethodOp
        /// @return method operation from Dalvik
        ::mlir::KUNAI::MjolnIR::MethodOp get_method(KUNAI::DEX::MethodAnalysis * M);


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

        /// @brief Lift an instruction of type Instruction22
        /// @param instr instruction to lift
        void gen_instruction(KUNAI::DEX::Instruction22c * instr);

        /// @brief Generate the IR from an instruction
        /// @param instr instruction from Dalvik to generate the IR
        void gen_instruction(KUNAI::DEX::Instruction *instr);

        /// @brief Generate a block into an mlir::Block*, we will lift each instruction.
        /// @param bb DVMBasicBlock to lift
        /// @param method method where the basic block is
        void gen_block(KUNAI::DEX::DVMBasicBlock* bb);

        /// @brief Generate a MethodOp from a MethodAnalysis
        /// @param method MethodAnalysis object to lift
        void gen_method(KUNAI::DEX::MethodAnalysis* method);
    
    public:

        /// @brief Constructor of Lifter
        /// @param context context from MjolnIR
        /// @param gen_exception generate a exception or nop instruction
        Lifter(mlir::MLIRContext & context, bool gen_exception)
            : context(context), builder(&context), gen_exception(gen_exception)
        {
            context.getOrLoadDialect<::mlir::KUNAI::MjolnIR::MjolnIRDialect>();
            context.getOrLoadDialect<::mlir::cf::ControlFlowDialect>();
        }

        /// @brief Generate a ModuleOp with the lifted instructions from a MethodAnalysis
        /// @param methodAnalysis method analysis to lift to MjolnIR
        /// @return reference to ModuleOp with the lifted instructions
        mlir::OwningOpRef<mlir::ModuleOp> mlirGen(KUNAI::DEX::MethodAnalysis* methodAnalysis);
    };
} // namespace MjolnIR
} // namespace KUNAI


#endif // LIFTER_MJOLNIRLIFTER_HPP