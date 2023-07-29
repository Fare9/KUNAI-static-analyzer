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
#include "MjolnIR/Dalvik/MjolnIRDialect.hpp"
#include "MjolnIR/Dalvik/MjolnIRTypes.hpp"
#include "MjolnIR/Dalvik/MjolnIROps.hpp"

/// KUNAI includes
#include "Kunai/DEX/analysis/analysis.hpp"

/// MLIR includes
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>

#include <unordered_map>
#include <utility>
#include <vector>
#include <stack>

namespace KUNAI
{
namespace MjolnIR
{
    class Lifter
    {
    public:
        using edge_t = std::pair<KUNAI::DEX::DVMBasicBlock*, KUNAI::DEX::DVMBasicBlock*>;

        /// @brief Definitions inside of a basic block of the registers, and the values
        struct BasicBlockDef
        {
            /// Map a register to its definition in IR
            mlir::DenseMap<std::uint32_t, mlir::Value> Defs;

            std::set<std::uint32_t> required;

            mlir::DenseMap<
                edge_t,
                mlir::SmallVector<mlir::Value,4>> jmpParameters;
            

            /// Block is sealed, means no more predecessors will be
            /// added nor analyzed.
            unsigned Analyzed : 1;

            BasicBlockDef() : Analyzed(0) {}
        };

        /// @brief Context for the analysis where we save the current method
        /// and the current basic blocks.
        struct AnalysisContext
        {
            /// @brief Method currently analyzed, must be updated for each analyzed method
            KUNAI::DEX::MethodAnalysis * current_method;
            /// @brief Basic block currently analyzed, must be updated for each basic block analyzed
            KUNAI::DEX::DVMBasicBlock * current_basic_block;
            /// @brief insertion point to be restored
            mlir::OpBuilder::InsertPoint ip;
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
        /// @param Val last value for register in basic block
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

        /// @brief In case the variable is not in the current basic block,
        /// this function is called, this function will retrieve the predecessors
        /// of the basic block and try to look for the value in those basic
        /// blocks
        /// @param BB analyzed basic block
        /// @param BBs reference to all the basic blocks in a structure
        /// @param Reg register we are looking for
        /// @return last value defined for the register.
        mlir::Value readLocalVariableRecursive(KUNAI::DEX::DVMBasicBlock* BB,
                                                 KUNAI::DEX::BasicBlocks& BBs,
                                                 std::uint32_t Reg);
        
        /// @brief For all the basic blocks, set the correct basic block parameters.
        /// @param BBs reference to all basic blocks
        /// @param block analyzed basic block
        void fillBlockArgs(KUNAI::DEX::BasicBlocks &BBs, KUNAI::DEX::DVMBasicBlock* block);

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


        std::stack<AnalysisContext> scope_context;
        /// @brief A context for the analysis, it contains the current analyzed
        /// method and the current analyzed basic block, it can be pushed into
        /// an stack and later retrieved.
        AnalysisContext analysis_context;

        /// @brief name of the module where we will write all the methods
        std::string module_name;

        // types from DVM for not generating it many times
        ::mlir::KUNAI::MjolnIR::DVMVoidType voidType;
        ::mlir::IntegerType boolType;
        ::mlir::IntegerType byteType;
        ::mlir::IntegerType charType;
        ::mlir::IntegerType shortType;
        ::mlir::IntegerType intType;
        ::mlir::IntegerType longType;
        ::mlir::FloatType floatType;
        ::mlir::FloatType doubleType;
        ::mlir::KUNAI::MjolnIR::DVMObjectType strObjectType;


        //===----------------------------------------------------------------------===//
        // Some generators methods
        //===----------------------------------------------------------------------===//
        mlir::Type get_array(KUNAI::DEX::DVMType * type);

        /// @brief Return an mlir::Type from an Array type of Dalvik
        /// @param array array type of Dalvik
        /// @return a mlir::Type which contains data from an array
        mlir::Type get_type(KUNAI::DEX::DVMArray *array);

        /// @brief Return an mlir::Type from a Fundamental type of Dalvik
        /// @param fundamental fundamental type of Dalvik
        /// @return different type depending on input
        mlir::Type get_type(KUNAI::DEX::DVMFundamental *fundamental);

        /// @brief Return an mlir::Type from a class type of Dalvik
        /// @param cls class type of Dalvik
        /// @return a mlir::Type which contains as attribute the name of the class
        mlir::Type get_type(KUNAI::DEX::DVMClass *cls);

        /// @brief Generic generator method for all DVMType
        /// @param type type to obtain the mlir::Type
        /// @return an mlir::Type from the dalvik type
        mlir::Type get_type(KUNAI::DEX::DVMType * type);

        /// @brief Given a prototype generate the types in MLIR
        /// @param proto prototype of the method
        /// @return vector with the generated types from the parameters
        llvm::SmallVector<mlir::Type> gen_prototype(KUNAI::DEX::ProtoID * proto);

        /// @brief Generate a FuncOp from a MethodAnalysis given
        /// @param method pointer to an encoded method to generate a MethodOp
        /// @return method operation from Dalvik
        ::mlir::func::FuncOp get_method(KUNAI::DEX::MethodAnalysis * M);

        /// @brief Initialize possible used types and other necessary stuff
        void init();

        /// @brief Cast to the indicated type
        /// @param reg register to cast
        /// @param type type to cast to
        /// @param loc location for cast operation
        void cast_to_type(std::uint32_t reg,
                          mlir::Type type, 
                          mlir::FileLineColLoc loc);
        
        /// @brief Check types from registers, and cast to the appropiate type
        /// @param reg1 first register to check
        /// @param reg2 second register to check
        /// @param loc location for the cast operation
        void cast_to_type(std::uint32_t reg1,
                          std::uint32_t reg2,
                          mlir::FileLineColLoc loc);

        /// @brief Cast two registers to the provided type
        /// @param reg1 first register to cast
        /// @param reg2 second register to cast
        /// @param type type to cast to
        /// @param loc location for the cast operation
        void cast_to_type(std::uint32_t reg1,
                          std::uint32_t reg2,
                          mlir::Type type,
                          mlir::FileLineColLoc loc);

        /// @brief Cast a given value to the provided type
        /// @param curr_value value to be casted
        /// @param type target type
        /// @param loc location for the cast operation
        /// @return casted type, or previous value if no change.
        mlir::Value cast_value(mlir::Value curr_value,
                        mlir::Type type, 
                        mlir::FileLineColLoc loc);
    public:
        //===----------------------------------------------------------------------===//
        // Lifting instructions, these class functions will be specialized for the
        // different function types.
        //===----------------------------------------------------------------------===//

        /// @brief Lift an instruction of the type Instruction31c
        /// @param instr instruction to lift
        void gen_instruction(KUNAI::DEX::Instruction31c *instr);

        /// @brief Lift an instruction of the type Instruction31i
        /// @param instr instruction to lift
        void gen_instruction(KUNAI::DEX::Instruction31i *instr);
        
        /// @brief Lift an instruction of the type Instruction32x
        /// @param instr instruction to lift
        void gen_instruction(KUNAI::DEX::Instruction32x *instr);

        /// @brief Lift an instruction of the type Instruction22x
        /// @param instr instruction to lift
        void gen_instruction(KUNAI::DEX::Instruction22x *instr);
        
        /// @brief Lift an instruction of the type Instruction21c
        /// @param instr instruction to lift
        void gen_instruction(KUNAI::DEX::Instruction21c *instr);
        
        /// @brief Lift an instruction of the type Instruction35c
        /// @param instr instruction to lift
        void gen_instruction(KUNAI::DEX::Instruction35c *instr);
        
        /// @brief Lift an instruction of the type Instruction51l
        /// @param instr instruction to lift
        void gen_instruction(KUNAI::DEX::Instruction51l *instr);

        /// @brief Lift an instruction of the type Instruction21h
        /// @param instr instruction to lift
        void gen_instruction(KUNAI::DEX::Instruction21h *instr);

        /// @brief Lift an instruction of the type Instruction21s
        /// @param instr instruction to lift
        void gen_instruction(KUNAI::DEX::Instruction21s *instr);

        /// @brief Lift an instruction of the type Instruction11n
        /// @param instr instruction to lift
        void gen_instruction(KUNAI::DEX::Instruction11n *instr);
        
        /// @brief Lift an instruction of the type Instruction10x
        /// @param instr instruction to lift
        void gen_instruction(KUNAI::DEX::Instruction10x *instr);

        /// @brief Lift an instruction of the type Instruction10t
        /// @param instr instruction to lift
        void gen_instruction(KUNAI::DEX::Instruction10t *instr);

        /// @brief Lift an instruction of the type Instruction20t
        /// @param instr instruction to lift
        void gen_instruction(KUNAI::DEX::Instruction20t *instr);

        /// @brief Lift an instruction of the type Instruction30t
        /// @param instr instruction to lift
        void gen_instruction(KUNAI::DEX::Instruction30t *instr);

        /// @brief Lift an instruction of the type Instruction31t
        /// @param instr instruction to lift
        void gen_instruction(KUNAI::DEX::Instruction31t *instr);

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

        /// @brief Constructor of Lifter
        /// @param context context from MjolnIR
        /// @param gen_exception generate a exception or nop instruction
        Lifter(mlir::MLIRContext & context, bool gen_exception)
            : context(context), builder(&context), gen_exception(gen_exception)
        {
            init();
        }

        /// @brief Generate a ModuleOp with the lifted instructions from a MethodAnalysis
        /// @param methodAnalysis method analysis to lift to MjolnIR
        /// @return reference to ModuleOp with the lifted instructions
        mlir::OwningOpRef<mlir::ModuleOp> mlirGen(KUNAI::DEX::MethodAnalysis* methodAnalysis);
    };
} // namespace MjolnIR
} // namespace KUNAI


#endif // LIFTER_MJOLNIRLIFTER_HPP