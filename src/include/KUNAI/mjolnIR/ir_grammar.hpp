/**
 * @file ir_grammar.hpp
 * @author Farenain
 * @brief Header file which includes all the grammar of the IR, this will be
 *        used as header file by all the intermediate representation cpp files.
 * @version 0.1
 * @date 2021-11-09
 *
 * @copyright Copyright (c) 2021
 *
 */

#ifndef IR_GRAMMAR_HPP
#define IR_GRAMMAR_HPP

#include <iostream>
#include <sstream>
#include <fstream>
#include <memory>
#include <vector>
#include <list>
#include <unordered_map>

// lifters
#include "KUNAI/mjolnIR/arch/ir_x86.hpp"
#include "KUNAI/mjolnIR/arch/ir_dalvik.hpp"

namespace KUNAI
{
    namespace MJOLNIR
    {
        class IRBlock;
        class IRStmnt;
        class IRUJmp;
        class IRCJmp;
        class IRRet;
        class IRNop;
        class IRSwitch;
        class IRExpr;
        class IRBinOp;
        class IRUnaryOp;
        class IRAssign;
        class IRPhi;
        class IRCall;
        class IRLoad;
        class IRStore;
        class IRZComp;
        class IRBComp;
        class IRNew;
        class IRType;
        class IRReg;
        class IRTempReg;
        class IRConstInt;
        class IRMemory;
        class IRString;
        class IRClass;
        class IRCallee;
        class IRField;

        using irblock_t = std::shared_ptr<IRBlock>;
        using irstmnt_t = std::shared_ptr<IRStmnt>;

        using irujmp_t = std::shared_ptr<IRUJmp>;
        /**
         * @brief Check if the given statement is an unconditional jump instruction.
         *
         * @param instr
         * @return irujmp_t or nullptr
         */
        irujmp_t unconditional_jump_ir(irstmnt_t &instr);

        using ircjmp_t = std::shared_ptr<IRCJmp>;
        /**
         * @brief Check if the given statement is a conditional jump instruction.
         *
         * @param instr
         * @return ircjmp_t or nullptr
         */
        ircjmp_t conditional_jump_ir(irstmnt_t &instr);

        using irret_t = std::shared_ptr<IRRet>;
        /**
         * @brief Check if given statement is a ret instruction.
         *
         * @param instr
         * @return irret_t or nullptr
         */
        irret_t ret_ir(irstmnt_t &instr);

        using irnop_t = std::shared_ptr<IRNop>;
        /**
         * @brief Check if given statement is a NOP instruction
         *
         * @param instr
         * @return irnop_t or nullptr
         */
        irnop_t nop_ir(irstmnt_t &instr);

        using irswitch_t = std::shared_ptr<IRSwitch>;
        /**
         * @brief Check if given statement is a switch instruction.
         *
         * @param instr
         * @return irswitch_t or nullptr
         */
        irswitch_t switch_ir(irstmnt_t &instr);

        using irexpr_t = std::shared_ptr<IRExpr>;
        /**
         * @brief Check if given statement is an expression instruction.
         *
         * @param instr
         * @return irexpr_t ornullptr
         */
        irexpr_t expr_ir(irstmnt_t &instr);

        using irbinop_t = std::shared_ptr<IRBinOp>;
        /**
         * @brief Check if given statement is a binary operation.
         *
         * @param instr
         * @return irbinop_t or nullptr
         */
        irbinop_t bin_op_ir(irstmnt_t &instr);

        using irunaryop_t = std::shared_ptr<IRUnaryOp>;
        /**
         * @brief Check if given statement is a unary operation.
         *
         * @param instr
         * @return irunaryop_t or nullptr
         */
        irunaryop_t unary_op_ir(irstmnt_t &instr);

        using irassign_t = std::shared_ptr<IRAssign>;
        /**
         * @brief Check if given statement is an assignment operation.
         *
         * @param instr
         * @return irassign_t or nullptr
         */
        irassign_t assign_ir(irstmnt_t &instr);

        using irphi_t = std::shared_ptr<IRPhi>;

        /**
         * @brief Check if given statement is a phi operation.
         *
         * @param instr
         * @return irphi_t or nullptr
         */
        irphi_t phi_ir(irstmnt_t &instr);

        using ircall_t = std::shared_ptr<IRCall>;
        /**
         * @brief Check if given statement is a call instruction.
         *
         * @param instr
         * @return ircall_t or nullptr
         */
        ircall_t call_ir(irstmnt_t &instr);

        using irload_t = std::shared_ptr<IRLoad>;
        /**
         * @brief Check if given statement is a load instruction.
         *
         * @param instr
         * @return irload_t or nullptr
         */
        irload_t load_ir(irstmnt_t &instr);

        using irstore_t = std::shared_ptr<IRStore>;
        /**
         * @brief Check if given statement is a store instruction.
         *
         * @param instr
         * @return irstore_t or nullptr
         */
        irstore_t store_ir(irstmnt_t &instr);

        using irzcomp_t = std::shared_ptr<IRZComp>;

        /**
         * @brief Check if given statement is a zero comparison.
         *
         * @param instr
         * @return irzcomp_t or nullptr
         */
        irzcomp_t zcomp_ir(irstmnt_t &instr);

        using irbcomp_t = std::shared_ptr<IRBComp>;

        /**
         * @brief Check if given statement is a binary comparison.
         *
         * @param instr
         * @return irbcomp_t or nullptr
         */
        irbcomp_t bcomp_ir(irstmnt_t &instr);

        /**
         * @brief Check if the given statements is one of the comparisons.
         *
         * @param instr
         * @return true
         * @return false
         */
        bool is_cmp(irstmnt_t &instr);

        using irnew_t = std::shared_ptr<IRNew>;
        /**
         * @brief check if given statement is a new statement.
         *
         * @param instr
         * @return irnew_t or nullptr
         */
        irnew_t new_ir(irstmnt_t &instr);

        using irtype_t = std::shared_ptr<IRType>;
        /**
         * @brief Check if given statement is type statement.
         *
         * @param instr
         * @return irtype_t or nullptr
         */
        irtype_t type_ir(irstmnt_t &instr);

        using irreg_t = std::shared_ptr<IRReg>;
        /**
         * @brief Check if the given statement object is
         *        a register.
         *
         * @param instr
         * @return irreg_t or nullptr
         */
        irreg_t register_ir(irstmnt_t &instr);

        using irtempreg_t = std::shared_ptr<IRTempReg>;
        /**
         * @brief Check if a given statement object is a temporal register.
         *
         * @param instr
         * @return irtempreg_t or nullptr
         */
        irtempreg_t temp_reg_ir(irstmnt_t &instr);

        using irconstint_t = std::shared_ptr<IRConstInt>;
        /**
         * @brief Check if given statement object is a constant integer.
         *
         * @param instr
         * @return irconstint_t or nullptr
         */
        irconstint_t const_int_ir(irstmnt_t &instr);

        using irmemory_t = std::shared_ptr<IRMemory>;
        /**
         * @brief Check if given statement object is a memory.
         *
         * @param instr
         * @return irmemory_t or nullptr
         */
        irmemory_t memory_ir(irstmnt_t &instr);

        using irstring_t = std::shared_ptr<IRString>;
        /**
         * @brief Check if given statement object is a string.
         *
         * @param instr
         * @return irstring_t or nullptr
         */
        irstring_t string_ir(irstmnt_t &instr);

        using irclass_t = std::shared_ptr<IRClass>;
        /**
         * @brief Check if given statement object is a class.
         *
         * @param instr
         * @return irclass_t or nullptr
         */
        irclass_t class_ir(irstmnt_t &instr);

        using ircallee_t = std::shared_ptr<IRCallee>;
        /**
         * @brief Check if given statement object is
         *        a callee method/function
         *
         * @param instr
         * @return ircallee_t or nullptr
         */
        ircallee_t callee_ir(irstmnt_t &instr);

        using irfield_t = std::shared_ptr<IRField>;
        /**
         * @brief Check if the given statement is a IRField.
         *
         * @param instr
         * @return irfield_t or nullptr
         */
        irfield_t field_ir(irstmnt_t &instr);

        class IRBlock
        {
        public:
            /**
             * @brief Constructor of IRBlock, this class represent blocks of statements.
             * @return void
             */
            IRBlock();

            /**
             * @brief Destructor of IRBlock, nothing to be done here.
             * @return void
             */
            ~IRBlock() = default;

            /**
             * @brief Add a statement at the beginning of the block of instructions,
             *        this will be useful to insert instructions like phi nodes.
             * @param statement statement to insert at the beginning of the statements.
             */
            void add_statement_at_beginning(irstmnt_t statement)
            {
                block_statements.insert(block_statements.begin(), statement);
            }

            /**
             * @brief Append one statement to the list, we use the class std::list as it allows us to
             *        insert instructions between other instructions easily.
             * @param statement: statement to append to the list.
             * @return void
             */
            void append_statement_to_block(irstmnt_t statement)
            {
                block_statements.push_back(statement);
            }

            /**
             * @brief Remove on of the statements given its position.
             * @param pos: position of the statement in the vector.
             * @return bool
             */
            bool delete_statement_by_position(size_t pos);

            /**
             * @brief Get the number of statements from the block.
             * @return size_t
             */
            size_t get_number_of_statements()
            {
                return block_statements.size();
            }

            /**
             * @brief Get the list of statements.
             * @return std::vector<irstmnt_t>
             */
            std::vector<irstmnt_t> &get_statements()
            {
                return block_statements;
            }

            /**
             * @brief Set the start_idx value from the block.
             *
             * @param start_idx
             */
            void set_start_idx(uint64_t start_idx)
            {
                this->start_idx = start_idx;
            }

            /**
             * @brief Set the end_idx value from the block.
             *
             * @param end_idx
             */
            void set_end_idx(uint64_t end_idx)
            {
                this->end_idx = end_idx;
            }

            /**
             * @brief Return the start_idx of the current block.
             *
             * @return uint64_t
             */
            uint64_t get_start_idx()
            {
                return start_idx;
            }

            /**
             * @brief Return the end_idx of the current block.
             *
             * @return uint64_t
             */
            uint64_t get_end_idx()
            {
                return end_idx;
            }

            /**
             * @brief Return a string with a representative name of the basic block.
             *
             * @return std::string
             */
            std::string get_name();

            /**
             * @brief Return a string representation of the basic block.
             *
             * @return std::string
             */
            std::string to_string();

            void set_phi_node()
            {
                phi_node = true;
            }

            void clear_phi_node()
            {
                phi_node = false;
            }

            bool contains_phi_node()
            {
                return phi_node;
            }

        private:
            bool phi_node = false;
            //! starting idx and last idx (used for jump calculation)
            uint64_t start_idx, end_idx;
            //! statements from the basic block.
            std::vector<irstmnt_t> block_statements;
        };

        class IRStmnt
        {
        public:
            enum stmnt_type_t
            {
                UJMP_STMNT_T,
                CJMP_STMNT_T,
                RET_STMNT_T,
                NOP_STMNT_T,
                SWITCH_STMNT_T,
                EXPR_STMNT_T,
                NONE_STMNT_T = 99 // used to finish the chain of statements
            };

            enum op_type_t
            {
                UJMP_OP_T,
                CJMP_OP_T,
                RET_OP_T,
                NOP_OP_T,
                SWITCH_OP_T,
                EXPR_OP_T,
                BINOP_OP_T,
                UNARYOP_OP_T,
                ASSIGN_OP_T,
                PHI_OP_T,
                CALL_OP_T,
                OP_T_OP_T,
                LOAD_OP_T,
                STORE_OP_T,
                ZCOMP_OP_T,
                BCOMP_OP_T,
                NEW_OP_T,
                TYPE_OP_T,
                REGISTER_OP_T,
                TEMP_REGISTER_OP_T,
                CONST_INT_OP_T,
                CONST_FLOAT_OP_T,
                FIELD_OP_T,
                MEM_OP_T,
                STRING_OP_T,
                CLASS_OP_T,
                CALLEE_OP_T,
                NONE_OP_T = 99 // used to finish the chain of statements
            };

            /**
             * @brief Constructor of IRStmnt.
             * @param stmnt_type type of the statement.
             * @param op_type type of the operation
             */
            IRStmnt(stmnt_type_t stmnt_type, op_type_t op_type);

            /**
             * @brief Destroy the IRStmnt object
             */
            virtual ~IRStmnt() = default;

            /**
             * @brief Get the type of the statement.
             * @return stmnt_type_t
             */
            stmnt_type_t get_statement_type()
            {
                return stmnt_type;
            }

            /**
             * @brief Return the type of the operation.
             *
             * @return op_type_t
             */
            op_type_t get_op_type()
            {
                return op_type;
            }

            /**
             * @brief Return correct string representation for IRStmnt.
             *
             * @return std::string
             */
            std::string to_string();

            /**
             * @brief Get the use-def chain from the result object.
             *
             * @return std::vector<irexpr_t&>&
             */
            const std::vector<irstmnt_t> &get_use_def_chain() const
            {
                return use_def_chain;
            }

            /**
             * @brief Get the def-use chain of the operand1 (if exists)
             *
             * @return const std::unordered_map<irexpr_t, std::vector<irstmnt_t>>&
             */
            const std::unordered_map<irexpr_t, std::vector<irstmnt_t>> &get_def_use_chains() const
            {
                return def_use_chains;
            }

            /**
             * @brief Get the def use chain by value object
             *
             * @param value
             * @return std::optional<std::vector<irexpr_t>*>
             */
            std::optional<std::vector<irstmnt_t> *> get_def_use_chain_by_value(irexpr_t value)
            {
                if (def_use_chains.find(value) == def_use_chains.end())
                    return std::nullopt;
                return &def_use_chains.at(value);
            }

            /**
             * @brief Add instruction to use_def_chain
             *
             * @param instr
             */
            void add_instr_to_use_def_chain(irstmnt_t instr)
            {
                use_def_chain.push_back(instr);
            }

            /**
             * @brief Add instruction to op1 def_use_chain
             *
             * @param var register, or temporal register to use in def_use_chains.
             * @param instr instruction with the definition of the value.
             */
            void add_instr_to_def_use_chain(irexpr_t var, irstmnt_t instr)
            {
                def_use_chains[var].push_back(instr);
            }

            /**
             * @brief Invalidate the use_def and def_use chains
             *
             */
            void invalidate_chains();

            /**
             * @brief Invalidate the use_def_chain clearing the vector.
             *
             */
            void invalidate_use_def_chain()
            {
                use_def_chain.clear();
            }

            /**
             * @brief Invalidate the def-use chains
             *
             */
            void invalidate_def_use_chains()
            {
                def_use_chains.clear();
            }

            /**
             * @brief Print the use def and def use chain
             *        for debugging purposes.
             *
             */
            void print_use_def_and_def_use_chain();

        private:
            //! Type of the statement.
            stmnt_type_t stmnt_type;
            //! Op type
            op_type_t op_type;

            //! Vectors used for use-def and def-use chains
            std::vector<irstmnt_t> use_def_chain;
            std::unordered_map<irexpr_t, std::vector<irstmnt_t>> def_use_chains;
        };

        class IRUJmp : public IRStmnt
        {
        public:
            /**
             * @brief Constructor of IRUJmp, this kind of class is an unconditional jump with just one target.
             * @param target: target of the jump.
             * @return void
             */
            IRUJmp(uint64_t addr, irblock_t target);

            /**
             * @brief Destructor of IRUJmp, nothing to be done here.
             * @return void
             */
            ~IRUJmp() = default;

            /**
             * @brief Set the jump target block.
             *
             * @param target: block target where will jump.
             */
            void set_jump_target(irblock_t target)
            {
                this->target = target;
            }

            /**
             * @brief Get the target of the unconditional jump.
             * @return irblock_t
             */
            irblock_t get_jump_target()
            {
                return target;
            }

            /**
             * @brief Return the address or offset from the jump.
             *
             * @return uint64_t
             */
            uint64_t get_jump_addr()
            {
                return addr;
            }

            /**
             * @brief Return a string representation of IRUJmp.
             *
             * @return std::string
             */
            std::string to_string();

        private:
            //! offset or address of target
            uint64_t addr;
            //! target where the jump will fall
            irblock_t target;
        };

        class IRCJmp : public IRStmnt
        {
        public:
            /**
             * @brief Constructor of IRCJmp, this represent a conditional jump.
             * @param addr: address where jump goes if not taken.
             * @param condition: condition (commonly a register from a CMP) that if true takes target.
             * @param target: block target of the conditional jump.
             * @param fallthrough: block of the statement if the condition is not true.
             * @return void
             */
            IRCJmp(uint64_t addr, irstmnt_t condition, irblock_t target, irblock_t fallthrough);

            /**
             * @brief Destructor of IRCJmp, nothing to be done.
             * @return void
             */
            ~IRCJmp() = default;

            /**
             * @brief Get address target from the jump.
             *
             * @return uint64_t
             */
            uint64_t get_addr()
            {
                return addr;
            }

            /**
             * @brief Get the condition of the conditional jump.
             * @return irstmnt_t
             */
            irstmnt_t get_condition()
            {
                return condition;
            }

            /**
             * @brief Set the target block.
             *
             * @param target: target where jump would go if taken.
             */
            void set_jump_target(irblock_t target)
            {
                this->target = target;
            }

            /**
             * @brief Get the jump target if the condition is true.
             * @return irstmnt_t
             */
            irblock_t get_jump_target()
            {
                return target;
            }

            /**
             * @brief Set fallthrough block
             *
             * @param fallthrough: block where jump would go if not taken.
             */
            void set_fallthrough_Target(irblock_t fallthrough)
            {
                this->fallthrough = fallthrough;
            }

            /**
             * @brief Get the fallthrough target in case condition is false.
             * @return irstmnt_t
             */
            irblock_t get_fallthrough_target()
            {
                return fallthrough;
            }

            /**
             * @brief Return string representation of IRCJmp.
             *
             * @return std::string
             */
            std::string to_string();

        private:
            //! offset or address of target
            uint64_t addr;
            //! Condition for taking the target jump
            irstmnt_t condition;
            //! target where the jump will fall
            irblock_t target;
            //! fallthrough target.
            irblock_t fallthrough;
        };

        class IRRet : public IRStmnt
        {
        public:
            /**
             * @brief Constructor of IRRet, this statement represent a return instruction.
             * @param ret_value: return value, this can be a NONE type or a register.
             * @return void
             */
            IRRet(irstmnt_t ret_value);

            /**
             * @brief Destructor of IRRet, nothing to be done.
             * @return void
             */
            ~IRRet() = default;

            /**
             * @brief Get the return value this will be a NONE IRType, or an IRReg.
             * @return irstmnt_t
             */
            irstmnt_t get_return_value()
            {
                return ret_value;
            }

            /**
             * @brief Return the string representation of IRRet.
             *
             * @return std::string
             */
            std::string to_string();

        private:
            //! Returned value, commonly a NONE IRType, or an IRReg.
            irstmnt_t ret_value;
        };

        class IRNop : public IRStmnt
        {
        public:
            /**
             * @brief Construct a new IRNop::IRNop object
             */
            IRNop();

            /**
             * @brief Destroy the IRNop::IRNop object
             */
            ~IRNop() = default;

            std::string to_string()
            {
                return "IRNop[]";
            }
        };

        class IRSwitch : public IRStmnt
        {
        public:
            /**
             * @brief Construct a new IRSwitch::IRSwitch object
             *
             * @param offsets: offsets where switch branch goes.
             * @param condition: possible condition checked on switch.
             * @param constants_checks: values checked on switch (if any)
             */
            IRSwitch(std::vector<int32_t> offsets,
                     irexpr_t condition,
                     std::vector<int32_t> constants_checks);

            /**
             * @brief Destroy the IRSwitch::IRSwitch object
             */
            ~IRSwitch() = default;

            /**
             * @brief Return the offsets where switch can jump to.
             *
             * @return std::vector<int32_t>
             */
            const std::vector<int32_t> &get_offsets() const
            {
                return offsets;
            }

            /**
             * @brief Return the possible condition checked on switch.
             *
             * @return irexpr_t
             */
            irexpr_t get_condition()
            {
                return condition;
            }

            /**
             * @brief Return possible constant checks used in switch.
             *
             * @return std::vector<int32_t>
             */
            const std::vector<int32_t> &get_constants_checks() const
            {
                return constants_checks;
            }

            /**
             * @brief To string method print IR instruction.
             *
             * @return std::string
             */
            std::string to_string();

        private:
            //! switch offsets where instruction will jump.
            std::vector<int32_t> offsets;
            //! condition taken to decide where to jump
            irexpr_t condition;
            //! conditions checked during switch.
            std::vector<int32_t> constants_checks;
        };

        class IRExpr : public IRStmnt
        {
        public:
            enum expr_type_t
            {
                BINOP_EXPR_T,
                UNARYOP_EXPR_T,
                ASSIGN_EXPR_T,
                PHI_EXPR_T,
                CALL_EXPR_T,
                TYPE_EXPR_T,
                LOAD_EXPR_T,
                STORE_EXPR_T,
                ZCOMP_EXPR_T,
                BCOMP_EXPR_T,
                NEW_EXPR_T,
                NONE_EXPR_T = 99 // used to finish the expressions
            };

            /**
             * @brief Constructor of IRExpr this will be the most common type of instruction of the IR.
             *
             * @param type type of the expression (type of the instruction).
             * @param op_type type of global operation
             */
            IRExpr(expr_type_t type, op_type_t op_type);

            /**
             * @brief Destructor of IRExpr, nothing to be done.
             * @return void
             */
            ~IRExpr() = default;

            /**
             * @brief Get the type of current expression.
             * @return expr_type_t
             */
            expr_type_t get_expression_type()
            {
                return type;
            }

            /**
             * @brief Return the correct string representation for the IRExpr.
             *
             * @return std::string
             */
            std::string to_string();

            /**
             * @brief Comparison of IRExpr instructions.
             * @return bool
             */
            bool equals(irexpr_t irexpr);

            /**
             * @brief Operator == for IRExpr.
             * @param ope1: first operation to compare.
             * @param ope2: second operation to compare.
             * @return bool
             */
            friend bool operator==(IRExpr &, IRExpr &);

        private:
            //! ir expression as string
            std::string ir_expr_str;

            //! expression type
            expr_type_t type;
        };

        class IRBinOp : public IRExpr
        {
        public:
            enum bin_op_t
            {
                // common arithmetic instructions
                ADD_OP_T,
                SUB_OP_T,
                S_MUL_OP_T,
                U_MUL_OP_T,
                S_DIV_OP_T,
                U_DIV_OP_T,
                MOD_OP_T,
                // common logic instructions
                AND_OP_T,
                XOR_OP_T,
                OR_OP_T,
                SHL_OP_T,
                SHR_OP_T,
                USHR_OP_T,
            };

            /**
             * @brief Constructor of IRBinOp, this class represent different instructions with two operators and a result.
             * @param bin_op_type: type of binary operation.
             * @param result: where result of operation is stored.
             * @param op1: first operand of the operation.
             * @param op2: second operand of the operation.
             */
            IRBinOp(bin_op_t bin_op_type,
                    irexpr_t result,
                    irexpr_t op1,
                    irexpr_t op2);

            /**
             * @brief Destructor of IRBinOp.
             * @return void
             */
            ~IRBinOp() = default;

            /**
             * @brief Get the type of binary operation applied.
             * @return bin_op_t
             */
            bin_op_t get_bin_op_type()
            {
                return bin_op_type;
            }

            /**
             * @brief Get expression of type where result is stored.
             * @return irexpr_t
             */
            irexpr_t get_result()
            {
                return result;
            }

            /**
             * @brief Get the operand 1 of the operation.
             * @return irexpr_t
             */
            irexpr_t get_op1()
            {
                return op1;
            }

            /**
             * @brief Get the operand 2 of the operation.
             * @return irexpr_t
             */
            irexpr_t get_op2()
            {
                return op2;
            }

            /**
             * @brief Return a string representation of IRBinOp.
             *
             * @return std::string
             */
            std::string to_string();

            /**
             * @brief Comparison of IRBinOp instructions.
             * @return bool
             */
            bool equals(irbinop_t irbinop);

            /**
             * @brief Operator == for IRBinOp.
             * @param ope1: first operation to compare.
             * @param ope2: second operation to compare.
             * @return bool
             */
            friend bool operator==(IRBinOp &, IRBinOp &);

        private:
            //! type of binary operation
            bin_op_t bin_op_type;
            //! IRBinOp =>  IRExpr(result) = IRExpr(op1) <binop> IRExpr(op2)
            //! for the result we will have an IRExpr too.
            irexpr_t result;
            //! now each one of the operators
            irexpr_t op1;
            irexpr_t op2;
        };

        class IRUnaryOp : public IRExpr
        {
        public:
            enum unary_op_t
            {
                NONE_UNARY_OP_T,
                INC_OP_T,
                DEC_OP_T,
                NOT_OP_T,
                NEG_OP_T,
                CAST_OP_T,  // maybe not used in binary
                Z_EXT_OP_T, // zero extend
                S_EXT_OP_T  // sign extend
            };

            enum cast_type_t
            {
                NONE_CAST,
                TO_BYTE,
                TO_CHAR,
                TO_SHORT,
                TO_INT,
                TO_LONG,
                TO_FLOAT,
                TO_DOUBLE,
                TO_ADDR,
                TO_BOOLEAN,
                TO_CLASS,
                TO_VOID,
                TO_ARRAY,
            };

            /**
             * @brief Constructor of IRUnaryOp class, this will get different values for the instruction.
             * @param unary_op_type: type of unary operation.
             * @param result: where the operation stores the result.
             * @param op: operand from the instruction.
             */
            IRUnaryOp(unary_op_t unary_op_type,
                      irexpr_t result,
                      irexpr_t op);

            /**
             * @brief Constructor of IRUnaryOp class, this will get different values for the instruction.
             * @param unary_op_type: type of unary operation.
             * @param cast_type: instruction is cast, specify type of cast.
             * @param result: where the operation stores the result.
             * @param op: operand from the instruction.
             */
            IRUnaryOp(unary_op_t unary_op_type,
                      cast_type_t cast_type,
                      irexpr_t result,
                      irexpr_t op);

            /**
             * @brief  Constructor of IRUnaryOp class, this will get different values for the instruction.
             *
             * @param unary_op_type: type of unary operation.
             * @param cast_type: instruction is cast, specify type of cast.
             * @param class_name: if cast is TO_CLASS, specify the name of the class.
             * @param result: where the operation stores the result.
             * @param op: operand from the instruction.
             * @return void
             */
            IRUnaryOp(unary_op_t unary_op_type,
                      cast_type_t cast_type,
                      std::string class_name,
                      irexpr_t result,
                      irexpr_t op);

            /**
             * @brief Destructor of IRUnaryOp class, nothing to be done.
             * @return void
             */
            ~IRUnaryOp() = default;

            /**
             * @brief Get the type of unary operation of the instruction.
             * @return unary_op_t
             */
            unary_op_t get_unary_op_type()
            {
                return unary_op_type;
            }

            /**
             * @brief Get the IRExpr where result of the operation would be stored.
             * @return irexpr_t
             */
            irexpr_t get_result()
            {
                return result;
            }

            /**
             * @brief Get the IRExpr of the operand of the instruction.
             * @return irexpr_t
             */
            irexpr_t get_op()
            {
                return op;
            }

            /**
             * @brief Set the cast type of the Cast operation.
             * @param cast_type: type of cast to assign.
             * @return void
             */
            void set_cast_type(cast_type_t cast_type)
            {
                this->cast_type = cast_type;
            }

            /**
             * @brief Get the type of cast of the instruction.
             * @return cast_type_t
             */
            cast_type_t get_cast_type()
            {
                return cast_type;
            }

            /**
             * @brief Get the class name where register is casted to.
             *
             * @return std::string
             */
            std::string get_class_cast()
            {
                return class_name;
            }

            /**
             * @brief Return a string representation of IRUnaryOp.
             *
             * @return std::string
             */
            std::string to_string();

            /**
             * @brief Comparison of IRUnaryOp instructions.
             * @return bool
             */
            bool equals(irunaryop_t irunaryop);

            /**
             * @brief Operator == for IRUnaryOp.
             * @param ope1: first operation to compare.
             * @param ope2: second operation to compare.
             * @return bool
             */
            friend bool operator==(IRUnaryOp &, IRUnaryOp &);

        private:
            //! type of unary operation =D
            unary_op_t unary_op_type;
            //! used for casting operations
            cast_type_t cast_type;
            //! Class casted to
            std::string class_name;
            //! IRUnaryOp => IRExpr(result) = <unaryop> IRExpr(op)
            //! an IRExpr for where the result is stored.
            irexpr_t result;
            // operator
            irexpr_t op;
        };

        class IRAssign : public IRExpr
        {
        public:
            /**
             * @brief Constructor of IRAssign, this one will have just those from the basic types,
             *        no more information is needed, the left part and the right part.
             * @param destination: type where the value will be assigned.
             * @param source: type from where the value is taken.
             */
            IRAssign(irexpr_t destination,
                     irexpr_t source);

            /**
             * @brief Destructor of IRAssign, nothing to be done here.
             * @return void
             */
            ~IRAssign() = default;

            /**
             * @brief Get destination type where value is assigned.
             * @return irexpr_t
             */
            irexpr_t get_destination()
            {
                return destination;
            }

            /**
             * @brief Get the source operand from the assignment.
             * @return irexpr_t
             */
            irexpr_t get_source()
            {
                return source;
            }

            /**
             * @brief Return the string representation of IRAssign.
             *
             * @return std::string
             */
            std::string to_string();

            /**
             * @brief Comparison of IRAssign instructions.
             * @return bool
             */
            bool equals(irassign_t irassign);

            /**
             * @brief Operator == for IRAssign.
             * @param ope1: first operation to compare.
             * @param ope2: second operation to compare.
             * @return bool
             */
            friend bool operator==(IRAssign &, IRAssign &);

        private:
            //! destination where the value will be stored.
            irexpr_t destination;
            //! source expression from where the value is taken
            irexpr_t source;
        };

        class IRPhi : public IRExpr
        {
        public:
            /**
             * @brief Construct a new IRPhi object, this will be used in
             *        the IRGraphSSA, these kind of instructions represent
             *        how a variable could contain different definitions of
             *        a variable.
             */
            IRPhi();

            /**
             * @brief Destroy the IRPhi object
             */
            ~IRPhi() = default;

            /**
             * @brief Get the vector with all the parameters.
             *
             * @return std::unordered_map<uint32_t, irexpr_t>
             */
            std::unordered_map<uint32_t, irexpr_t> &get_params()
            {
                return params;
            }

            /**
             * @brief Add a parameter into the Phi node.
             *
             * @param param parameter to include
             * @param id identifier where to insert the parameter
             */
            void add_param(irexpr_t param, uint32_t id);

            /**
             * @brief Get the result register for the phi node.
             *
             * @return irexpr_t
             */
            irexpr_t get_result()
            {
                return result;
            }

            /**
             * @brief Add the register used as result of the phi node.
             *
             * @param result
             */
            void add_result(irexpr_t result)
            {
                this->result = result;
            }

            /**
             * @brief Return the string representation of IRPhi.
             *
             * @return std::string
             */
            std::string to_string();

            /**
             * @brief Comparison of IRPhi instructions.
             * @return bool
             */
            bool equals(irphi_t irphi);

            /**
             * @brief Operator == for IRPhi.
             * @param ope1: first operation to compare.
             * @param ope2: second operation to compare.
             * @return bool
             */
            friend bool operator==(IRPhi &, IRPhi &);

        private:
            irexpr_t result;
            std::unordered_map<uint32_t, irexpr_t> params;
        };

        class IRCall : public IRExpr
        {
        public:
            enum call_type_t
            {
                INTERNAL_CALL_T,  // call to internal component
                EXTERNAL_CALL_T,  // call to external library (example DLL, .so file, external component, etc)
                SYSCALL_T,        // a syscall type
                NONE_CALL_T = 99, // Not specified
            };

            /**
             * @brief Constructor of IRCall, this expression represents a Call to a function or method.
             * @param callee: function/method called represented as IRExpr (super-super-class of IRCallee).
             * @param args: vector of arguments to the function/method.
             */
            IRCall(irexpr_t callee,
                   std::vector<irexpr_t> args);

            /**
             * @brief Constructor of IRCall, this expression represents a Call to a function or method.
             * @param callee: function/method called represented as IRExpr (super-super-class of IRCallee).
             * @param call_type: type of call instruction.
             * @param args: vector of arguments to the function/method.
             */
            IRCall(irexpr_t callee,
                   call_type_t call_type,
                   std::vector<irexpr_t> args);

            /**
             * @brief Destructor of IRCall, nothing to be done.
             * @return void
             */
            ~IRCall() = default;

            /**
             * @brief Get the callee object from the function/method called.
             * @return irexpr_t
             */
            irexpr_t get_callee()
            {
                return callee;
            }

            /**
             * @brief Get the arguments from the call.
             * @return std::vector<irexpr_t>
             */
            const std::vector<irexpr_t> &get_args() const
            {
                return args;
            }

            /**
             * @brief Get a string representation of IRCall.
             *
             * @return std::string
             */
            std::string to_string();

            /**
             * @brief Set the ret_val object
             *
             * @param ret_val
             */
            void set_ret_val(irexpr_t ret_val)
            {
                this->ret_val = ret_val;
            }

            /**
             * @brief Get the ret_val object
             *
             * @return irexpr_t
             */
            irexpr_t get_ret_val()
            {
                return ret_val;
            }

            /**
             * @brief Get the call type value
             *
             * @return call_type_t
             */
            call_type_t get_call_type()
            {
                return call_type;
            }

            /**
             * @brief Comparison of IRCall instructions.
             * @return bool
             */
            bool equals(ircall_t ircall);

            /**
             * @brief Operator == for IRCall.
             * @param ope1: first operation to compare.
             * @param ope2: second operation to compare.
             * @return bool
             */
            friend bool operator==(IRCall &, IRCall &);

        private:
            //! Type of call
            call_type_t call_type;
            //! Type representing the function/method called
            irexpr_t callee;
            //! Vector with possible arguments
            std::vector<irexpr_t> args;
            //! Return value (if it's for example a register)
            irexpr_t ret_val;
        };

        class IRLoad : public IRExpr
        {
        public:
            /**
             * @brief Constructor of IRLoad class, this class represent a load from memory (using memory or using register).
             * @param destination: register where the value will be stored.
             * @param source: expression from where the memory will be retrieved.
             * @param size: loaded size.
             * @return void
             */
            IRLoad(irexpr_t destination,
                   irexpr_t source,
                   std::uint32_t size);

            /**
             * @brief Constructor of IRLoad class, this class represent a load from memory (using memory or using register).
             * @param destination: register where the value will be stored.
             * @param source: expression from where the memory will be retrieved.
             * @param index: index from the load if this is referenced with an index.
             * @param size: loaded size.
             * @return void
             */
            IRLoad(irexpr_t destination,
                   irexpr_t source,
                   irexpr_t index,
                   std::uint32_t size);

            /**
             * @brief Destructor of IRLoad, nothing to be done.
             * @return void
             */
            ~IRLoad() = default;

            /**
             * @brief Get destination register from the load.
             * @return irexpr_t
             */
            irexpr_t get_destination()
            {
                return destination;
            }

            /**
             * @brief Get the expression from where value is loaded.
             * @return irexpr_t
             */
            irexpr_t get_source()
            {
                return source;
            }

            /**
             * @brief Get the index of the instruction used.
             *
             * @return irexpr_t
             */
            irexpr_t get_index()
            {
                return index;
            }

            /**
             * @brief Get the loaded size.
             * @return std::uint32_t
             */
            std::uint32_t get_size()
            {
                return size;
            }

            /**
             * @brief Return string representation of IRLoad.
             *
             * @return std::string
             */
            std::string to_string();

            /**
             * @brief Comparison of IRLoad instruction.
             * @return bool
             */
            bool equals(irload_t irload);

            /**
             * @brief Operator == for IRLoad.
             * @param ope1: first operation to compare.
             * @param ope2: second operation to compare.
             * @return bool
             */
            friend bool operator==(IRLoad &, IRLoad &);

        private:
            //! Register where the memory pointed by a register will be loaded.
            irexpr_t destination;
            //! Expression from where memory is read.
            irexpr_t source;
            //! Index if this is referenced by for example a register.
            irexpr_t index;
            //! Size of loaded value
            std::uint32_t size;
        };

        class IRStore : public IRExpr
        {
        public:
            /**
             * @brief Constructor of IRStore class, this represent an store to memory instruction.
             * @param destination: Expression where value is written to.
             * @param source: register with the value to be stored.
             * @param size: size of the stored value.
             * @return void
             */
            IRStore(irexpr_t destination,
                    irexpr_t source,
                    std::uint32_t size);

            /**
             * @brief Constructor of IRStore class, this represent an store to memory instruction.
             * @param destination: Expression where value is written to.
             * @param source: register with the value to be stored.
             * @param index: index where value is stored.
             * @param size: size of the stored value.
             * @return void
             */
            IRStore(irexpr_t destination,
                    irexpr_t source,
                    irexpr_t index,
                    std::uint32_t size);

            /**
             * @brief Destructor of IRStore class, nothing to be done.
             * @return void
             */
            ~IRStore() = default;

            /**
             * @brief Get expression destination where value is going to be written.
             * @return irexpr_t
             */
            irexpr_t get_destination()
            {
                return destination;
            }

            /**
             * @brief Get source where value is taken from.
             * @return irexpr_t
             */
            irexpr_t get_source()
            {
                return source;
            }

            /**
             * @brief Get the index from the store instruction.
             *
             * @return irexpr_t
             */
            irexpr_t get_index()
            {
                return index;
            }

            /**
             * @brief Get size of value to be written.
             * @return std::uint32_t
             */
            std::uint32_t get_size()
            {
                return size;
            }

            /**
             * @brief Return a string representation of IRStore.
             *
             * @return std::string
             */
            std::string to_string();

            /**
             * @brief Comparison of IRStore instructions.
             * @return bool
             */
            bool equals(irstore_t irstore);

            /**
             * @brief Operator == for IRStore.
             * @param ope1: first operation to compare.
             * @param ope2: second operation to compare.
             * @return bool
             */
            friend bool operator==(IRStore &, IRStore &);

        private:
            //! Memory pointed by register where value will be stored.
            irexpr_t destination;
            //! Expression with source of value to be stored.
            irexpr_t source;
            //! Index if this is referenced by for example a register.
            irexpr_t index;
            //! Size of stored value
            std::uint32_t size;
        };

        class IRZComp : public IRExpr
        {
        public:
            enum zero_comp_t
            {
                EQUAL_ZERO_T,       // ==
                NOT_EQUAL_ZERO_T,   // !=
                LOWER_ZERO_T,       // <
                GREATER_EQUAL_ZERO, // >=
                GREATER_ZERO_T,     // >
                LOWER_EQUAL_ZERO    // <=
            };

            /**
             * @brief Constructor of IRZComp, this is a comparison with zero.
             * @param comp: type of comparison (== or !=).
             * @param result: register or temporal register where result is stored.
             * @param reg: register used in the comparison.
             * @return void
             */
            IRZComp(zero_comp_t comp,
                    irexpr_t result,
                    irexpr_t reg);

            /**
             * @brief Destructor of IRZComp, nothing to be done.
             * @return void
             */
            ~IRZComp() = default;

            /**
             * @brief Return the result register or temporal register.
             *
             * @return irexpr_t
             */
            irexpr_t get_result()
            {
                return result;
            }

            /**
             * @brief Get register used in the comparison.
             * @return irexpr_t
             */
            irexpr_t get_reg()
            {
                return reg;
            }

            /**
             * @brief Get the type of comparison with zero.
             * @return zero_comp_t
             */
            zero_comp_t get_comparison()
            {
                return comp;
            }

            /**
             * @brief Return the string expression of IRZComp.
             *
             * @return std::string
             */
            std::string to_string();

            /**
             * @brief Comparison of IRZComp instruction with shared_ptr.
             * @return bool
             */
            bool equals(irzcomp_t irzcomp);

            /**
             * @brief Operator == for IRZComp.
             * @param ope1: first operation to compare.
             * @param ope2: second operation to compare.
             * @return bool
             */
            friend bool operator==(IRZComp &, IRZComp &);

        private:
            //! Register where result is stored
            irexpr_t result;
            //! Register for comparison with zero.
            irexpr_t reg;
            //! Type of comparison
            zero_comp_t comp;
        };

        class IRBComp : public IRExpr
        {
        public:
            enum comp_t
            {
                EQUAL_T,         // ==
                NOT_EQUAL_T,     // !=
                GREATER_T,       // >
                GREATER_EQUAL_T, // >=
                LOWER_T,         // <
                LOWER_EQUAL_T,   // <=
                ABOVE_T,         // (unsigned) >
                ABOVE_EQUAL_T,   // (unsigned) >=
                BELOW_T,         // (unsigned) <
            };

            /**
             * @brief Constructor of IRBComp, this class represent a comparison between two types.
             * @param comp: type of comparison from an enum.
             * @param result: register or temporal register where result is stored.
             * @param reg1: first type where the comparison is applied.
             * @param reg2: second type where the comparison is applied.
             * @return void
             */
            IRBComp(comp_t comp,
                    irexpr_t result,
                    irexpr_t reg1,
                    irexpr_t reg2);

            /**
             * @brief Destructor of IRBComp, nothing to be done.
             * @return void
             */
            ~IRBComp() = default;

            /**
             * @brief Get the result register or temporal register.
             *
             * @return irexpr_t
             */
            irexpr_t get_result()
            {
                return result;
            }

            /**
             * @brief Return the first part of the comparison.
             * @return irexpr_t
             */
            irexpr_t get_reg1()
            {
                return reg1;
            }

            /**
             * @brief Return the second part of the comparison.
             * @return irexpr_t
             */
            irexpr_t get_reg2()
            {
                return reg2;
            }

            /**
             * @brief Return the type of comparison
             * @return comp_t
             */
            comp_t get_comparison()
            {
                return comp;
            }

            /**
             * @brief Return the string representation of IRBComp.
             *
             * @return std::string
             */
            std::string to_string();

            /**
             * @brief Comparison of IRBComp instruction with shared_ptr.
             * @return bool
             */
            bool equals(irbcomp_t bcomp);

            /**
             * @brief Operator == for IRBComp.
             * @param ope1: first operation to compare.
             * @param ope2: second operation to compare.
             * @return bool
             */
            friend bool operator==(IRBComp &, IRBComp &);

        private:
            //! register or temporal register where result is stored
            irexpr_t result;
            //! registers used in the comparisons.
            irexpr_t reg1;
            irexpr_t reg2;
            //! Type of comparison
            comp_t comp;
        };

        class IRNew : public IRExpr
        {
        public:
            /**
             * @brief Construct a new IRNew::IRNew object which represents
             *        the creation of an instance of a class.
             *
             * @param result: result register where object is stored.
             * @param class_instance: IRClass object which represent the instance.
             * @return void
             */
            IRNew(irexpr_t result,
                  irexpr_t class_instance);

            /**
             * @brief Destroy the IRNew::IRNew object
             *
             */
            ~IRNew() = default;

            /**
             * @brief Get the destination register
             *
             * @return irexpr_t
             */
            irexpr_t get_result()
            {
                return result;
            }

            /**
             * @brief Get the source class.
             *
             * @return irexpr_t
             */
            irexpr_t get_source_class()
            {
                return class_instance;
            }

            /**
             * @brief Return string representation of IRNew
             *
             * @return std::string
             */
            std::string to_string();

            /**
             * @brief Compare two IRNew instructions with shared_ptr
             *
             * @param new_i
             * @return true
             * @return false
             */
            bool equals(irnew_t bcomp);

            /**
             * @brief Operator == for IRNew instruction.
             *
             * @param new1
             * @param new2
             * @return true
             * @return false
             */
            friend bool operator==(IRNew &, IRNew &);

        private:
            //! register where the result will be stored.
            irexpr_t result;
            //! class type which will create a new instance.
            irexpr_t class_instance;
        };

        class IRType : public IRExpr
        {
        public:
            enum type_t
            {
                REGISTER_TYPE = 0,
                TEMP_REGISTER_TYPE,
                CONST_INT_TYPE,
                CONST_FLOAT_TYPE,
                FIELD_TYPE,
                MEM_TYPE,
                STRING_TYPE,
                CLASS_TYPE,
                CALLEE_TYPE,
                NONE_TYPE = 99
            };

            enum mem_access_t
            {
                LE_ACCESS = 0, //! little-endian access
                BE_ACCESS,     //! big-endian access
                ME_ACCESS,     //! This shouldn't commonly happen?
                NONE_ACCESS = 99
            };

            /**
             * @brief Constructor of the IRType, this will be the generic type used for the others.
             * @param type: type of the class.
             * @param op_type: global type of operation
             * @param type_name: name used for representing the type while printing.
             * @param type_size: size of the type in bytes.
             * @return void
             */
            IRType(type_t type, op_type_t op_type, std::string type_name, size_t type_size);

            /**
             * @brief Destructor of the IRType.
             * @return void
             */
            ~IRType() = default;

            /**
             * @brief retrieve the type name.
             * @return std::string
             */
            std::string get_type_name()
            {
                return type_name;
            }

            /**
             * @brief retrieve the type size.
             * @return size_t
             */
            size_t get_type_size()
            {
                return type_size;
            }

            /**
             * @brief method from IRType this return one of the types given by class.
             * @return type_t
             */
            type_t get_type()
            {
                return type;
            }

            virtual std::string get_type_str()
            {
                return "";
            }

            /**
             * @brief virtual method from IRType this must be implemented by other types too. Returns one of the mem_access_t enum values.
             * @return mem_access_t
             */
            mem_access_t get_access()
            {
                return NONE_ACCESS;
            }

            /**
             * @brief Write an annotations to the type object.
             * @return void
             */
            void write_annotations(std::string annotations)
            {
                this->annotations = annotations;
            }

            /**
             * @brief Read the annotations from the type object.
             * @return std::string
             */
            std::string read_annotations()
            {
                return annotations;
            }

            /**
             * @brief Return the proper to_string.
             *
             * @return std::string
             */
            std::string to_string();

            /**
             * @brief Comparison of two IRType with shared_ptr.
             * @return bool
             */
            bool equal(irtype_t type);

            /**
             * @brief == operator for IRType, we have specific check to do.
             * @param type1: first type for comparison.
             * @param type2: second type for comparison
             * @return bool
             */
            friend bool operator==(IRType &, IRType &);

        private:
            //! type value as a type_t
            type_t type;

            //! name used to represent the type in IR representation.
            std::string type_name;

            //! size of the type, this can vary depending on architecture
            //! and so on.
            size_t type_size;

            //! annotations are there for you to write whatever you want
            std::string annotations;
        };

        class IRReg : public IRType
        {
        public:
            /**
             * @brief Constructor of IRReg type.
             * @param reg_id: id of the register this can be an enum if is a well known register, or just an id.
             * @param current_arch: curreng architecture to create the register.
             * @param type_name: string for representing the register.
             * @param type_size: size of the register.
             * @return void
             */
            IRReg(std::uint32_t reg_id, int current_arch, std::string type_name, size_t type_size);

            /**
             * @brief Constructor of IRReg type.
             * @param reg_id: id of the register this can be an enum if is a well known register, or just an id.
             * @param reg_sub_id: sub id of the register used in the SSA form.
             * @param current_arch: curreng architecture to create the register.
             * @param type_name: string for representing the register.
             * @param type_size: size of the register.
             * @return void
             */
            IRReg(std::uint32_t reg_id, std::int32_t reg_sub_id, int current_arch, std::string type_name, size_t type_size);

            /**
             * @brief Destructor of IRReg.
             * @return void
             */
            ~IRReg() = default;

            /**
             * @brief Return the ID from the register.
             * @return std::uint32_t
             */
            std::uint32_t get_id()
            {
                return id;
            }

            /**
             * @brief Get the sub ID from the register.
             * @return std::uint32_t
             */
            std::uint32_t get_sub_id()
            {
                return sub_id;
            }

            /**
             * @brief Get the current arch value
             *
             * @return int
             */
            int get_current_arch()
            {
                return current_arch;
            }

            /**
             * @brief Return the register type as str.
             * @return std::string
             */
            std::string get_type_str()
            {
                return "Register";
            }

            /**
             * @brief Return the access type to the register. We get as all the register are accessed as Little-Endian.
             * @return mem_access_t
             */
            mem_access_t get_access()
            {
                return NONE_ACCESS;
            }

            /**
             * @brief Return string representation of IRReg.
             * @return std::string
             */
            std::string to_string();

            /**
             * @brief Compare two IRRegs given a smart pointer
             *        this time do not include the sub_id, just
             *        a simple function to check this.
             * 
             * @param reg 
             * @return true 
             * @return false 
             */
            bool same(irreg_t reg);

            /**
             * @brief Compare two IRRegs given by smart pointer.
             * @return bool
             */
            bool equal(irreg_t reg);

            /**
             * @brief == operator for IRRegs, we have specific check to do.
             * @param type1: first type for comparison.
             * @param type2: second type for comparison
             * @return bool
             */
            friend bool operator==(IRReg &, IRReg &);

        private:
            //! id of the register, this will be an enum
            //! in case the arquitecture contains a known set
            //! of registers, for example x86-64 will have a
            //! well known set of registers, e.g. EAX, AX, RSP
            //! RIP, etc.
            //! Other arquitectures like DEX VM will not have
            //! an specific set.
            std::uint32_t id;
            //! sub id of the register, this sub id will be used
            //! in the SSA form, and used to check if a register
            //! is the same than other.
            std::int32_t sub_id;

            int current_arch;
        };

        class IRTempReg : public IRType
        {
        public:
            /**
             * @brief Constructor of IRTempReg type.
             * @param reg_id: id of the register this will be an incremental id.
             * @param type_name: string for representing the register.
             * @param type_size: size of the register.
             * @return void
             */
            IRTempReg(std::uint32_t reg_id, std::string type_name, size_t type_size);

            /**
             * @brief Destructor of IRTempReg.
             * @return void
             */
            ~IRTempReg() = default;

            /**
             * @brief Return the ID from the register.
             * @return std::uint32_t
             */
            std::uint32_t get_id()
            {
                return id;
            }

            /**
             * @brief Return the temporal register type as str.
             * @return std::string
             */
            std::string get_type_str()
            {
                return "Temporal Register";
            }

            /**
             * @brief Return the access type to the register. We get as all the register are accessed as Little-Endian.
             * @return mem_access_t
             */
            mem_access_t get_access()
            {
                return NONE_ACCESS;
            }

            /**
             * @brief String representation of IRTempReg.
             * @return std::string
             */
            std::string to_string();
            

            /**
             * @brief Compare two IRTempReg temporal registers.
             * @param temp_reg: IRTempReg to compare with.
             * @return bool
             */
            bool equal(irtempreg_t temp_reg);

            /**
             * @brief == operator for IRTempReg, we have specific check to do.
             * @param type1: first type for comparison.
             * @param type2: second type for comparison
             * @return bool
             */
            friend bool operator==(IRTempReg &, IRTempReg &);

        private:
            //! This id will be just an incremental number
            //! as these are temporal registers.
            std::uint32_t id;
        };

        class IRConstInt : public IRType
        {
        public:
            /**
             * @brief Constructor of IRConstInt this represent any integer used in the code.
             * @param value: value of the constant integer
             * @param is_signed: is signed value (true) or unsigned (false).
             * @param byte_order: byte order of the value.
             * @param type_name: name used for representing the value.
             * @param type_size: size of the integer.
             * @return void
             */
            IRConstInt(std::uint64_t value, bool is_signed, mem_access_t byte_order, std::string type_name, size_t type_size);

            /**
             * @brief Destructor of IRConstInt.
             * @return void
             */
            ~IRConstInt() = default;

            /**
             * @brief Get if the int value is signed  or not.
             * @return bool
             */
            bool get_is_signed()
            {
                return is_signed;
            }

            /**
             * @brief Return the CONST_INT type as string.
             * @return std::string
             */
            std::string get_type_str()
            {
                return "ConstInt";
            }

            /**
             * @brief Return the type of access of the constant int.
             * @return mem_access_t
             */
            mem_access_t get_access()
            {
                return byte_order;
            }

            /**
             * @brief Get the value unsigned object
             *
             * @return uint64_t
             */
            uint64_t get_value_unsigned()
            {
                return value;
            }

            /**
             * @brief Get the value signed object
             *
             * @return int64_t
             */
            int64_t get_value_signed()
            {
                return static_cast<int64_t>(value);
            }

            /**
             * @brief Return a string representation of the IRConstInt instruction.
             * @return bool
             */
            std::string to_string();

            /**
             * @brief Comparison of two IRConstInt instructions with shared pointers.
             * @param const_int: IRConstInt to compare with.
             * @return bool
             */
            bool equal(irconstint_t const_int);

            /**
             * @brief == operator for IRConstInt, we have specific check to do.
             * @param type1: first type for comparison.
             * @param type2: second type for comparison
             * @return bool
             */
            friend bool operator==(IRConstInt &, IRConstInt &);

            /**
             * @brief ADD operator for IRConstInt, some checks are applied.
             *
             * @param a
             * @param b
             * @return IRConstInt
             */
            friend IRConstInt operator+(IRConstInt &a, IRConstInt &b);

            /**
             * @brief SUB operator for IRConstInt, some checks are applied.
             *
             * @param a
             * @param b
             * @return IRConstInt
             */
            friend IRConstInt operator-(IRConstInt &a, IRConstInt &b);

            /**
             * @brief DIV operator for IRConstInt, some checks are applied.
             *
             * @param a
             * @param b
             * @return IRConstInt
             */
            friend IRConstInt operator/(IRConstInt &a, IRConstInt &b);

            /**
             * @brief MUL operator for IRConstInt, some checks are applied.
             *
             * @param a
             * @param b
             * @return IRConstInt
             */
            friend IRConstInt operator*(IRConstInt &a, IRConstInt &b);

            /**
             * @brief MOD operator for IRConstInt, some checks are applied.
             *
             * @param a
             * @param b
             * @return IRConstInt
             */
            friend IRConstInt operator%(IRConstInt &a, IRConstInt &b);

            /**
             * @brief AND operator for IRConstInt, some checks are applied.
             *
             * @param a
             * @param b
             * @return IRConstInt
             */
            friend IRConstInt operator&(IRConstInt &a, IRConstInt &b);

            /**
             * @brief OR operator for IRConstInt, some checks are applied.
             *
             * @param a
             * @param b
             * @return IRConstInt
             */
            friend IRConstInt operator|(IRConstInt &a, IRConstInt &b);

            /**
             * @brief XOR operator for IRConstInt, some checks are applied.
             *
             * @param a
             * @param b
             * @return IRConstInt
             */
            friend IRConstInt operator^(IRConstInt &a, IRConstInt &b);

            /**
             * @brief SHL operator for IRConstInt, some checks are applied.
             *
             * @param a
             * @param b
             * @return IRConstInt
             */
            friend IRConstInt operator<<(IRConstInt &a, IRConstInt &b);

            /**
             * @brief SHR operator for IRConstInt, some checks are applied.
             *
             * @param a
             * @param b
             * @return IRConstInt
             */
            friend IRConstInt operator>>(IRConstInt &a, IRConstInt &b);

            /**
             * @brief INC operator for IRConstInt, some checks are applied.
             *
             * @param a
             * @return IRConstInt
             */
            friend IRConstInt operator++(IRConstInt &a, int);

            /**
             * @brief DEC operator for IRConstInt, some checks are applied.
             *
             * @param a
             * @return IRConstInt
             */
            friend IRConstInt operator--(IRConstInt &a, int);

            /**
             * @brief NOT operator for IRConstInt, some checks are applied.
             *
             * @param a
             * @return IRConstInt
             */
            friend IRConstInt operator!(IRConstInt &a);

            /**
             * @brief NEG operator for IRConstInt, some checks are applied.
             *
             * @param a
             * @return IRConstInt
             */
            friend IRConstInt operator~(IRConstInt &a);

        private:
            //! Value of the integer
            std::uint64_t value;
            //! Check to know if the constant is a unsigned
            //! or signed value.
            bool is_signed;
            //! byte order of the value.
            mem_access_t byte_order;
        };

        class IRMemory : public IRType
        {
        public:
            /**
             * @brief IRMemory constructor this represent a memory address with accessed offset and size.
             * @param mem_address: address of the memory.
             * @param offset: offset accessed (commonly 0).
             * @param byte_order: byte order of the memory (LE, BE, ME?).
             * @param type_name: memory representation with a string.
             * @param type_size: size of the memory.
             * @return void
             */
            IRMemory(std::uint64_t mem_address, std::int32_t offset, mem_access_t byte_order, std::string type_name, size_t type_size);

            /**
             * @brief IRMemory destructor.
             * @return void
             */
            ~IRMemory() = default;

            /**
             * @brief Get the memory address of the type.
             * @return std::uint64_t
             */
            std::uint64_t get_mem_address()
            {
                return mem_address;
            }

            /**
             * @brief Get the accessed offset of the type.
             * @return std::int32_t
             */
            std::int32_t get_offset()
            {
                return offset;
            }

            /**
             * @brief Get the MEM_TYPE as string.
             * @return std::string
             */
            std::string get_type_str()
            {
                return "Memory";
            }

            /**
             * @brief Get the mem access type from the memory.
             * @return mem_access_t
             */
            mem_access_t get_access()
            {
                return byte_order;
            }

            /**
             * @brief Return a string representation of IRMemory instruction.
             * @return std::string
             */
            std::string to_string();

            /**
             * @brief Comparison of two IRMemory by shared_ptr.
             * @param memory: IRMemory instruction to compare.
             * @return bool.
             */
            bool equal(irmemory_t memory);

            /**
             * @brief == operator for IRMemory, we have specific check to do.
             * @param type1: first type for comparison.
             * @param type2: second type for comparison
             * @return bool
             */
            friend bool operator==(IRMemory &, IRMemory &);

        private:
            //! accessed address
            std::uint64_t mem_address;
            //! offset of the memory accessed
            std::int32_t offset;
            //! byte order of the memory.
            mem_access_t byte_order;
        };

        class IRString : public IRType
        {
        public:
            /**
             * @brief Constructor of IRString class, this represent strings used in code.
             * @param str_value: value of that string.
             * @param type_name: some meaninful string name.
             * @param type_size: size of the type (probably here string length)
             * @return void
             */
            IRString(std::string str_value, std::string type_name, size_t type_size);

            /**
             * @brief Destructor of IRString, nothing to be done.
             * @return void
             */
            ~IRString() = default;

            /**
             * @brief Return the value of the string.
             * @return std::string
             */
            std::string get_str_value()
            {
                return str_value;
            }

            /**
             * @brief Get the type as a string.
             * @return std::string
             */
            std::string get_type_str()
            {
                return "String";
            }

            /**
             * @brief Get the type of access of string, in this one NONE.
             * @return mem_access_t
             */
            mem_access_t get_access()
            {
                return NONE_ACCESS;
            }

            /**
             * @brief Get a string representation from IRString instruction.
             * @return std::string
             */
            std::string to_string();

            /**
             * @brief Compare two IRString values.
             * @return bool
             */
            bool equal(irstring_t str);

            /**
             * @brief == operator for IRString, we have specific check to do.
             * @param type1: first type for comparison.
             * @param type2: second type for comparison
             * @return bool
             */
            friend bool operator==(IRString &, IRString &);

        private:
            //! string value, probably nothing more will be here
            std::string str_value;
        };

        class IRClass : public IRType
        {
        public:
            /**
             * @brief Constructor of IRClass, this represent the name of a class
             *        that is assigned as a type.
             * @param class_name: name of the class.
             * @param type_name: should be the same value than previous one.
             * @param type_size: should be 0.
             * @return void
             */
            IRClass(std::string class_name, std::string type_name, size_t type_size);

            /**
             * @brief Destructor of IRClass, nothing to be done here.
             * @return void
             */
            ~IRClass() = default;

            /**
             * @brief get the name of the class.
             * @return std::string
             */
            std::string get_class()
            {
                return class_name;
            }

            /**
             * @brief Get the name of the type as a string
             * @return std::string
             */
            std::string get_type_str()
            {
                return "Class";
            }

            /**
             * @brief Get type of access in this case NONE.
             * @return mem_access_t
             */
            mem_access_t get_access()
            {
                return NONE_ACCESS;
            }

            /**
             * @brief Return a string representing the IRClass.
             * @return std::string
             */
            std::string to_string();

            /**
             * @brief Compare two IRClass with smart pointers
             * @return bool
             */
            bool equal(irclass_t class_);

            /**
             * @brief Operator == of IRClass we will compare name of classes.
             * @param type1: first class to compare.
             * @param type2: second class to compare.
             * @return bool
             */
            friend bool operator==(IRClass &, IRClass &);

        private:
            //! class name including path, used for instructions
            //! of type const-class
            std::string class_name;
        };

        class IRCallee : public IRType
        {
        public:
            /**
             * @brief Constructor of IRCallee this represent any function/method called by a caller!
             * @param addr: address of the function/method called (if available).
             * @param name: name of the function/method called (if available).
             * @param class_name: name of the class from the method called (if available).
             * @param n_of_params: number of the parameters for the function/method (if available).
             * @param description: description of the parameters from the function/method (if available).
             * @param type_name: some meaninful string name.
             * @param type_size: size of the type (probably here 0)
             * @return void
             */
            IRCallee(std::uint64_t addr,
                     std::string name,
                     std::string class_name,
                     int n_of_params,
                     std::string description,
                     std::string type_name,
                     size_t type_size);

            /**
             * @brief Destructor of IRCallee, nothing to be done.
             * @return void
             */
            ~IRCallee() = default;

            /**
             * @brief Get address of function/method
             * @return std::uint64_t
             */
            std::uint64_t get_addr()
            {
                return addr;
            }
            /**
             * @brief Get name of the called function/method.
             * @return std::string
             */
            std::string get_name()
            {
                return name;
            }

            /**
             * @brief Get the class name of the called method.
             * @return std::string
             */
            std::string get_class_name()
            {
                return class_name;
            }

            /**
             * @brief Get the number of parameters from the called function/method.
             * @return int
             */
            int get_number_of_params()
            {
                return n_of_params;
            }

            /**
             * @brief Get the description of the method if exists.
             * @return std::string
             */
            std::string get_description()
            {
                return description;
            }

            /**
             * @brief Get the type of IRCallee as string
             * @return std::string
             */
            std::string get_type_str()
            {
                return "Callee";
            }

            /**
             * @brief Get memory access NONE in this case.
             * @return mem_access_t
             */
            mem_access_t get_access()
            {
                return NONE_ACCESS;
            }

            /**
             * @brief Return a string representation of the IRCallee type.
             * @return std::string
             */
            std::string to_string();

            /**
             * @brief Check if two shared objects of Callee are the same.
             * @param callee: IRType instruction to compare.
             * @return bool
             */
            bool equal(ircallee_t callee);

            /**
             * @brief == operator for IRCallee, we have specific check to do.
             * @param type1: first type for comparison.
             * @param type2: second type for comparison
             * @return bool
             */
            friend bool operator==(IRCallee &, IRCallee &);

        private:
            //! for those functions of binary formats we will mostly have the address
            //! only, these can be from a library, from the same binary, etc.
            std::uint64_t addr;
            //! name of the callee function or method, this can be resolved from the
            //! binary symbols if those exist or is given in case of other formats.
            std::string name;
            //! in case it is a method, probably we will need to know class name
            //! for possible analysis which requires to know about a calls.
            std::string class_name;
            //! there are cases where functions/methods can have the same name but
            //! different parameters, you can give the number of parameters (if recognized)
            //! or the string with the description of the method
            int n_of_params;
            std::string description;
        };

        class IRField : public IRType
        {
        public:
            enum field_t
            {
                CLASS_F,
                BOOLEAN_F,
                BYTE_F,
                CHAR_F,
                DOUBLE_F,
                FLOAT_F,
                INT_F,
                LONG_F,
                SHORT_F,
                VOID_F,
                ARRAY_F
            };

            /**
             * @brief Construct a new IRField::IRField object
             *
             * @param class_name: class name of the field
             * @param type: type from field_t
             * @param field_name: name of the field.
             * @param type_name: some meaninful string name.
             * @param type_size: size of the type (probably here 0)
             */
            IRField(std::string class_name,
                    field_t type,
                    std::string field_name,
                    std::string type_name,
                    size_t type_size);

            IRField(std::string class_name,
                    std::string type_class_name,
                    std::string field_name,
                    std::string type_name,
                    size_t type_size);

            /**
             * @brief Destroy the IRField::IRField object
             */
            ~IRField() = default;

            /**
             * @brief Getter for class name
             *
             * @return std::string
             */
            std::string get_class_name()
            {
                return class_name;
            }

            /**
             * @brief Getter for type name.
             *
             * @return field_t
             */
            field_t get_type()
            {
                return type;
            }

            /**
             * @brief Return if type == CLASS the class name.
             *
             * @return std::string
             */
            std::string get_type_class()
            {
                return type_class;
            }
            /**
             * @brief Getter for field name.
             *
             * @return std::string
             */
            std::string get_name()
            {
                return field_name;
            }
            /**
             * @brief Get a string representation of IRField
             *
             * @return std::string
             */
            std::string to_string();

            /**
             * @brief Function to check if two IRField are the same with shared_ptr.
             *
             * @param field
             * @return true
             * @return false
             */
            bool equal(irfield_t field);

            /**
             * @brief == operator for IRField.
             *
             * @param field1
             * @param field2
             * @return true
             * @return false
             */
            friend bool operator==(IRField &, IRField &);

        private:
            //! Class name of the field
            std::string class_name;
            //! Type of the field
            field_t type;
            //! if type is class set class name
            std::string type_class;
            //! Field name
            std::string field_name;
        };
    }
}

#endif