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

#include <iostream>
#include <sstream>
#include <fstream>
#include <memory>
#include <vector>
#include <list>

// lifters
#include "arch/ir_x86.hpp"
#include "arch/ir_dalvik.hpp"

namespace KUNAI
{
    namespace MJOLNIR
    {
        class IRBlock;
        class IRStmnt;
        class IRUJmp;
        class IRCJmp;
        class IRRet;
        class IRExpr;
        class IRBinOp;
        class IRUnaryOp;
        class IRAssign;
        class IRCall;
        class IRLoad;
        class IRStore;
        class IRZComp;
        class IRBComp;
        class IRType;
        class IRReg;
        class IRTempReg;
        class IRConstInt;
        class IRMemory;
        class IRString;
        class IRClass;
        class IRCallee;

        /**
         * Useful methods you can use
         * for getting information about
         * the instructions.
         */
        bool is_ir_field(std::shared_ptr<IRStmnt> instr);
        bool is_ir_callee(std::shared_ptr<IRStmnt> instr);
        bool is_ir_class(std::shared_ptr<IRStmnt> instr);
        bool is_ir_string(std::shared_ptr<IRStmnt> instr);
        bool is_ir_memory(std::shared_ptr<IRStmnt> instr);
        bool is_ir_const_int(std::shared_ptr<IRStmnt> instr);
        bool is_ir_temp_reg(std::shared_ptr<IRStmnt> instr);
        bool is_ir_register(std::shared_ptr<IRStmnt> instr);

        

        bool is_unconditional_jump(std::shared_ptr<IRStmnt> instr);
        bool is_conditional_jump(std::shared_ptr<IRStmnt> instr);
        bool is_ret(std::shared_ptr<IRStmnt> instr);
        bool is_call(std::shared_ptr<IRStmnt> instr);

        class IRBlock
        {
        public:
            IRBlock();
            ~IRBlock();

            void append_statement_to_block(std::shared_ptr<IRStmnt> statement);

            bool delete_statement_by_position(size_t pos);

            size_t get_number_of_statements();
            std::list<std::shared_ptr<IRStmnt>> get_statements();

            void set_start_idx(uint64_t start_idx);
            void set_end_idx(uint64_t end_idx);
            uint64_t get_start_idx();
            uint64_t get_end_idx();

            std::string get_name();
            std::string to_string();
        private:
            //! starting idx and last idx (used for jump calculation)
            uint64_t start_idx, end_idx;
            //! statements from the basic block.
            std::list<std::shared_ptr<IRStmnt>> block_statements;
        };

        class IRStmnt
        {
        public:
            enum stmnt_type_t
            {
                UJMP_STMNT_T,
                CJMP_STMNT_T,
                RET_STMNT_T,
                EXPR_STMNT_T,
                NONE_STMNT_T = 99 // used to finish the chain of statements
            };

            IRStmnt(stmnt_type_t stmnt_type);
            virtual ~IRStmnt() = default;
            

            stmnt_type_t get_statement_type();

            std::string to_string();
        private:
            //! Type of the statement.
            stmnt_type_t stmnt_type;
        };

        class IRUJmp : public IRStmnt
        {
        public:
            IRUJmp(uint64_t addr, std::shared_ptr<IRBlock> target);
            ~IRUJmp();

            void set_jump_target(std::shared_ptr<IRBlock> target);
            std::shared_ptr<IRBlock> get_jump_target();

            uint64_t get_jump_addr();

            std::string to_string();
        private:
            //! offset or address of target
            uint64_t addr;
            //! target where the jump will fall
            std::shared_ptr<IRBlock> target;
        };

        class IRCJmp : public IRStmnt
        {
        public:
            IRCJmp(uint64_t addr, std::shared_ptr<IRStmnt> condition, std::shared_ptr<IRBlock> target, std::shared_ptr<IRBlock> fallthrough);
            ~IRCJmp();

            uint64_t get_addr();
            std::shared_ptr<IRStmnt> get_condition();
            void set_jump_target(std::shared_ptr<IRBlock> target);
            std::shared_ptr<IRBlock> get_jump_target();
            void set_fallthrough_Target(std::shared_ptr<IRBlock> fallthrough);
            std::shared_ptr<IRBlock> get_fallthrough_target();

            std::string to_string();
        private:
            //! offset or address of target
            uint64_t addr;
            //! Condition for taking the target jump
            std::shared_ptr<IRStmnt> condition;
            //! target where the jump will fall
            std::shared_ptr<IRBlock> target;
            //! fallthrough target.
            std::shared_ptr<IRBlock> fallthrough;
        };

        class IRRet : public IRStmnt
        {
        public:
            IRRet(std::shared_ptr<IRStmnt> ret_value);
            ~IRRet();

            std::shared_ptr<IRStmnt> get_return_value();

            std::string to_string();
        private:
            //! Returned value, commonly a NONE IRType, or an IRReg.
            std::shared_ptr<IRStmnt> ret_value;
        };
    
        class IRExpr : public IRStmnt
        {
        public:
            enum expr_type_t
            {
                BINOP_EXPR_T,
                UNARYOP_EXPR_T,
                ASSIGN_EXPR_T,
                CALL_EXPR_T,
                TYPE_EXPR_T,
                LOAD_EXPR_T,
                STORE_EXPR_T,
                ZCOMP_EXPR_T,
                BCOMP_EXPR_T,
                NONE_EXPR_T = 99 // used to finish the expressions
            };

            IRExpr(expr_type_t type, std::shared_ptr<IRExpr> left, std::shared_ptr<IRExpr> right);
            ~IRExpr();

            void set_left_expr(std::shared_ptr<IRExpr> left);
            void set_right_expr(std::shared_ptr<IRExpr> right);

            std::shared_ptr<IRExpr> get_left_expr();
            std::shared_ptr<IRExpr> get_right_expr();

            expr_type_t get_expression_type();

            std::string to_string();
            bool equals(std::shared_ptr<IRExpr> irexpr);

            friend bool operator==(IRExpr&, IRExpr&);

        private:
            //! pointers used for the abstract syntax tree
            //! these pointers are common in the IRExpr.
            std::shared_ptr<IRExpr> left;
            std::shared_ptr<IRExpr> right;

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

            IRBinOp(bin_op_t bin_op_type,
                    std::shared_ptr<IRExpr> result,
                    std::shared_ptr<IRExpr> op1,
                    std::shared_ptr<IRExpr> op2,
                    std::shared_ptr<IRExpr> left,
                    std::shared_ptr<IRExpr> right);

            ~IRBinOp();

            bin_op_t get_bin_op_type();
            std::shared_ptr<IRExpr> get_result();
            std::shared_ptr<IRExpr> get_op1();
            std::shared_ptr<IRExpr> get_op2();

            std::string to_string();
            bool equals(std::shared_ptr<IRBinOp> irbinop);
            friend bool operator==(IRBinOp&, IRBinOp&);
        private:
            //! type of binary operation
            bin_op_t bin_op_type;
            //! IRBinOp =>  IRExpr(result) = IRExpr(op1) <binop> IRExpr(op2)
            //! for the result we will have an IRExpr too.
            std::shared_ptr<IRExpr> result;
            //! now each one of the operators
            std::shared_ptr<IRExpr> op1;
            std::shared_ptr<IRExpr> op2;
        };

        class IRUnaryOp : public IRExpr
        {
        public:
            enum unary_op_t
            {
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
            };

            IRUnaryOp(unary_op_t unary_op_type,
                      std::shared_ptr<IRExpr> result,
                      std::shared_ptr<IRExpr> op,
                      std::shared_ptr<IRExpr> left,
                      std::shared_ptr<IRExpr> right);

            IRUnaryOp(unary_op_t unary_op_type,
                      cast_type_t cast_type,
                      std::shared_ptr<IRExpr> result,
                      std::shared_ptr<IRExpr> op,
                      std::shared_ptr<IRExpr> left,
                      std::shared_ptr<IRExpr> right);

            ~IRUnaryOp();

            unary_op_t get_unary_op_type();
            std::shared_ptr<IRExpr> get_result();
            std::shared_ptr<IRExpr> get_op();

            void set_cast_type(cast_type_t cast_type);
            cast_type_t get_cast_type();

            std::string to_string();
            bool equals(std::shared_ptr<IRUnaryOp> irunaryop);
            friend bool operator==(IRUnaryOp&, IRUnaryOp&);
        private:
            //! type of unary operation =D
            unary_op_t unary_op_type;
            //! used for casting operations
            cast_type_t cast_type;
            //! IRUnaryOp => IRExpr(result) = <unaryop> IRExpr(op)
            //! an IRExpr for where the result is stored.
            std::shared_ptr<IRExpr> result;
            // operator
            std::shared_ptr<IRExpr> op;
        };

        class IRAssign : public IRExpr
        {
        public:
            IRAssign(std::shared_ptr<IRExpr> destination,
                     std::shared_ptr<IRExpr> source,
                     std::shared_ptr<IRExpr> left,
                     std::shared_ptr<IRExpr> right);
            ~IRAssign();

            std::shared_ptr<IRExpr> get_destination();
            std::shared_ptr<IRExpr> get_source();
            std::string to_string();
            bool equals(std::shared_ptr<IRAssign> irassign);
            friend bool operator==(IRAssign&, IRAssign&);
        private:
            //! destination where the value will be stored.
            std::shared_ptr<IRExpr> destination;
            //! source expression from where the value is taken
            std::shared_ptr<IRExpr> source;
        };

        class IRCall : public IRExpr
        {
        public:
            IRCall(std::shared_ptr<IRExpr> callee,
                   std::vector<std::shared_ptr<IRExpr>> args,
                   std::shared_ptr<IRExpr> left,
                   std::shared_ptr<IRExpr> right);

            ~IRCall();

            std::shared_ptr<IRExpr> get_callee();
            std::vector<std::shared_ptr<IRExpr>> get_args();
            std::string to_string();

            void set_ret_val(std::shared_ptr<IRExpr> ret_val);
            std::shared_ptr<IRExpr> get_ret_val();

            bool equals(std::shared_ptr<IRCall> ircall);
            friend bool operator==(IRCall&, IRCall&);
        private:
            //! Type representing the function/method called
            std::shared_ptr<IRExpr> callee;
            //! Vector with possible arguments
            std::vector<std::shared_ptr<IRExpr>> args;
            //! Return value (if it's for example a register)
            std::shared_ptr<IRExpr> ret_val;
        };

        class IRLoad : public IRExpr
        {
        public:
            IRLoad(std::shared_ptr<IRExpr> destination,
                   std::shared_ptr<IRExpr> source,
                   std::uint32_t size,
                   std::shared_ptr<IRExpr> left,
                   std::shared_ptr<IRExpr> right);

            IRLoad(std::shared_ptr<IRExpr> destination,
                   std::shared_ptr<IRExpr> source,
                   std::shared_ptr<IRExpr> index,
                   std::uint32_t size,
                   std::shared_ptr<IRExpr> left,
                   std::shared_ptr<IRExpr> right);

            ~IRLoad();

            std::shared_ptr<IRExpr> get_destination();
            std::shared_ptr<IRExpr> get_source();
            std::shared_ptr<IRExpr> get_index();
            std::uint32_t get_size();

            std::string to_string();
            bool equals(std::shared_ptr<IRLoad> irload);

            friend bool operator==(IRLoad&, IRLoad&);
        private:
            //! Register where the memory pointed by a register will be loaded.
            std::shared_ptr<IRExpr> destination;
            //! Expression from where memory is read.
            std::shared_ptr<IRExpr> source;
            //! Index if this is referenced by for example a register.
            std::shared_ptr<IRExpr> index;
            //! Size of loaded value
            std::uint32_t size;
        };

        class IRStore : public IRExpr
        {
        public:
            IRStore(std::shared_ptr<IRExpr> destination,
                    std::shared_ptr<IRExpr> source,
                    std::uint32_t size,
                    std::shared_ptr<IRExpr> left,
                    std::shared_ptr<IRExpr> right);
            IRStore(std::shared_ptr<IRExpr> destination,
                    std::shared_ptr<IRExpr> source,
                    std::shared_ptr<IRExpr> index,
                    std::uint32_t size,
                    std::shared_ptr<IRExpr> left,
                    std::shared_ptr<IRExpr> right);
            ~IRStore();

            std::shared_ptr<IRExpr> get_destination();
            std::shared_ptr<IRExpr> get_source();
            std::uint32_t get_size();

            std::string to_string();
            bool equals(std::shared_ptr<IRStore> irstore);

            friend bool operator==(IRStore&, IRStore&);
        private:
            //! Memory pointed by register where value will be stored.
            std::shared_ptr<IRExpr> destination;
            //! Expression with source of value to be stored.
            std::shared_ptr<IRExpr> source;
            //! Index if this is referenced by for example a register.
            std::shared_ptr<IRExpr> index;
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

            IRZComp(zero_comp_t comp,
                    std::shared_ptr<IRExpr> result,
                    std::shared_ptr<IRExpr> reg,
                    std::shared_ptr<IRExpr> left,
                    std::shared_ptr<IRExpr> right);
            ~IRZComp();

            std::shared_ptr<IRExpr> get_result();
            std::shared_ptr<IRExpr> get_reg();
            zero_comp_t get_comparison();

            std::string to_string();
            bool equals(std::shared_ptr<IRZComp> irzcomp);
            
            friend bool operator==(IRZComp&, IRZComp&);
        private:
            //! Register where result is stored
            std::shared_ptr<IRExpr> result;
            //! Register for comparison with zero.
            std::shared_ptr<IRExpr> reg;
            //! Type of comparison
            zero_comp_t comp;
        };

        class IRBComp : public IRExpr
        {
        public:
            enum comp_t
            {
                EQUAL_T, // ==
                NOT_EQUAL_T, // !=
                GREATER_T, // >
                GREATER_EQUAL_T, // >=
                LOWER_T, // <
                LOWER_EQUAL_T, // <=
                ABOVE_T, // (unsigned) >
                ABOVE_EQUAL_T, // (unsigned) >=
                BELOW_T, // (unsigned) <
            };

            IRBComp(comp_t comp,
                    std::shared_ptr<IRExpr> result,
                    std::shared_ptr<IRExpr> reg1,
                    std::shared_ptr<IRExpr> reg2,
                    std::shared_ptr<IRExpr> left,
                    std::shared_ptr<IRExpr> right);
            ~IRBComp();

            std::shared_ptr<IRExpr> get_result();
            std::shared_ptr<IRExpr> get_reg1();
            std::shared_ptr<IRExpr> get_reg2();
            comp_t get_comparison();

            std::string to_string();
            bool equals(std::shared_ptr<IRBComp> bcomp);

            friend bool operator==(IRBComp&, IRBComp&);

        private:
            //! register or temporal register where result is stored
            std::shared_ptr<IRExpr> result;
            //! registers used in the comparisons.
            std::shared_ptr<IRExpr> reg1;
            std::shared_ptr<IRExpr> reg2;
            //! Type of comparison
            comp_t comp;
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

            IRType(type_t type, std::string type_name, size_t type_size);
            ~IRType();

            std::string get_type_name();
            size_t get_type_size();

            type_t get_type();
            std::string get_type_str();
            mem_access_t get_access();

            void write_annotations(std::string annotations);
            std::string read_annotations();

            std::string to_string();
            bool equal(std::shared_ptr<IRType> type);

            friend bool operator==(IRType&, IRType&);
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
            IRReg(std::uint32_t reg_id, int current_arch, std::string type_name, size_t type_size);
            ~IRReg();

            std::uint32_t get_id();

            std::string get_type_str();
            mem_access_t get_access();

            std::string to_string();
            bool equal(std::shared_ptr<IRReg> reg);

            friend bool operator==(IRReg&, IRReg&);
        private:
            //! id of the register, this will be an enum
            //! in case the arquitecture contains a known set
            //! of registers, for example x86-64 will have a
            //! well known set of registers, e.g. EAX, AX, RSP
            //! RIP, etc.
            //! Other arquitectures like DEX VM will not have
            //! an specific set.
            std::uint32_t id;

            int current_arch;
        };

        class IRTempReg : public IRType
        {
        public:
            IRTempReg(std::uint32_t reg_id, std::string type_name, size_t type_size);
            ~IRTempReg();

            std::uint32_t get_id();

            std::string get_type_str();
            mem_access_t get_access();

            std::string to_string();
            bool equal(std::shared_ptr<IRTempReg> temp_reg);

            friend bool operator==(IRTempReg&, IRTempReg&);
        private:
            //! This id will be just an incremental number
            //! as these are temporal registers.
            std::uint32_t id;
        };

        class IRConstInt : public IRType
        {
        public:
            IRConstInt(std::uint64_t value, bool is_signed, mem_access_t byte_order, std::string type_name, size_t type_size);
            ~IRConstInt();

            bool get_is_signed();
            
            std::string get_type_str();
            mem_access_t get_access();

            std::string to_string();
            bool equal(std::shared_ptr<IRConstInt> const_int);

            friend bool operator==(IRConstInt&, IRConstInt&);
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
            IRMemory(std::uint64_t mem_address, std::int32_t offset, mem_access_t byte_order, std::string type_name, size_t type_size);
            ~IRMemory();

            std::uint64_t get_mem_address();
            std::int32_t get_offset();

            std::string get_type_str();
            mem_access_t get_access();

            std::string to_string();
            bool equal(std::shared_ptr<IRMemory> memory);

            friend bool operator==(IRMemory&, IRMemory&);
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
            IRString(std::string str_value, std::string type_name, size_t type_size);
            ~IRString();

            std::string get_str_value();

            std::string get_type_str();
            mem_access_t get_access();

            std::string to_string();
            bool equal(std::shared_ptr<IRString> str);

            friend bool operator==(IRString&, IRString&);
        private:
            //! string value, probably nothing more will be here
            std::string str_value;
        };

        class IRClass : public IRType
        {
        public:
            IRClass(std::string class_name, std::string type_name, size_t type_size);
            ~IRClass();

            std::string get_class();

            std::string get_type_str();
            mem_access_t get_access();

            std::string to_string();
            bool equal(std::shared_ptr<IRClass> class_);

            friend bool operator==(IRClass&, IRClass&);
        private:
            //! class name including path, used for instructions
            //! of type const-class
            std::string class_name;
        };

        class IRCallee : public IRType
        {
        public:
            IRCallee(std::uint64_t addr,
                     std::string name,
                     std::string class_name,
                     int n_of_params,
                     std::string description,
                     std::string type_name, 
                     size_t type_size);
            ~IRCallee();

            std::uint64_t get_addr();
            std::string get_name();
            std::string get_class_name();
            int get_number_of_params();
            std::string get_description();

            std::string get_type_str();
            mem_access_t get_access();

            std::string to_string();
            bool equal(std::shared_ptr<IRCallee> callee);
            
            friend bool operator==(IRCallee&, IRCallee&);
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

            ~IRField();

            std::string get_class_name();
            field_t get_type();
            std::string get_type_class();
            std::string get_name();

            std::string to_string();
            bool equal(std::shared_ptr<IRField> field);
            
            friend bool operator==(IRField&, IRField&);
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