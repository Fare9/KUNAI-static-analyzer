#include "ir_grammar.hpp"

namespace KUNAI
{
    namespace MJOLNIR
    {

        /**
         * @brief Constructor of IRExpr this will be the most common type of instruction of the IR.
         * @param type: type of the expression (type of the instruction).
         * @param left: left expression for the AST.
         * @param right: right expression for the AST.
         * @return void
         */
        IRExpr::IRExpr(expr_type_t type, std::shared_ptr<IRExpr> left, std::shared_ptr<IRExpr> right)
            : IRStmnt(EXPR_STMNT_T)
        {
            this->type = type;
            this->left = left;
            this->right = right;
        }

        /**
         * @brief Destructor of IRExpr, nothing to be done.
         * @return void
         */
        IRExpr::~IRExpr() {}

        /**
         * @brief Set the left expression (for the AST).
         * @param left: left part of expression for the AST.
         */
        void IRExpr::set_left_expr(std::shared_ptr<IRExpr> left)
        {
            this->left = left;
        }

        /**
         * @brief Set the right expression (for the AST).
         * @param left: right part of expression for the AST.
         */
        void IRExpr::set_right_expr(std::shared_ptr<IRExpr> right)
        {
            this->right = right;
        }

        /**
         * @brief Get the left expression (from the AST)
         * @return std::shared_ptr<IRExpr>
         */
        std::shared_ptr<IRExpr> IRExpr::get_left_expr()
        {
            return left;
        }

        /**
         * @brief Get the right expression (from the AST)
         * @return std::shared_ptr<IRExpr>
         */
        std::shared_ptr<IRExpr> IRExpr::get_right_expr()
        {
            return right;
        }

        /**
         * @brief Get the type of current expression.
         * @return expr_type_t
         */
        IRExpr::expr_type_t IRExpr::get_expression_type()
        {
            return type;
        }

        /**
         * @brief Return the correct string representation for the IRExpr.
         * 
         * @return std::string 
         */
        std::string IRExpr::to_string()
        {
            if (type == BINOP_EXPR_T)
            {
                auto binop = reinterpret_cast<IRBinOp *>(this);

                return binop->to_string();
            }
            else if (type == UNARYOP_EXPR_T)
            {
                auto unaryop = reinterpret_cast<IRUnaryOp *>(this);

                return unaryop->to_string();
            }
            else if (type == ASSIGN_EXPR_T)
            {
                auto assign = reinterpret_cast<IRAssign *>(this);

                return assign->to_string();
            }
            else if (type == TYPE_EXPR_T)
            {
                auto type_ = reinterpret_cast<IRType *>(this);

                return type_->to_string();
            }
            else if (type == LOAD_EXPR_T)
            {
                auto load = reinterpret_cast<IRLoad *>(this);

                return load->to_string();
            }
            else if (type == STORE_EXPR_T)
            {
                auto store = reinterpret_cast<IRStore *>(this);

                return store->to_string();
            }
            else if (type == ZCOMP_EXPR_T)
            {
                auto zcomp = reinterpret_cast<IRZComp *>(this);

                return zcomp->to_string();
            }
            else if (type == BCOMP_EXPR_T)
            {
                auto bcomp = reinterpret_cast<IRBComp *>(this);

                return bcomp->to_string();
            }
            else if (type == CALL_EXPR_T)
            {
                auto call = reinterpret_cast<IRCall *>(this);

                return call->to_string();
            }
            else if (type == NEW_EXPR_T)
            {
                auto new_i = reinterpret_cast<IRNew *>(this);

                return new_i->to_string();
            }
            else if (type == NONE_EXPR_T)
            {
                return "IRExpr [NONE]";
            }

            return "";
        }

        /**
         * @brief Comparison of IRExpr instructions.
         * @return bool
         */
        bool IRExpr::equals(std::shared_ptr<IRExpr> irexpr)
        {
            return *(this) == *(irexpr.get());
        }

        /**
         * @brief Operator == for IRExpr.
         * @param ope1: first operation to compare.
         * @param ope2: second operation to compare.
         * @return bool
         */
        bool operator==(IRExpr &ope1, IRExpr &ope2)
        {
            if (ope1.type != ope2.type)
                return false;

            if (ope1.type == IRExpr::BINOP_EXPR_T)
            {
                IRBinOp &binop1 = reinterpret_cast<IRBinOp &>(ope1);
                IRBinOp &binop2 = reinterpret_cast<IRBinOp &>(ope2);

                return binop1 == binop2;
            }
            else if (ope1.type == IRExpr::UNARYOP_EXPR_T)
            {
                IRUnaryOp &unop1 = reinterpret_cast<IRUnaryOp &>(ope1);
                IRUnaryOp &unop2 = reinterpret_cast<IRUnaryOp &>(ope2);

                return unop1 == unop2;
            }
            else if (ope1.type == IRExpr::ASSIGN_EXPR_T)
            {
                IRAssign &assign1 = reinterpret_cast<IRAssign &>(ope1);
                IRAssign &assign2 = reinterpret_cast<IRAssign &>(ope2);

                return assign1 == assign2;
            }
            else if (ope1.type == IRExpr::CALL_EXPR_T)
            {
                IRCall &call1 = reinterpret_cast<IRCall &>(ope1);
                IRCall &call2 = reinterpret_cast<IRCall &>(ope2);

                return call1 == call2;
            }
            else if (ope1.type == IRExpr::LOAD_EXPR_T)
            {
                IRLoad &load1 = reinterpret_cast<IRLoad &>(ope1);
                IRLoad &load2 = reinterpret_cast<IRLoad &>(ope2);

                return load1 == load2;
            }
            else if (ope1.type == IRExpr::STORE_EXPR_T)
            {
                IRStore &store1 = reinterpret_cast<IRStore &>(ope1);
                IRStore &store2 = reinterpret_cast<IRStore &>(ope2);

                return store1 == store2;
            }
            else if (ope1.type == IRExpr::ZCOMP_EXPR_T)
            {
                IRZComp &zcomp1 = reinterpret_cast<IRZComp &>(ope1);
                IRZComp &zcomp2 = reinterpret_cast<IRZComp &>(ope2);

                return zcomp1 == zcomp2;
            }
            else if (ope1.type == IRExpr::BCOMP_EXPR_T)
            {
                IRBComp &bcomp1 = reinterpret_cast<IRBComp &>(ope1);
                IRBComp &bcomp2 = reinterpret_cast<IRBComp &>(ope2);

                return bcomp1 == bcomp2;
            }
            else if (ope1.type == IRExpr::NEW_EXPR_T)
            {
                IRNew &new1 = reinterpret_cast<IRNew &>(ope1);
                IRNew &new2 = reinterpret_cast<IRNew &>(ope2);

                return new1 == new2;
            }
            else if (ope1.type == IRExpr::TYPE_EXPR_T)
            {
                IRType &type1 = reinterpret_cast<IRType &>(ope1);
                IRType &type2 = reinterpret_cast<IRType &>(ope2);

                return type1 == type2;
            }

            return false;
        }

        /**
         * IRBinOp class
         */

        /**
         * @brief Constructor of IRBinOp, this class represent different instructions with two operators and a result.
         * @param bin_op_type: type of binary operation.
         * @param result: where result of operation is stored.
         * @param op1: first operand of the operation.
         * @param op2: second operand of the operation.
         * @param left: left expression for the AST.
         * @param right: right expression for the AST.
         * @return void
         */
        IRBinOp::IRBinOp(bin_op_t bin_op_type,
                         std::shared_ptr<IRExpr> result,
                         std::shared_ptr<IRExpr> op1,
                         std::shared_ptr<IRExpr> op2,
                         std::shared_ptr<IRExpr> left,
                         std::shared_ptr<IRExpr> right)
            : IRExpr(BINOP_EXPR_T, left, right)
        {
            this->bin_op_type = bin_op_type;
            this->result = result;
            this->op1 = op1;
            this->op2 = op2;
        }

        /**
         * @brief Destructor of IRBinOp.
         * @return void
         */
        IRBinOp::~IRBinOp() {}

        /**
         * @brief Get the type of binary operation applied.
         * @return bin_op_t
         */
        IRBinOp::bin_op_t IRBinOp::get_bin_op_type()
        {
            return bin_op_type;
        }

        /**
         * @brief Get expression of type where result is stored.
         * @return std::shared_ptr<IRExpr>
         */
        std::shared_ptr<IRExpr> IRBinOp::get_result()
        {
            return result;
        }

        /**
         * @brief Get the operand 1 of the operation.
         * @return std::shared_ptr<IRExpr>
         */
        std::shared_ptr<IRExpr> IRBinOp::get_op1()
        {
            return op1;
        }

        /**
         * @brief Get the operand 2 of the operation.
         * @return std::shared_ptr<IRExpr>
         */
        std::shared_ptr<IRExpr> IRBinOp::get_op2()
        {
            return op2;
        }

        /**
         * @brief Return a string representation of IRBinOp.
         * 
         * @return std::string 
         */
        std::string IRBinOp::to_string()
        {
            std::stringstream str_stream;

            str_stream << "IRBinOp ";

            switch (bin_op_type)
            {
            case ADD_OP_T:
                str_stream << "[Op: ADD_OP]";
                break;
            case SUB_OP_T:
                str_stream << "[Op: SUB_OP]";
                break;
            case S_MUL_OP_T:
                str_stream << "[Op: SIGNED_MUL_OP]";
                break;
            case U_MUL_OP_T:
                str_stream << "[Op: UNSIGNED_MUL_OP]";
                break;
            case S_DIV_OP_T:
                str_stream << "[Op: SIGNED_DIV_OP]";
                break;
            case U_DIV_OP_T:
                str_stream << "[Op: UNSIGNED_DIV_OP]";
                break;
            case MOD_OP_T:
                str_stream << "[Op: MOD_OP]";
                break;
            case AND_OP_T:
                str_stream << "[Op: AND_OP]";
                break;
            case XOR_OP_T:
                str_stream << "[Op: XOR_OP]";
                break;
            case OR_OP_T:
                str_stream << "[Op: OR_OP]";
                break;
            case SHL_OP_T:
                str_stream << "[Op: SHL_OP]";
                break;
            case SHR_OP_T:
                str_stream << "[Op: SHR_OP]";
                break;
            case USHR_OP_T:
                str_stream << "[Op: UNSIGNED_SHR_OP]";
                break;
            }

            str_stream << "[Result: " << result->to_string() << "]";
            str_stream << "[Operand1: " << op1->to_string() << "]";
            str_stream << "[Operand2: " << op2->to_string() << "]";

            return str_stream.str();
        }

        /**
         * @brief Comparison of IRBinOp instructions.
         * @return bool
         */
        bool IRBinOp::equals(std::shared_ptr<IRBinOp> irbinop)
        {
            return *this == *(irbinop.get());
        }

        /**
         * @brief Operator == for IRBinOp.
         * @param ope1: first operation to compare.
         * @param ope2: second operation to compare.
         * @return bool
         */
        bool operator==(IRBinOp &ope1, IRBinOp &ope2)
        {
            return (ope1.bin_op_type == ope2.bin_op_type) &&
                   (ope1.op1->equals(ope2.op1)) &&
                   (ope1.op2->equals(ope2.op2)) &&
                   (ope1.result->equals(ope2.result));
        }

        /**
         * IRUnaryOp class
         */

        /**
         * @brief Constructor of IRUnaryOp class, this will get different values for the instruction.
         * @param unary_op_type: type of unary operation.
         * @param result: where the operation stores the result.
         * @param op: operand from the instruction.
         * @param left: left expression for the AST.
         * @param right: right expression for the AST.
         * @return void
         */
        IRUnaryOp::IRUnaryOp(unary_op_t unary_op_type,
                             std::shared_ptr<IRExpr> result,
                             std::shared_ptr<IRExpr> op,
                             std::shared_ptr<IRExpr> left,
                             std::shared_ptr<IRExpr> right)
            : IRExpr(UNARYOP_EXPR_T, left, right)
        {
            this->unary_op_type = unary_op_type;
            this->result = result;
            this->op = op;
            this->cast_type = NONE_CAST;
        }

        /**
         * @brief Constructor of IRUnaryOp class, this will get different values for the instruction.
         * @param unary_op_type: type of unary operation.
         * @param result: where the operation stores the result.
         * @param op: operand from the instruction.
         * @param left: left expression for the AST.
         * @param right: right expression for the AST.
         * @return void
         */
        IRUnaryOp::IRUnaryOp(unary_op_t unary_op_type,
                             cast_type_t cast_type,
                             std::shared_ptr<IRExpr> result,
                             std::shared_ptr<IRExpr> op,
                             std::shared_ptr<IRExpr> left,
                             std::shared_ptr<IRExpr> right)
            : IRExpr(UNARYOP_EXPR_T, left, right)
        {
            this->unary_op_type = unary_op_type;
            this->result = result;
            this->op = op;
            this->cast_type = cast_type;
        }

        /**
         * @brief Destructor of IRUnaryOp class, nothing to be done.
         * @return void
         */
        IRUnaryOp::~IRUnaryOp() {}

        /**
         * @brief Get the type of unary operation of the instruction.
         * @return unary_op_t
         */
        IRUnaryOp::unary_op_t IRUnaryOp::get_unary_op_type()
        {
            return unary_op_type;
        }

        /**
         * @brief Get the IRExpr where result of the operation would be stored.
         * @return std::shared_ptr<IRExpr>
         */
        std::shared_ptr<IRExpr> IRUnaryOp::get_result()
        {
            return result;
        }

        /**
         * @brief Get the IRExpr of the operand of the instruction.
         * @return std::shared_ptr<IRExpr>
         */
        std::shared_ptr<IRExpr> IRUnaryOp::get_op()
        {
            return op;
        }

        /**
         * @brief Set the cast type of the Cast operation.
         * @param cast_type: type of cast to assign.
         * @return void
         */
        void IRUnaryOp::set_cast_type(cast_type_t cast_type)
        {
            this->cast_type = cast_type;
        }

        /**
         * @brief Get the type of cast of the instruction.
         * @return cast_type_t
         */
        IRUnaryOp::cast_type_t IRUnaryOp::get_cast_type()
        {
            return this->cast_type;
        }

        /**
         * @brief Return a string representation of IRUnaryOp.
         * 
         * @return std::string 
         */
        std::string IRUnaryOp::to_string()
        {
            std::stringstream str_stream;

            str_stream << "IRUnaryOp ";

            switch (unary_op_type)
            {
            case INC_OP_T:
                str_stream << "[Type: INC_OP]";
                break;
            case DEC_OP_T:
                str_stream << "[Type: DEC_OP]";
                break;
            case NOT_OP_T:
                str_stream << "[Type: NOT_OP]";
                break;
            case NEG_OP_T:
                str_stream << "[Type: NEG_OP]";
                break;
            case CAST_OP_T:
                str_stream << "[Type: CAST_OP]";
                break;
            case Z_EXT_OP_T:
                str_stream << "[Type: ZERO_EXT_OP]";
                break;
            case S_EXT_OP_T:
                str_stream << "[Type: SIGN_EXT_OP]";
                break;
            }

            if (unary_op_type == CAST_OP_T)
            {
                switch (cast_type)
                {
                case TO_BYTE:
                    str_stream << "[Cast: TO_BYTE]";
                    break;
                case TO_CHAR:
                    str_stream << "[Cast: TO_CHAR]";
                    break;
                case TO_SHORT:
                    str_stream << "[Cast: TO_SHORT]";
                    break;
                case TO_INT:
                    str_stream << "[Cast: TO_INT]";
                    break;
                case TO_LONG:
                    str_stream << "[Cast: TO_LONG]";
                    break;
                case TO_FLOAT:
                    str_stream << "[Cast: TO_FLOAT]";
                    break;
                case TO_DOUBLE:
                    str_stream << "[Cast: TO_DOUBLE]";
                    break;
                case TO_ADDR:
                    str_stream << "[Cast: TO_ADDR]";
                    break;
                case TO_BOOLEAN:
                    str_stream << "[Cast: TO_BOOLEAN]";
                    break;
                }
            }

            str_stream << "[Dest: " << result->to_string() << "]";
            str_stream << "[Src: " << op->to_string() << "]";

            return str_stream.str();
        }

        /**
         * @brief Comparison of IRUnaryOp instructions.
         * @return bool
         */
        bool IRUnaryOp::equals(std::shared_ptr<IRUnaryOp> irunaryop)
        {
            return *this == *(irunaryop.get());
        }

        /**
         * @brief Operator == for IRUnaryOp.
         * @param ope1: first operation to compare.
         * @param ope2: second operation to compare.
         * @return bool
         */
        bool operator==(IRUnaryOp &ope1, IRUnaryOp &ope2)
        {
            return (ope1.unary_op_type == ope2.unary_op_type) &&
                   (ope1.op->equals(ope2.op)) &&
                   (ope1.result->equals(ope2.result));
        }

        /**
         * IRAssign class
         */

        /**
         * @brief Constructor of IRAssign, this one will have just those from the basic types,
         *        no more information is needed, the left part and the right part.
         * @param destination: type where the value will be assigned.
         * @param source: type from where the value is taken.
         * @param left: left part of the assignment (also used for AST).
         * @param right: right part of the assignment (also used for AST).
         * @return void
         */
        IRAssign::IRAssign(std::shared_ptr<IRExpr> destination,
                           std::shared_ptr<IRExpr> source,
                           std::shared_ptr<IRExpr> left,
                           std::shared_ptr<IRExpr> right)
            : IRExpr(ASSIGN_EXPR_T, left, right)
        {
            this->destination = destination;
            this->source = source;
        }

        /**
         * @brief Destructor of IRAssign, nothing to be done here.
         * @return void
         */
        IRAssign::~IRAssign() {}

        /**
         * @brief Get destination type where value is assigned.
         * @return std::shared_ptr<IRExpr>
         */
        std::shared_ptr<IRExpr> IRAssign::get_destination()
        {
            return destination;
        }

        /**
         * @brief Get the source operand from the assignment.
         * @return std::shared_ptr<IRExpr>
         */
        std::shared_ptr<IRExpr> IRAssign::get_source()
        {
            return source;
        }

        /**
         * @brief Return the string representation of IRAssign.
         * 
         * @return std::string 
         */
        std::string IRAssign::to_string()
        {
            std::stringstream str_stream;

            str_stream << "IRAssign ";
            str_stream << "[Dest: " << destination->to_string() << "]";
            str_stream << "[Src: " << source->to_string() << "]";

            return str_stream.str();
        }

        /**
         * @brief Comparison of IRAssign instructions.
         * @return bool
         */
        bool IRAssign::equals(std::shared_ptr<IRAssign> irassign)
        {
            return *this == *(irassign.get());
        }

        /**
         * @brief Operator == for IRAssign.
         * @param ope1: first operation to compare.
         * @param ope2: second operation to compare.
         * @return bool
         */
        bool operator==(IRAssign &ope1, IRAssign &ope2)
        {
            return (ope1.destination->equals(ope2.destination)) &&
                   (ope1.source->equals(ope2.source));
        }

        /**
         * IRCall class
         */

        /**
         * @brief Constructor of IRCall, this expression represents a Call to a function or method.
         * @param callee: function/method called represented as IRExpr (super-super-class of IRCallee).
         * @param args: vector of arguments to the function/method.
         * @param left: left part of the call (used for AST).
         * @param right: right part of the call (used for AST).
         * @return void
         */
        IRCall::IRCall(std::shared_ptr<IRExpr> callee,
                       std::vector<std::shared_ptr<IRExpr>> args,
                       std::shared_ptr<IRExpr> left,
                       std::shared_ptr<IRExpr> right)
            : IRExpr(CALL_EXPR_T, left, right)
        {
            this->callee = callee;
            this->args = args;
            this->ret_val = nullptr;
        }

        /**
         * @brief Destructor of IRCall, nothing to be done.
         * @return void
         */
        IRCall::~IRCall() {}

        /**
         * @brief Get the callee object from the function/method called.
         * @return std::shared_ptr<IRExpr>
         */
        std::shared_ptr<IRExpr> IRCall::get_callee()
        {
            return callee;
        }

        /**
         * @brief Get the arguments from the call.
         * @return std::vector<std::shared_ptr<IRExpr>>
         */
        std::vector<std::shared_ptr<IRExpr>> IRCall::get_args()
        {
            return args;
        }

        /**
         * @brief Get a string representation of IRCall.
         * 
         * @return std::string 
         */
        std::string IRCall::to_string()
        {
            std::stringstream str_stream;

            str_stream << "IRCall ";

            str_stream << "[Callee: " << callee->to_string() << "]";

            for (auto arg : args)
                str_stream << "[Param: " << arg->to_string() << "]";

            if (ret_val)
                str_stream << "[Return: " << ret_val->to_string() << "]";

            return str_stream.str();
        }

        /**
         * @brief Set the ret_val object
         * 
         * @param ret_val 
         */
        void IRCall::set_ret_val(std::shared_ptr<IRExpr> ret_val)
        {
            this->ret_val = ret_val;
        }

        /**
         * @brief Get the ret_val object
         * 
         * @return std::shared_ptr<IRExpr> 
         */
        std::shared_ptr<IRExpr> IRCall::get_ret_val()
        {
            return ret_val;
        }

        /**
         * @brief Comparison of IRCall instructions.
         * @return bool
         */
        bool IRCall::equals(std::shared_ptr<IRCall> ircall)
        {
            return *(this) == *(ircall.get());
        }

        /**
         * @brief Operator == for IRCall.
         * @param ope1: first operation to compare.
         * @param ope2: second operation to compare.
         * @return bool
         */
        bool operator==(IRCall &ope1, IRCall &ope2)
        {
            return (ope1.callee->equals(ope2.callee)) &&
                   (std::equal(ope1.args.begin(), ope1.args.end(), ope2.args.begin()));
        }

        /**
         * IRLoad class
         */

        /**
         * @brief Constructor of IRLoad class, this class represent a load from memory (using memory or using register).
         * @param destination: register where the value will be stored.
         * @param source: expression from where the memory will be retrieved.
         * @param size: loaded size.
         * @param left: left part of the load (used for AST).
         * @param right: right part of the load (used for AST).
         * @return void
         */
        IRLoad::IRLoad(std::shared_ptr<IRExpr> destination,
                       std::shared_ptr<IRExpr> source,
                       std::uint32_t size,
                       std::shared_ptr<IRExpr> left,
                       std::shared_ptr<IRExpr> right)
            : IRExpr(LOAD_EXPR_T, left, right)
        {
            this->destination = destination;
            this->source = source;
            this->index = nullptr;
            this->size = size;
        }

        /**
         * @brief Constructor of IRLoad class, this class represent a load from memory (using memory or using register).
         * @param destination: register where the value will be stored.
         * @param source: expression from where the memory will be retrieved.
         * @param index: index from the load if this is referenced with an index.
         * @param size: loaded size.
         * @param left: left part of the load (used for AST).
         * @param right: right part of the load (used for AST).
         * @return void
         */
        IRLoad::IRLoad(std::shared_ptr<IRExpr> destination,
                       std::shared_ptr<IRExpr> source,
                       std::shared_ptr<IRExpr> index,
                       std::uint32_t size,
                       std::shared_ptr<IRExpr> left,
                       std::shared_ptr<IRExpr> right)
            : IRExpr(LOAD_EXPR_T, left, right)
        {
            this->destination = destination;
            this->source = source;
            this->index = index;
            this->size = size;
        }

        /**
         * @brief Destructor of IRLoad, nothing to be done.
         * @return void
         */
        IRLoad::~IRLoad() {}

        /**
         * @brief Get destination register from the load.
         * @return std::shared_ptr<IRExpr>
         */
        std::shared_ptr<IRExpr> IRLoad::get_destination()
        {
            return destination;
        }

        /**
         * @brief Get the expression from where value is loaded.
         * @return std::shared_ptr<IRExpr>
         */
        std::shared_ptr<IRExpr> IRLoad::get_source()
        {
            return source;
        }

        /**
         * @brief Get the loaded size.
         * @return std::uint32_t
         */
        std::uint32_t IRLoad::get_size()
        {
            return size;
        }

        /**
         * @brief Return string representation of IRLoad.
         * 
         * @return std::string 
         */
        std::string IRLoad::to_string()
        {
            std::stringstream str_stream;

            str_stream << "IRLoad ";

            str_stream << "[Size: " << size / 8 << "]";
            str_stream << "[Dest: " << destination->to_string() << "]";
            str_stream << "[Src: Mem(" << source->to_string() << ")]";

            if (index != nullptr)
                str_stream << "[Index: " << index->to_string() << "]";

            return str_stream.str();
        }

        /**
         * @brief Comparison of IRLoad instruction.
         * @return bool
         */
        bool IRLoad::equals(std::shared_ptr<IRLoad> irload)
        {
            return *this == *(irload.get());
        }

        /**
         * @brief Operator == for IRLoad.
         * @param ope1: first operation to compare.
         * @param ope2: second operation to compare.
         * @return bool
         */
        bool operator==(IRLoad &ope1, IRLoad &ope2)
        {
            return (ope1.destination->equals(ope2.destination)) &&
                   (ope1.source->equals(ope2.source)) &&
                   (ope1.size == ope2.size);
        }

        /**
         * IRStore class
         */

        /**
         * @brief Constructor of IRStore class, this represent an store to memory instruction.
         * @param destination: Expression where value is written to.
         * @param source: register with the value to be stored.
         * @param size: size of the stored value.
         * @param left: left part of the store (used for AST).
         * @param right: right part of the store (used for AST).
         * @return void
         */
        IRStore::IRStore(std::shared_ptr<IRExpr> destination,
                         std::shared_ptr<IRExpr> source,
                         std::uint32_t size,
                         std::shared_ptr<IRExpr> left,
                         std::shared_ptr<IRExpr> right)
            : IRExpr(STORE_EXPR_T, left, right)
        {
            this->destination = destination;
            this->source = source;
            this->index = nullptr;
            this->size = size;
        }

        /**
         * @brief Constructor of IRStore class, this represent an store to memory instruction.
         * @param destination: Expression where value is written to.
         * @param source: register with the value to be stored.
         * @param index: index where value is stored.
         * @param size: size of the stored value.
         * @param left: left part of the store (used for AST).
         * @param right: right part of the store (used for AST).
         * @return void
         */
        IRStore::IRStore(std::shared_ptr<IRExpr> destination,
                         std::shared_ptr<IRExpr> source,
                         std::shared_ptr<IRExpr> index,
                         std::uint32_t size,
                         std::shared_ptr<IRExpr> left,
                         std::shared_ptr<IRExpr> right)
            : IRExpr(STORE_EXPR_T, left, right)
        {
            this->destination = destination;
            this->source = source;
            this->index = index;
            this->size = size;
        }

        /**
         * @brief Destructor of IRStore class, nothing to be done.
         * @return void
         */
        IRStore::~IRStore() {}

        /**
         * @brief Get expression destination where value is going to be written.
         * @return std::shared_ptr<IRExpr>
         */
        std::shared_ptr<IRExpr> IRStore::get_destination()
        {
            return destination;
        }

        /**
         * @brief Get source where value is taken from.
         * @return std::shared_ptr<IRExpr>
         */
        std::shared_ptr<IRExpr> IRStore::get_source()
        {
            return source;
        }

        /**
         * @brief Get size of value to be written.
         * @return std::uint32_t
         */
        std::uint32_t IRStore::get_size()
        {
            return size;
        }

        /**
         * @brief Return a string representation of IRStore.
         * 
         * @return std::string 
         */
        std::string IRStore::to_string()
        {
            std::stringstream str_stream;

            str_stream << "IRStore ";

            str_stream << "[Size: " << size / 8 << "]";
            str_stream << "[Dest: Mem(" << destination->to_string() << ")]";

            if (index != nullptr)
                str_stream << "[Index: " << index->to_string() << "]";

            str_stream << "[Src: " << source->to_string() << "]";

            return str_stream.str();
        }
        /**
         * @brief Comparison of IRStore instructions.
         * @return bool
         */
        bool IRStore::equals(std::shared_ptr<IRStore> irstore)
        {
            return *(this) == *(irstore.get());
        }

        /**
         * @brief Operator == for IRStore.
         * @param ope1: first operation to compare.
         * @param ope2: second operation to compare.
         * @return bool
         */
        bool operator==(IRStore &ope1, IRStore &ope2)
        {
            return (ope1.destination->equals(ope2.destination)) &&
                   (ope1.source->equals(ope2.source)) &&
                   (ope1.size == ope2.size);
        }

        /**
         * IRZComp class
         */

        /**
         * @brief Constructor of IRZComp, this is a comparison with zero.
         * @param comp: type of comparison (== or !=).
         * @param result: register or temporal register where result is stored.
         * @param reg: register used in the comparison.
         * @param left: left part of the zero comparison (used for AST).
         * @param right: right part of the zero comparison (used for AST).
         * @return void
         */

        IRZComp::IRZComp(zero_comp_t comp,
                         std::shared_ptr<IRExpr> result,
                         std::shared_ptr<IRExpr> reg,
                         std::shared_ptr<IRExpr> left,
                         std::shared_ptr<IRExpr> right)
            : IRExpr(ZCOMP_EXPR_T, left, right)
        {
            this->comp = comp;
            this->result = result;
            this->reg = reg;
        }

        /**
         * @brief Destructor of IRZComp, nothing to be done.
         * @return void
         */
        IRZComp::~IRZComp() {}

        /**
         * @brief Return the result register or temporal register.
         * 
         * @return std::shared_ptr<IRExpr> 
         */
        std::shared_ptr<IRExpr> IRZComp::get_result()
        {
            return result;
        }

        /**
         * @brief Get register used in the comparison.
         * @return std::shared_ptr<IRExpr>
         */
        std::shared_ptr<IRExpr> IRZComp::get_reg()
        {
            return reg;
        }

        /**
         * @brief Get the type of comparison with zero.
         * @return zero_comp_t
         */
        IRZComp::zero_comp_t IRZComp::get_comparison()
        {
            return comp;
        }

        /**
         * @brief Return the string expression of IRZComp.
         * 
         * @return std::string 
         */
        std::string IRZComp::to_string()
        {
            std::stringstream str_stream;

            str_stream << "IRZComp ";

            switch (comp)
            {
            case EQUAL_ZERO_T:
                str_stream << "[Comp: EQUAL_ZERO_T]";
                break;
            case NOT_EQUAL_ZERO_T:
                str_stream << "[Comp: NOT_EQUAL_ZERO_T]";
                break;
            case LOWER_ZERO_T:
                str_stream << "[Comp: LOWER_ZERO_T]";
                break;
            case GREATER_EQUAL_ZERO:
                str_stream << "[Comp: GREATER_EQUAL_ZERO]";
                break;
            case GREATER_ZERO_T:
                str_stream << "[Comp: GREATER_ZERO_T]";
                break;
            case LOWER_EQUAL_ZERO:
                str_stream << "[Comp: LOWER_EQUAL_ZERO]";
            }

            str_stream << "[Result: " << result->to_string() << "]";
            str_stream << "[Operand: " << reg->to_string() << "]";

            return str_stream.str();
        }

        /**
         * @brief Comparison of IRZComp instruction with shared_ptr.
         * @return bool
         */
        bool IRZComp::equals(std::shared_ptr<IRZComp> irzcomp)
        {
            return *this == *(irzcomp.get());
        }

        /**
         * @brief Operator == for IRZComp.
         * @param ope1: first operation to compare.
         * @param ope2: second operation to compare.
         * @return bool
         */
        bool operator==(IRZComp &ope1, IRZComp &ope2)
        {
            return (ope1.comp == ope2.comp) &&
                   (ope1.reg->equals(ope2.reg));
        }

        /**
         * IRBComp class
         */

        /**
         * @brief Constructor of IRBComp, this class represent a comparison between two types.
         * @param comp: type of comparison from an enum.
         * @param result: register or temporal register where result is stored.
         * @param reg1: first type where the comparison is applied.
         * @param reg2: second type where the comparison is applied.
         * @param left: left part of the comparison (used for AST).
         * @param right: right part of the comparison (used for AST).
         * @return void
         */
        IRBComp::IRBComp(comp_t comp,
                         std::shared_ptr<IRExpr> result,
                         std::shared_ptr<IRExpr> reg1,
                         std::shared_ptr<IRExpr> reg2,
                         std::shared_ptr<IRExpr> left,
                         std::shared_ptr<IRExpr> right)
            : IRExpr(BCOMP_EXPR_T, left, right)
        {
            this->comp = comp;
            this->result = result;
            this->reg1 = reg1;
            this->reg2 = reg2;
        }

        /**
         * @brief Destructor of IRBComp, nothing to be done.
         * @return void
         */
        IRBComp::~IRBComp() {}

        /**
         * @brief Get the result register or temporal register.
         * 
         * @return std::shared_ptr<IRExpr> 
         */
        std::shared_ptr<IRExpr> IRBComp::get_result()
        {
            return result;
        }

        /**
         * @brief Return the first part of the comparison.
         * @return std::shared_ptr<IRExpr>
         */
        std::shared_ptr<IRExpr> IRBComp::get_reg1()
        {
            return reg1;
        }

        /**
         * @brief Return the second part of the comparison.
         * @return std::shared_ptr<IRExpr>
         */
        std::shared_ptr<IRExpr> IRBComp::get_reg2()
        {
            return reg2;
        }

        /**
         * @brief Return the type of comparison
         * @return comp_t
         */
        IRBComp::comp_t IRBComp::get_comparison()
        {
            return comp;
        }

        /**
         * @brief Return the string representation of IRBComp.
         * 
         * @return std::string 
         */
        std::string IRBComp::to_string()
        {
            std::stringstream str_stream;

            str_stream << "IRBComp ";

            switch (comp)
            {
            case EQUAL_T:
                str_stream << "[Comp: EQUAL_T]";
                break;
            case NOT_EQUAL_T:
                str_stream << "[Comp: NOT_EQUAL_T]";
                break;
            case GREATER_T:
                str_stream << "[Comp: GREATER_T]";
                break;
            case GREATER_EQUAL_T:
                str_stream << "[Comp: GREATER_EQUAL_T]";
                break;
            case LOWER_T:
                str_stream << "[Comp: LOWER_T]";
                break;
            case LOWER_EQUAL_T:
                str_stream << "[Comp: LOWER_EQUAL_T]";
                break;
            case ABOVE_T:
                str_stream << "[Comp: ABOVE_T]";
                break;
            case ABOVE_EQUAL_T:
                str_stream << "[Comp: ABOVE_EQUAL_T]";
                break;
            case BELOW_T:
                str_stream << "[Comp: BELOW_T]";
                break;
            }

            str_stream << "[Result: " << result->to_string() << "]";
            str_stream << "[Op1: " << reg1->to_string() << "]";
            str_stream << "[Op2: " << reg2->to_string() << "]";

            return str_stream.str();
        }

        /**
         * @brief Comparison of IRBComp instruction with shared_ptr.
         * @return bool
         */
        bool IRBComp::equals(std::shared_ptr<IRBComp> bcomp)
        {
            return *this == *(bcomp.get());
        }

        /**
         * @brief Operator == for IRBComp.
         * @param ope1: first operation to compare.
         * @param ope2: second operation to compare.
         * @return bool
         */
        bool operator==(IRBComp &ope1, IRBComp &ope2)
        {
            return (ope1.comp == ope2.comp) &&
                   (ope1.reg1->equals(ope2.reg1)) &&
                   (ope1.reg2->equals(ope2.reg2));
        }

        /**
         * IRNew class
         */
        
        /**
         * @brief Construct a new IRNew::IRNew object which represents
         *        the creation of an instance of a class.
         * 
         * @param result: result register where object is stored.
         * @param class_instance: IRClass object which represent the instance.
         * @param left: left part of the comparison (used for AST).
         * @param right: right part of the comparison (used for AST).
         * @return void
         */
        IRNew::IRNew(std::shared_ptr<IRExpr> result,
                     std::shared_ptr<IRExpr> class_instance,
                     std::shared_ptr<IRExpr> left,
                     std::shared_ptr<IRExpr> right)
            : IRExpr(NEW_EXPR_T, nullptr, nullptr)
        {
            this->result = result;
            this->class_instance;
        }

        /**
         * @brief Destroy the IRNew::IRNew object
         * 
         */
        IRNew::~IRNew() {}

        /**
         * @brief Get the destination register
         * 
         * @return std::shared_ptr<IRExpr> 
         */
        std::shared_ptr<IRExpr> IRNew::get_result()
        {
            return this->result;
        }

        /**
         * @brief Get the source class.
         * 
         * @return std::shared_ptr<IRExpr> 
         */
        std::shared_ptr<IRExpr> IRNew::get_source_class()
        {
            return this->class_instance;
        }

        /**
         * @brief Return string representation of IRNew
         * 
         * @return std::string 
         */
        std::string IRNew::to_string()
        {
            std::stringstream stream;

            stream << "IRNew ";

            stream << "[Destination: " << result->to_string() << "]";
            stream << "[Source: " << result->to_string() << "]";

            return stream.str();
        }

        /**
         * @brief Compare two IRNew instructions with shared_ptr
         * 
         * @param new_i 
         * @return true 
         * @return false 
         */
        bool IRNew::equals(std::shared_ptr<IRNew> new_i)
        {
            return (*this) == *(new_i.get());
        }

        /**
         * @brief Operator == for IRNew instruction.
         * 
         * @param new1 
         * @param new2 
         * @return true 
         * @return false 
         */
        bool operator==(IRNew& new1, IRNew& new2)
        {
            return (new1.result->equals(new2.result)) &&
                    (new1.class_instance->equals(new2.class_instance));
        }
    }
}