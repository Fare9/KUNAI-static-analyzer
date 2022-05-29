#include "ir_grammar.hpp"

namespace KUNAI
{
    namespace MJOLNIR
    {

        IRExpr::IRExpr(expr_type_t type)
            : IRStmnt(EXPR_STMNT_T),
              type(type)
        {
        }


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

        bool IRExpr::equals(irexpr_t irexpr)
        {
            return *(this) == *(irexpr.get());
        }

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

        IRBinOp::IRBinOp(bin_op_t bin_op_type,
                         irexpr_t result,
                         irexpr_t op1,
                         irexpr_t op2)
            : IRExpr(BINOP_EXPR_T),
              bin_op_type(bin_op_type),
              result(result),
              op1(op1),
              op2(op2)
        {
        }

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

        bool IRBinOp::equals(irbinop_t irbinop)
        {
            return *this == *(irbinop.get());
        }

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

        IRUnaryOp::IRUnaryOp(unary_op_t unary_op_type,
                             irexpr_t result,
                             irexpr_t op)
            : IRExpr(UNARYOP_EXPR_T),
              unary_op_type(unary_op_type),
              result(result),
              op(op),
              cast_type(NONE_CAST)
        {
        }

        IRUnaryOp::IRUnaryOp(unary_op_t unary_op_type,
                             cast_type_t cast_type,
                             irexpr_t result,
                             irexpr_t op)
            : IRExpr(UNARYOP_EXPR_T),
              unary_op_type(unary_op_type),
              result(result),
              op(op)
        {
        }


        IRUnaryOp::IRUnaryOp(unary_op_t unary_op_type,
                             cast_type_t cast_type,
                             std::string class_name,
                             irexpr_t result,
                             irexpr_t op)
            : IRExpr(UNARYOP_EXPR_T),
              unary_op_type(unary_op_type),
              result(result),
              op(op),
              cast_type(cast_type),
              class_name(class_name)
        {
        }

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
                default:
                    break;
                }
            }

            str_stream << "[Dest: " << result->to_string() << "]";
            str_stream << "[Src: " << op->to_string() << "]";

            return str_stream.str();
        }

        bool IRUnaryOp::equals(irunaryop_t irunaryop)
        {
            return *this == *(irunaryop.get());
        }

        bool operator==(IRUnaryOp &ope1, IRUnaryOp &ope2)
        {
            return (ope1.unary_op_type == ope2.unary_op_type) &&
                   (ope1.op->equals(ope2.op)) &&
                   (ope1.result->equals(ope2.result));
        }

        /**
         * IRAssign class
         */
        IRAssign::IRAssign(irexpr_t destination,
                           irexpr_t source)
            : IRExpr(ASSIGN_EXPR_T),
              destination(destination),
              source(source)
        {
        }

        std::string IRAssign::to_string()
        {
            std::stringstream str_stream;

            str_stream << "IRAssign ";
            str_stream << "[Dest: " << destination->to_string() << "]";
            str_stream << "[Src: " << source->to_string() << "]";

            return str_stream.str();
        }

        bool IRAssign::equals(irassign_t irassign)
        {
            return *this == *(irassign.get());
        }

        bool operator==(IRAssign &ope1, IRAssign &ope2)
        {
            return (ope1.destination->equals(ope2.destination)) &&
                   (ope1.source->equals(ope2.source));
        }

        /**
         * IRCall class
         */

        IRCall::IRCall(irexpr_t callee,
                       std::vector<irexpr_t> args)
            : IRExpr(CALL_EXPR_T),
              callee(callee),
              args(args),
              ret_val(nullptr),
              call_type(NONE_CALL_T)
        {
        }

        IRCall::IRCall(irexpr_t callee,
                       call_type_t call_type,
                       std::vector<irexpr_t> args)
            : IRExpr(CALL_EXPR_T),
              callee(callee),
              call_type(call_type),
              args(args),
              ret_val(nullptr)
        {
        }

        std::string IRCall::to_string()
        {
            std::stringstream str_stream;

            str_stream << "IRCall ";

            str_stream << "[Callee: " << callee->to_string() << "]";

            switch (call_type)
            {
            case INTERNAL_CALL_T:
                str_stream << "[Type: INTERNAL_CALL]";
                break;
            case EXTERNAL_CALL_T:
                str_stream << "[Type: EXTERNAL_CALL]";
                break;
            case SYSCALL_T:
                str_stream << "[Type: SYSCALL]";
                break;
            default:
                break;
            }

            for (auto arg : args)
                str_stream << "[Param: " << arg->to_string() << "]";

            if (ret_val)
                str_stream << "[Return: " << ret_val->to_string() << "]";

            return str_stream.str();
        }

        bool IRCall::equals(ircall_t ircall)
        {
            return *(this) == *(ircall.get());
        }

        bool operator==(IRCall &ope1, IRCall &ope2)
        {
            return (ope1.callee->equals(ope2.callee)) &&
                   (std::equal(ope1.args.begin(), ope1.args.end(), ope2.args.begin()));
        }

        /**
         * IRLoad class
         */
        IRLoad::IRLoad(irexpr_t destination,
                       irexpr_t source,
                       std::uint32_t size)
            : IRExpr(LOAD_EXPR_T),
              destination(destination),
              source(source),
              index(nullptr),
              size(size)
        {
        }

        IRLoad::IRLoad(irexpr_t destination,
                       irexpr_t source,
                       irexpr_t index,
                       std::uint32_t size)
            : IRExpr(LOAD_EXPR_T),
              destination(destination),
              source(source),
              index(index),
              size(size)
        {
        }

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

        bool IRLoad::equals(irload_t irload)
        {
            return *this == *(irload.get());
        }

        bool operator==(IRLoad &ope1, IRLoad &ope2)
        {
            return (ope1.destination->equals(ope2.destination)) &&
                   (ope1.source->equals(ope2.source)) &&
                   (ope1.size == ope2.size);
        }

        /**
         * IRStore class
         */

        IRStore::IRStore(irexpr_t destination,
                         irexpr_t source,
                         std::uint32_t size)
            : IRExpr(STORE_EXPR_T),
              destination(destination),
              source(source),
              index(nullptr),
              size(size)
        {
        }

        IRStore::IRStore(irexpr_t destination,
                         irexpr_t source,
                         irexpr_t index,
                         std::uint32_t size)
            : IRExpr(STORE_EXPR_T),
              destination(destination),
              source(source),
              index(index),
              size(size)
        {
        }

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

        bool IRStore::equals(irstore_t irstore)
        {
            return *(this) == *(irstore.get());
        }

        bool operator==(IRStore &ope1, IRStore &ope2)
        {
            return (ope1.destination->equals(ope2.destination)) &&
                   (ope1.source->equals(ope2.source)) &&
                   (ope1.size == ope2.size);
        }

        /**
         * IRZComp class
         */

        IRZComp::IRZComp(zero_comp_t comp,
                    irexpr_t result,
                    irexpr_t reg)
            : IRExpr(ZCOMP_EXPR_T),
              comp(comp),
              result(result),
              reg(reg)
        {
        }

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

        bool IRZComp::equals(irzcomp_t irzcomp)
        {
            return *this == *(irzcomp.get());
        }

        bool operator==(IRZComp &ope1, IRZComp &ope2)
        {
            return (ope1.comp == ope2.comp) &&
                   (ope1.reg->equals(ope2.reg));
        }

        /**
         * IRBComp class
         */

        IRBComp::IRBComp(comp_t comp,
                         irexpr_t result,
                         irexpr_t reg1,
                         irexpr_t reg2)
            : IRExpr(BCOMP_EXPR_T),
              comp(comp),
              result(result),
              reg1(reg1),
              reg2(reg2)
        {
        }

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

        bool IRBComp::equals(irbcomp_t bcomp)
        {
            return *this == *(bcomp.get());
        }

        bool operator==(IRBComp &ope1, IRBComp &ope2)
        {
            return (ope1.comp == ope2.comp) &&
                   (ope1.reg1->equals(ope2.reg1)) &&
                   (ope1.reg2->equals(ope2.reg2));
        }

        /**
         * IRNew class
         */

        IRNew::IRNew(irexpr_t result,
                     irexpr_t class_instance)
            : IRExpr(NEW_EXPR_T),
              result(result),
              class_instance(class_instance)
        {
        }

        std::string IRNew::to_string()
        {
            std::stringstream stream;

            stream << "IRNew ";

            stream << "[Destination: " << result->to_string() << "]";
            stream << "[Source: " << class_instance->to_string() << "]";

            return stream.str();
        }

        bool IRNew::equals(irnew_t new_i)
        {
            return (*this) == *(new_i.get());
        }

        bool operator==(IRNew &new1, IRNew &new2)
        {
            return (new1.result->equals(new2.result)) &&
                   (new1.class_instance->equals(new2.class_instance));
        }
    }
}