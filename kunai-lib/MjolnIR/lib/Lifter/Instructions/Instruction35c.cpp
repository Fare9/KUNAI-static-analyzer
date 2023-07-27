#include "MjolnIR/Lifter/MjolnIRLifter.hpp"
#include "Kunai/Exceptions/lifter_exception.hpp"
#include <mlir/IR/OpDefinition.h>

using namespace KUNAI::MjolnIR;

void Lifter::gen_instruction(KUNAI::DEX::Instruction35c *instr)
{
    auto op_code = instr->get_instruction_opcode();

    auto location = mlir::FileLineColLoc::get(&context, module_name, instr->get_address(), 0);

    switch (op_code)
    {
    case KUNAI::DEX::TYPES::OP_INVOKE_VIRTUAL:
    case KUNAI::DEX::TYPES::OP_INVOKE_SUPER:
    case KUNAI::DEX::TYPES::OP_INVOKE_DIRECT:
    case KUNAI::DEX::TYPES::OP_INVOKE_STATIC:
    case KUNAI::DEX::TYPES::OP_INVOKE_INTERFACE:
    {
        mlir::SmallVector<mlir::Value, 4> parameters;

        auto called_method = instr->get_method();
        auto method_ref = instr->get_type_idx();
        auto method_name = called_method->get_name();

        auto parameters_protos = called_method->get_proto()->get_parameters();
        auto invoke_parameters = instr->get_registers();

        ::mlir::Type retType = get_type(called_method->get_proto()->get_return_type());

        bool is_static = op_code == KUNAI::DEX::TYPES::OP_INVOKE_STATIC ? true : false;

        for (size_t I = 0, P = 0, Limit = invoke_parameters.size();
             I < Limit;
             ++I)
        {
            parameters.push_back(readLocalVariable(analysis_context.current_basic_block,
                                                   analysis_context.current_method->get_basic_blocks(), invoke_parameters[I]));

            /// If the method is not static, the first
            /// register is a pointer to the object
            if (I == 0 && op_code != KUNAI::DEX::TYPES::OP_INVOKE_STATIC)
                continue;

            auto fundamental = reinterpret_cast<KUNAI::DEX::DVMFundamental *>(parameters_protos[P]);

            /// if the parameter is a long or a double, skip the second register
            if (fundamental &&
                    fundamental->get_fundamental_type() == KUNAI::DEX::DVMFundamental::LONG ||
                fundamental->get_fundamental_type() == KUNAI::DEX::DVMFundamental::DOUBLE)
                ++I; // skip next register
            /// go to next parameter
            P++;
        }

        if (mlir::isa<::mlir::KUNAI::MjolnIR::DVMVoidType>(retType))
        {
            mlir::Type NoneType;
            builder.create<::mlir::KUNAI::MjolnIR::InvokeOp>(
                location,
                NoneType,
                method_name,
                method_ref,
                is_static,
                parameters);
        }
        else
            builder.create<::mlir::KUNAI::MjolnIR::InvokeOp>(
                location,
                retType,
                method_name,
                method_ref,
                is_static,
                parameters);
    }
    /* code */
    break;

    default:
        throw exceptions::LifterException("Lifter::gen_instruction: Instruction35c not supported");
        break;
    }
}
