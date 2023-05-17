#include "Lifter/MjolnIRLifter.hpp"
#include "Kunai/Exceptions/lifter_exception.hpp"
#include <mlir/IR/OpDefinition.h>

using namespace KUNAI::MjolnIR;

void Lifter::gen_instruction(KUNAI::DEX::Instruction21c *instr)
{
    auto op_code = instr->get_instruction_opcode();

    auto location = mlir::FileLineColLoc::get(&context, module_name, instr->get_address(), 0);

    auto dest = instr->get_destination();

    switch (op_code)
    {
    case KUNAI::DEX::TYPES::OP_NEW_INSTANCE:
    {
        auto cls = instr->get_source_dvmclass();

        auto cls_type = get_type(cls);

        auto gen_value = builder.create<::mlir::KUNAI::MjolnIR::NewOp>(
            location,
            cls_type);

        writeLocalVariable(current_basic_block, dest, gen_value);
    }
    break;
    case KUNAI::DEX::TYPES::OP_CONST_STRING:
    {
        auto str_value = instr->get_source_str();
        auto str_ref = instr->get_source();

        auto gen_value = builder.create<::mlir::KUNAI::MjolnIR::LoadString>(
            location,
            strObjectType,
            str_value,
            str_ref);

        writeLocalVariable(current_basic_block, dest, gen_value);
    }
    break;
    case KUNAI::DEX::TYPES::OP_SGET:
    case KUNAI::DEX::TYPES::OP_SGET_WIDE:
    case KUNAI::DEX::TYPES::OP_SGET_OBJECT:
    case KUNAI::DEX::TYPES::OP_SGET_BOOLEAN:
    case KUNAI::DEX::TYPES::OP_SGET_BYTE:
    case KUNAI::DEX::TYPES::OP_SGET_CHAR:
    case KUNAI::DEX::TYPES::OP_SGET_SHORT:
    {
        auto field = instr->get_source_field();
        auto field_ref = instr->get_source();

        std::string &field_name = field->get_name();
        std::string &field_class = field->get_class()->get_raw();

        auto dest_type = get_type(field->get_type());

        auto generated_value = builder.create<::mlir::KUNAI::MjolnIR::LoadFieldOp>(
            location,
            dest_type,
            field_name,
            field_class,
            field_ref);

        writeLocalVariable(current_basic_block, dest, generated_value);
    }
    break;

    default:
        throw exceptions::LifterException("Lifter::gen_instruction: Instruction21c not supported");
        break;
    }
}
