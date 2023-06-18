#include "MjolnIR/Lifter/MjolnIRLifter.hpp"
#include "Kunai/Exceptions/lifter_exception.hpp"
#include <mlir/IR/OpDefinition.h>

using namespace KUNAI::MjolnIR;

void Lifter::gen_instruction(KUNAI::DEX::Instruction22c *instr)
{
    auto op_code = instr->get_instruction_opcode();

    auto location = mlir::FileLineColLoc::get(&context, module_name, instr->get_address(), 0);

    auto reg = instr->get_destination();

    mlir::Type destination_type;

    switch (op_code)
    {
    case KUNAI::DEX::TYPES::OP_IGET:
    case KUNAI::DEX::TYPES::OP_IGET_WIDE:
    case KUNAI::DEX::TYPES::OP_IGET_OBJECT:
    case KUNAI::DEX::TYPES::OP_IGET_BOOLEAN:
    case KUNAI::DEX::TYPES::OP_IGET_BYTE:
    case KUNAI::DEX::TYPES::OP_IGET_CHAR:
    case KUNAI::DEX::TYPES::OP_IGET_SHORT:
    {
        auto field = instr->get_checked_field();
        auto field_ref = instr->get_checked_id();

        std::string &field_name = field->get_name();
        std::string &field_class = field->get_class()->get_raw();

        if (!destination_type)
            destination_type = get_type(field->get_type());

        auto generated_value = builder.create<::mlir::KUNAI::MjolnIR::LoadFieldOp>(
            location,
            destination_type,
            field_name,
            field_class,
            field_ref);

        writeLocalVariable(current_basic_block, reg, generated_value);
    }
    break;
    case KUNAI::DEX::TYPES::OP_IPUT:
    case KUNAI::DEX::TYPES::OP_IPUT_WIDE:
    case KUNAI::DEX::TYPES::OP_IPUT_OBJECT:
    case KUNAI::DEX::TYPES::OP_IPUT_BOOLEAN:
    case KUNAI::DEX::TYPES::OP_IPUT_BYTE:
    case KUNAI::DEX::TYPES::OP_IPUT_CHAR:
    case KUNAI::DEX::TYPES::OP_IPUT_SHORT:
    {
        auto field = instr->get_checked_field();
        auto field_ref = instr->get_checked_id();

        std::string &field_name = field->get_name();
        std::string &field_class = field->get_class()->get_raw();

        auto reg_value = readLocalVariable(current_basic_block, current_method->get_basic_blocks(), reg);

        builder.create<::mlir::KUNAI::MjolnIR::StoreFieldOp>(
            location,
            reg_value,
            field_name,
            field_class,
            field_ref);
    }
    break;
    case KUNAI::DEX::TYPES::OP_NEW_ARRAY:
    {
        auto array_type = instr->get_checked_dvmtype();

        assert(array_type && "type of the array cannot be null");

        auto array = get_array(array_type);

        auto size = readLocalVariable(current_basic_block, current_method->get_basic_blocks(), instr->get_operand());

        auto gen_value = builder.create<::mlir::KUNAI::MjolnIR::NewArrayOp>(
            location,
            array,
            size
        );

        writeLocalVariable(current_basic_block, reg, gen_value);
    }
    break;
    default:
        throw exceptions::LifterException("MjolnIRLifter::gen_instruction: Instruction22c not implemented yet");
        break;
    }
}
