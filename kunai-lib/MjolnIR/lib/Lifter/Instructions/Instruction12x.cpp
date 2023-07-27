#include "MjolnIR/Lifter/MjolnIRLifter.hpp"
#include "Kunai/Exceptions/lifter_exception.hpp"
#include <mlir/IR/OpDefinition.h>

using namespace KUNAI::MjolnIR;

void Lifter::gen_instruction(KUNAI::DEX::Instruction12x *instr)
{
    auto op_code = instr->get_instruction_opcode();

    auto location = mlir::FileLineColLoc::get(&context, module_name, instr->get_address(), 0);

    auto dest = instr->get_destination();
    auto src = instr->get_source();

    mlir::Type dest_type = nullptr;

    switch (op_code)
    {
    case KUNAI::DEX::TYPES::OP_MOVE:
    case KUNAI::DEX::TYPES::OP_MOVE_WIDE:
    case KUNAI::DEX::TYPES::OP_MOVE_OBJECT:
    {
        auto src_value = readLocalVariable(analysis_context.current_basic_block, analysis_context.current_method->get_basic_blocks(), src);

        auto gen_value = builder.create<::mlir::KUNAI::MjolnIR::MoveOp>(
            location,
            src_value.getType(),
            src_value);

        writeLocalVariable(analysis_context.current_basic_block, dest, gen_value);
    }
    break;
    case KUNAI::DEX::TYPES::OP_ADD_INT_2ADDR:
        if (!dest_type)
            dest_type = intType;
    case KUNAI::DEX::TYPES::OP_ADD_LONG_2ADDR:
        if (!dest_type)
            dest_type = longType;
    case KUNAI::DEX::TYPES::OP_ADD_FLOAT_2ADDR:
        if (!dest_type)
            dest_type = floatType;
    case KUNAI::DEX::TYPES::OP_ADD_DOUBLE_2ADDR:
        if (!dest_type)
            dest_type = doubleType;
        {
            auto src1_value = readLocalVariable(analysis_context.current_basic_block, analysis_context.current_method->get_basic_blocks(), dest);
            auto src2_value = readLocalVariable(analysis_context.current_basic_block, analysis_context.current_method->get_basic_blocks(), src);
            mlir::Value generated_value;

            if (llvm::isa<mlir::IntegerType>(dest_type))
            {
                generated_value = builder.create<::mlir::arith::AddIOp>(
                    location,
                    dest_type,
                    src1_value,
                    src2_value);
            }
            else if (llvm::isa<mlir::FloatType>(dest_type))
            {
                generated_value = builder.create<::mlir::arith::AddFOp>(
                    location,
                    dest_type,
                    src1_value,
                    src2_value);
            }

            writeLocalVariable(analysis_context.current_basic_block, dest, generated_value);
        }
        break;

    case KUNAI::DEX::TYPES::OP_SUB_INT_2ADDR:
        if (!dest_type)
            dest_type = intType;
    case KUNAI::DEX::TYPES::OP_SUB_LONG_2ADDR:
        if (!dest_type)
            dest_type = longType;
    case KUNAI::DEX::TYPES::OP_SUB_FLOAT_2ADDR:
        if (!dest_type)
            dest_type = floatType;
    case KUNAI::DEX::TYPES::OP_SUB_DOUBLE_2ADDR:
        if (!dest_type)
            dest_type = doubleType;
        {
            auto src1_value = readLocalVariable(analysis_context.current_basic_block, analysis_context.current_method->get_basic_blocks(), dest);
            auto src2_value = readLocalVariable(analysis_context.current_basic_block, analysis_context.current_method->get_basic_blocks(), src);
            mlir::Value generated_value;

            if (llvm::isa<mlir::IntegerType>(dest_type))
            {
                generated_value = builder.create<::mlir::arith::SubIOp>(
                    location,
                    dest_type,
                    src1_value,
                    src2_value);
            }
            else if (llvm::isa<mlir::FloatType>(dest_type))
            {
                generated_value = builder.create<::mlir::arith::SubFOp>(
                    location,
                    dest_type,
                    src1_value,
                    src2_value);
            }

            writeLocalVariable(analysis_context.current_basic_block, dest, generated_value);
        }
        break;

    case KUNAI::DEX::TYPES::OP_MUL_INT_2ADDR:
        if (!dest_type)
            dest_type = intType;
    case KUNAI::DEX::TYPES::OP_MUL_LONG_2ADDR:
        if (!dest_type)
            dest_type = longType;
    case KUNAI::DEX::TYPES::OP_MUL_FLOAT_2ADDR:
        if (!dest_type)
            dest_type = floatType;
    case KUNAI::DEX::TYPES::OP_MUL_DOUBLE_2ADDR:
        if (!dest_type)
            dest_type = doubleType;
        {
            auto src1_value = readLocalVariable(analysis_context.current_basic_block, analysis_context.current_method->get_basic_blocks(), dest);
            auto src2_value = readLocalVariable(analysis_context.current_basic_block, analysis_context.current_method->get_basic_blocks(), src);
            ::mlir::Value generated_value;

            if (llvm::isa<::mlir::IntegerType>(dest_type))
            {
                generated_value = builder.create<::mlir::arith::MulIOp>(
                    location,
                    dest_type,
                    src1_value,
                    src2_value);
            }
            else if (llvm::isa<::mlir::FloatType>(dest_type))
            {
                generated_value = builder.create<::mlir::arith::MulFOp>(
                    location,
                    dest_type,
                    src1_value,
                    src2_value);
            }

            writeLocalVariable(analysis_context.current_basic_block, dest, generated_value);
        }
        break;

    case KUNAI::DEX::TYPES::OP_DIV_INT_2ADDR:
        if (!dest_type)
            dest_type = intType;
    case KUNAI::DEX::TYPES::OP_DIV_LONG_2ADDR:
        if (!dest_type)
            dest_type = longType;
    case KUNAI::DEX::TYPES::OP_DIV_FLOAT_2ADDR:
        if (!dest_type)
            dest_type = floatType;
    case KUNAI::DEX::TYPES::OP_DIV_DOUBLE_2ADDR:
        if (!dest_type)
            dest_type = doubleType;
        {
            auto src1_value = readLocalVariable(analysis_context.current_basic_block, analysis_context.current_method->get_basic_blocks(), dest);
            auto src2_value = readLocalVariable(analysis_context.current_basic_block, analysis_context.current_method->get_basic_blocks(), src);
            ::mlir::Value generated_value;

            if (llvm::isa<::mlir::IntegerType>(dest_type))
            {
                generated_value = builder.create<::mlir::arith::DivSIOp>(
                    location,
                    dest_type,
                    src1_value,
                    src2_value);
            }
            else if (llvm::isa<::mlir::FloatType>(dest_type))
            {
                generated_value = builder.create<::mlir::arith::DivFOp>(
                    location,
                    dest_type,
                    src1_value,
                    src2_value);
            }

            writeLocalVariable(analysis_context.current_basic_block, dest, generated_value);
        }
        break;

    case KUNAI::DEX::TYPES::OP_REM_INT_2ADDR:
        if (!dest_type)
            dest_type = intType;
    case KUNAI::DEX::TYPES::OP_REM_LONG_2ADDR:
        if (!dest_type)
            dest_type = longType;
    case KUNAI::DEX::TYPES::OP_REM_FLOAT_2ADDR:
        if (!dest_type)
            dest_type = floatType;
    case KUNAI::DEX::TYPES::OP_REM_DOUBLE_2ADDR:
        if (!dest_type)
            dest_type = doubleType;
        {
            auto src1_value = readLocalVariable(analysis_context.current_basic_block, analysis_context.current_method->get_basic_blocks(), dest);
            auto src2_value = readLocalVariable(analysis_context.current_basic_block, analysis_context.current_method->get_basic_blocks(), src);
            ::mlir::Value generated_value;

            if (llvm::isa<::mlir::IntegerType>(dest_type))
            {
                generated_value = builder.create<::mlir::arith::RemSIOp>(
                    location,
                    dest_type,
                    src1_value,
                    src2_value);
            }
            else if (llvm::isa<::mlir::FloatType>(dest_type))
            {
                generated_value = builder.create<::mlir::arith::RemFOp>(
                    location,
                    dest_type,
                    src1_value,
                    src2_value);
            }

            writeLocalVariable(analysis_context.current_basic_block, dest, generated_value);
        }
        break;

    case KUNAI::DEX::TYPES::OP_AND_INT_2ADDR:
        if (!dest_type)
            dest_type = intType;
    case KUNAI::DEX::TYPES::OP_AND_LONG_2ADDR:
        if (!dest_type)
            dest_type = longType;
        {
            auto src1_value = readLocalVariable(analysis_context.current_basic_block, analysis_context.current_method->get_basic_blocks(), dest);
            auto src2_value = readLocalVariable(analysis_context.current_basic_block, analysis_context.current_method->get_basic_blocks(), src);

            auto generated_value = builder.create<::mlir::arith::AndIOp>(
                location,
                dest_type,
                src1_value,
                src2_value);

            writeLocalVariable(analysis_context.current_basic_block, dest, generated_value);
        }
        break;

    case KUNAI::DEX::TYPES::OP_OR_INT_2ADDR:
        if (!dest_type)
            dest_type = intType;
    case KUNAI::DEX::TYPES::OP_OR_LONG_2ADDR:
        if (!dest_type)
            dest_type = longType;
        {
            auto src1_value = readLocalVariable(analysis_context.current_basic_block, analysis_context.current_method->get_basic_blocks(), dest);
            auto src2_value = readLocalVariable(analysis_context.current_basic_block, analysis_context.current_method->get_basic_blocks(), src);

            auto generated_value = builder.create<::mlir::arith::OrIOp>(
                location,
                dest_type,
                src1_value,
                src2_value);

            writeLocalVariable(analysis_context.current_basic_block, dest, generated_value);
        }
        break;

    case KUNAI::DEX::TYPES::OP_XOR_INT_2ADDR:
        if (!dest_type)
            dest_type = intType;
    case KUNAI::DEX::TYPES::OP_XOR_LONG_2ADDR:
        if (!dest_type)
            dest_type = longType;
        {
            auto src1_value = readLocalVariable(analysis_context.current_basic_block, analysis_context.current_method->get_basic_blocks(), dest);
            auto src2_value = readLocalVariable(analysis_context.current_basic_block, analysis_context.current_method->get_basic_blocks(), src);

            auto generated_value = builder.create<::mlir::arith::XOrIOp>(
                location,
                dest_type,
                src1_value,
                src2_value);

            writeLocalVariable(analysis_context.current_basic_block, dest, generated_value);
        }
        break;

    case KUNAI::DEX::TYPES::OP_SHL_INT_2ADDR:
        if (!dest_type)
            dest_type = intType;
    case KUNAI::DEX::TYPES::OP_SHL_LONG_2ADDR:
        if (!dest_type)
            dest_type = longType;
        {
            auto src1_value = readLocalVariable(analysis_context.current_basic_block, analysis_context.current_method->get_basic_blocks(), dest);
            auto src2_value = readLocalVariable(analysis_context.current_basic_block, analysis_context.current_method->get_basic_blocks(), src);

            auto generated_value = builder.create<::mlir::arith::ShLIOp>(
                location,
                dest_type,
                src1_value,
                src2_value);

            writeLocalVariable(analysis_context.current_basic_block, dest, generated_value);
        }
        break;

    case KUNAI::DEX::TYPES::OP_SHR_INT_2ADDR:
        if (!dest_type)
            dest_type = intType;
    case KUNAI::DEX::TYPES::OP_SHR_LONG_2ADDR:
        if (!dest_type)
            dest_type = longType;
        {
            auto src1_value = readLocalVariable(analysis_context.current_basic_block, analysis_context.current_method->get_basic_blocks(), dest);
            auto src2_value = readLocalVariable(analysis_context.current_basic_block, analysis_context.current_method->get_basic_blocks(), src);

            auto generated_value = builder.create<::mlir::arith::ShRSIOp>(
                location,
                dest_type,
                src1_value,
                src2_value);

            writeLocalVariable(analysis_context.current_basic_block, dest, generated_value);
        }
        break;

    case KUNAI::DEX::TYPES::OP_USHR_INT_2ADDR:
        if (!dest_type)
            dest_type = intType;
    case KUNAI::DEX::TYPES::OP_USHR_LONG_2ADDR:
        if (!dest_type)
            dest_type = longType;
        {
            auto src1_value = readLocalVariable(analysis_context.current_basic_block, analysis_context.current_method->get_basic_blocks(), dest);
            auto src2_value = readLocalVariable(analysis_context.current_basic_block, analysis_context.current_method->get_basic_blocks(), src);

            auto generated_value = builder.create<::mlir::arith::ShRUIOp>(
                location,
                dest_type,
                src1_value,
                src2_value);

            writeLocalVariable(analysis_context.current_basic_block, dest, generated_value);
        }
        break;

    case KUNAI::DEX::TYPES::OP_NEG_INT:
        if (!dest_type)
            dest_type = intType;
    case KUNAI::DEX::TYPES::OP_NEG_LONG:
        if (!dest_type)
            dest_type = longType;
    case KUNAI::DEX::TYPES::OP_NEG_FLOAT:
        if (!dest_type)
            dest_type = floatType;
    case KUNAI::DEX::TYPES::OP_NEG_DOUBLE:
        if (!dest_type)
            dest_type = doubleType;
        {
            auto src_value = readLocalVariable(analysis_context.current_basic_block, analysis_context.current_method->get_basic_blocks(), src);

            auto generated_value = builder.create<::mlir::KUNAI::MjolnIR::Neg>(
                location,
                dest_type,
                src_value);

            writeLocalVariable(analysis_context.current_basic_block, dest, generated_value);
        }
        break;
    case KUNAI::DEX::TYPES::OP_NOT_INT:
        if (!dest_type)
            dest_type = intType;
    case KUNAI::DEX::TYPES::OP_NOT_LONG:
        if (!dest_type)
            dest_type = longType;
        {
            auto src_value = readLocalVariable(analysis_context.current_basic_block, analysis_context.current_method->get_basic_blocks(), src);

            auto generated_value = builder.create<::mlir::KUNAI::MjolnIR::Not>(
                location,
                dest_type,
                src_value);

            writeLocalVariable(analysis_context.current_basic_block, dest, generated_value);
        }
        break;
    /// casts
    case KUNAI::DEX::TYPES::OP_INT_TO_LONG:
    case KUNAI::DEX::TYPES::OP_FLOAT_TO_LONG:
    case KUNAI::DEX::TYPES::OP_DOUBLE_TO_LONG:
        if (!dest_type)
            dest_type = longType;
    case KUNAI::DEX::TYPES::OP_INT_TO_FLOAT:
    case KUNAI::DEX::TYPES::OP_LONG_TO_FLOAT:
    case KUNAI::DEX::TYPES::OP_DOUBLE_TO_FLOAT:
        if (!dest_type)
            dest_type = floatType;
    case KUNAI::DEX::TYPES::OP_INT_TO_DOUBLE:
    case KUNAI::DEX::TYPES::OP_LONG_TO_DOUBLE:
    case KUNAI::DEX::TYPES::OP_FLOAT_TO_DOUBLE:
        if (!dest_type)
            dest_type = doubleType;
    case KUNAI::DEX::TYPES::OP_LONG_TO_INT:
    case KUNAI::DEX::TYPES::OP_FLOAT_TO_INT:
    case KUNAI::DEX::TYPES::OP_DOUBLE_TO_INT:
        if (!dest_type)
            dest_type = intType;
    case KUNAI::DEX::TYPES::OP_INT_TO_BYTE:
        if (!dest_type)
            dest_type = byteType;
    case KUNAI::DEX::TYPES::OP_INT_TO_CHAR:
        if (!dest_type)
            dest_type = charType;
    case KUNAI::DEX::TYPES::OP_INT_TO_SHORT:
        if (!dest_type)
            dest_type = shortType;
        {
            auto src_value = readLocalVariable(analysis_context.current_basic_block, analysis_context.current_method->get_basic_blocks(), src);

            auto generated_value = builder.create<::mlir::KUNAI::MjolnIR::CastOp>(
                location,
                dest_type,
                src_value);

            writeLocalVariable(analysis_context.current_basic_block, dest, generated_value);
        }
        break;
    default:
        throw exceptions::LifterException("MjolnIRLifter::gen_instruction: Instruction12x not supported");
    }
}
