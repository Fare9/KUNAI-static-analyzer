#include "KUNAI/mjolnIR/Lifters/lifter_android.hpp"

namespace KUNAI
{
    namespace LIFTER
    {

        std::map<std::uint32_t, MJOLNIR::irreg_t> created_registers;

        LifterAndroid::LifterAndroid() : temp_reg_id(0),
                                         current_idx(0)
        {
            optimizer = std::make_shared<MJOLNIR::Optimizer>();
        }

        MJOLNIR::irgraph_t LifterAndroid::lift_android_method(DEX::methodanalysis_t &method_analysis, DEX::analysis_t &android_analysis)
        {
            auto bbs = method_analysis->get_basic_blocks()->get_basic_blocks();
            size_t n_bbs = bbs.size();
            // set android_analysis
            this->android_analysis = android_analysis;
            // graph returnedd by
            MJOLNIR::irgraph_t method_graph = std::make_shared<MJOLNIR::IRGraph>();

            // first of all lift all the blocks
            for (auto bb : bbs)
            {
                MJOLNIR::irblock_t lifted_bb = std::make_shared<MJOLNIR::IRBlock>();

                this->lift_android_basic_block(bb, lifted_bb);

                if (lifted_bb->get_number_of_statements() == 0)
                    continue;

                lifted_blocks[bb.get()] = lifted_bb;

                method_graph->add_node(lifted_bb);
            }

            // Create Control Flow Graph using the children nodes
            // from the method blocks.
            for (auto bb : bbs)
            {
                auto next_bbs = bb->get_next();

                auto current_bb = lifted_blocks[bb.get()];

                for (auto next_bb : next_bbs)
                {
                    auto block = std::get<2>(next_bb);

                    auto last_instr = lifted_blocks[block]->get_statements().back();

                    // unsigned jumps are fixed later, they only have to point
                    // to where jump targets
                    if (last_instr->get_op_type() != MJOLNIR::IRStmnt::UJMP_OP_T)
                        method_graph->add_edge(current_bb, lifted_blocks[block]);
                }
            }

            this->jump_target_analysis(bbs, method_graph);
            optimizer->fallthrough_target_analysis(method_graph);


            method_graph->set_last_temporal(temp_reg_id - 1);
            // clean android_analysis
            this->android_analysis = nullptr;

            return method_graph;
        }

        bool LifterAndroid::lift_android_basic_block(DEX::dvmbasicblock_t &basic_block, MJOLNIR::irblock_t &bb)
        {
            auto instructions = basic_block->get_instructions();
            auto next = basic_block->get_next();

            for (auto instruction : instructions)
            {
                auto op_code = static_cast<DEX::DVMTypes::Opcode>(instruction->get_OP());

                switch (op_code)
                {
                case DEX::DVMTypes::Opcode::OP_MOVE_RESULT:
                case DEX::DVMTypes::Opcode::OP_MOVE_RESULT_WIDE:
                case DEX::DVMTypes::Opcode::OP_MOVE_RESULT_OBJECT:
                {
                    if (bb->get_number_of_statements() == 0) // security check if there are or not statements
                        continue;

                    auto last_instr = bb->get_statements().back();

                    if (auto ir_call = MJOLNIR::call_ir(last_instr))
                    {
                        auto move_result = std::dynamic_pointer_cast<DEX::Instruction11x>(instruction);
                        ir_call->set_ret_val(make_android_register(move_result->get_destination()));
                    }

                    break;
                }
                default:
                    this->lift_android_instruction(instruction, bb);
                    break;
                }

                current_idx += instruction->get_length();
            }

            bb->set_start_idx(basic_block->get_start());
            bb->set_end_idx(basic_block->get_end());

            return true;
        }

        bool LifterAndroid::lift_android_instruction(DEX::instruction_t &instruction, MJOLNIR::irblock_t &bb)
        {
            auto logger = KUNAI::LOGGER::logger();

            auto op_code = static_cast<DEX::DVMTypes::Opcode>(instruction->get_OP());

            switch (op_code)
            {
            case DEX::DVMTypes::Opcode::OP_NOP:
            {
                MJOLNIR::irstmnt_t nop = std::make_shared<MJOLNIR::IRNop>();
                bb->append_statement_to_block(nop);
                break;
            }
            case DEX::DVMTypes::Opcode::OP_MOVE:
            case DEX::DVMTypes::Opcode::OP_MOVE_WIDE:
            case DEX::DVMTypes::Opcode::OP_MOVE_OBJECT:
            {
                MJOLNIR::irstmnt_t assignment_instr;

                auto instr = std::dynamic_pointer_cast<DEX::Instruction12x>(instruction);
                auto dest = instr->get_destination();
                auto src = instr->get_source();

                auto dest_reg = make_android_register(dest);
                auto src_reg = make_android_register(src);

                assignment_instr = std::make_shared<MJOLNIR::IRAssign>(dest_reg, src_reg);

                bb->append_statement_to_block(assignment_instr);

                break;
            }
            case DEX::DVMTypes::Opcode::OP_MOVE_FROM16:
            case DEX::DVMTypes::Opcode::OP_MOVE_WIDE_FROM16:
            case DEX::DVMTypes::Opcode::OP_MOVE_OBJECT_FROM16:
            {
                MJOLNIR::irstmnt_t assignment_instr;

                auto instr = std::dynamic_pointer_cast<DEX::Instruction22x>(instruction);
                auto dest = instr->get_destination();
                auto src = instr->get_source();

                auto dest_reg = make_android_register(dest);
                auto src_reg = make_android_register(src);

                assignment_instr = std::make_shared<MJOLNIR::IRAssign>(dest_reg, src_reg);

                bb->append_statement_to_block(assignment_instr);

                break;
            }
            case DEX::DVMTypes::Opcode::OP_MOVE_16:
            case DEX::DVMTypes::Opcode::OP_MOVE_WIDE_16:
            case DEX::DVMTypes::Opcode::OP_MOVE_OBJECT_16:
            {
                MJOLNIR::irstmnt_t assignment_instr;

                auto instr = std::dynamic_pointer_cast<DEX::Instruction32x>(instruction);

                auto dest = instr->get_destination();
                auto src = instr->get_source();

                auto dest_reg = make_android_register(dest);
                auto src_reg = make_android_register(src);

                assignment_instr = std::make_shared<MJOLNIR::IRAssign>(dest_reg, src_reg);

                bb->append_statement_to_block(assignment_instr);

                break;
            }
            case DEX::DVMTypes::Opcode::OP_CONST_4:
            {
                MJOLNIR::irstmnt_t assignment_instr;

                auto instr = std::dynamic_pointer_cast<DEX::Instruction11n>(instruction);

                auto dest = instr->get_destination();
                auto src = instr->get_source();

                auto dest_reg = make_android_register(dest);
                auto src_int = make_int(src, true, NIBBLE_S);

                assignment_instr = std::make_shared<MJOLNIR::IRAssign>(dest_reg, src_int);

                bb->append_statement_to_block(assignment_instr);

                break;
            }
            case DEX::DVMTypes::Opcode::OP_CONST_16:
            case DEX::DVMTypes::Opcode::OP_CONST_WIDE_16:
            {
                MJOLNIR::irstmnt_t assignment_instr;

                auto instr = std::dynamic_pointer_cast<DEX::Instruction21s>(instruction);

                auto dest = instr->get_destination();
                auto src = instr->get_source();

                auto dest_reg = make_android_register(dest);
                auto src_int = make_int(src, true, WORD_S);

                assignment_instr = std::make_shared<MJOLNIR::IRAssign>(dest_reg, src_int);

                bb->append_statement_to_block(assignment_instr);

                break;
            }
            case DEX::DVMTypes::Opcode::OP_CONST:
            case DEX::DVMTypes::Opcode::OP_CONST_WIDE_32:
            {
                MJOLNIR::irstmnt_t assignment_instr;

                auto instr = std::dynamic_pointer_cast<DEX::Instruction31i>(instruction);

                auto dest = instr->get_destination();
                auto src = instr->get_source();

                auto dest_reg = make_android_register(dest);
                auto src_int = make_int(src, true, DWORD_S);

                assignment_instr = std::make_shared<MJOLNIR::IRAssign>(dest_reg, src_int);

                bb->append_statement_to_block(assignment_instr);

                break;
            }
            case DEX::DVMTypes::Opcode::OP_CONST_HIGH16:
            case DEX::DVMTypes::Opcode::OP_CONST_WIDE_HIGH16:
            {
                MJOLNIR::irstmnt_t assignment_instr;

                auto instr = std::dynamic_pointer_cast<DEX::Instruction21h>(instruction);

                auto dest = instr->get_destination();
                auto src = instr->get_source();

                auto dest_reg = make_android_register(dest);
                std::shared_ptr<KUNAI::MJOLNIR::IRConstInt> src_int = nullptr;

                if (op_code == DEX::DVMTypes::Opcode::OP_CONST_HIGH16)
                    src_int = make_int(src << 16, true, QWORD_S);
                else if (op_code == DEX::DVMTypes::Opcode::OP_CONST_WIDE_HIGH16)
                    src_int = make_int(src << 48, true, QWORD_S);

                assignment_instr = std::make_shared<MJOLNIR::IRAssign>(dest_reg, src_int);

                bb->append_statement_to_block(assignment_instr);

                break;
            }
            case DEX::DVMTypes::Opcode::OP_CONST_WIDE:
            {
                MJOLNIR::irstmnt_t assignment_instr;

                auto instr = std::dynamic_pointer_cast<DEX::Instruction51l>(instruction);

                auto dest = instr->get_destination();
                auto src = instr->get_source();

                auto dest_reg = make_android_register(dest);
                auto src_int = make_int(src, true, QWORD_S);

                assignment_instr = std::make_shared<MJOLNIR::IRAssign>(dest_reg, src_int);

                bb->append_statement_to_block(assignment_instr);

                break;
            }
            case DEX::DVMTypes::Opcode::OP_CONST_STRING:
            {
                MJOLNIR::irstmnt_t assignment_instr;

                auto instr = std::dynamic_pointer_cast<DEX::Instruction21c>(instruction);

                auto dest = instr->get_destination();
                auto dest_reg = make_android_register(dest);

                auto src = make_str(*instr->get_source_str());

                assignment_instr = std::make_shared<MJOLNIR::IRAssign>(dest_reg, src);

                bb->append_statement_to_block(assignment_instr);

                break;
            }
            case DEX::DVMTypes::Opcode::OP_CONST_CLASS:
            {
                MJOLNIR::irstmnt_t assignment_instr;

                auto instr = std::dynamic_pointer_cast<DEX::Instruction21c>(instruction);

                auto dest = instr->get_destination();
                auto dest_reg = make_android_register(dest);

                if (instr->get_source_typeid()->get_type() != DEX::Type::CLASS)
                {
                    // ToDo generate an exception
                    return false;
                }
                auto src_class = std::dynamic_pointer_cast<DEX::Class>(instr->get_source_typeid());
                auto src = make_class(src_class);

                assignment_instr = std::make_shared<MJOLNIR::IRAssign>(dest_reg, src);

                bb->append_statement_to_block(assignment_instr);

                break;
            }
            case DEX::DVMTypes::Opcode::OP_SGET:
            case DEX::DVMTypes::Opcode::OP_SGET_WIDE:
            case DEX::DVMTypes::Opcode::OP_SGET_OBJECT:
            case DEX::DVMTypes::Opcode::OP_SGET_BOOLEAN:
            case DEX::DVMTypes::Opcode::OP_SGET_BYTE:
            case DEX::DVMTypes::Opcode::OP_SGET_CHAR:
            case DEX::DVMTypes::Opcode::OP_SGET_SHORT:
            {
                MJOLNIR::irstmnt_t assignment_instr;
                MJOLNIR::irunaryop_t cast_instr = nullptr;

                auto instr = std::dynamic_pointer_cast<DEX::Instruction21c>(instruction);

                auto dest = instr->get_destination();
                auto dest_reg = make_android_register(dest);

                // check the type of the source operand
                switch (instr->get_source_kind())
                {
                case DEX::DVMTypes::METH:
                {
                    logger->error("Not implemented DEX::DVMTypes::METH yet");
                    auto src = make_none_type();
                    assignment_instr = std::make_shared<MJOLNIR::IRAssign>(dest_reg, src);
                    break;
                }
                case DEX::DVMTypes::STRING:
                {
                    auto src = make_str(*instr->get_source_str());
                    assignment_instr = std::make_shared<MJOLNIR::IRAssign>(dest_reg, src);
                    break;
                }
                case DEX::DVMTypes::TYPE:
                {
                    logger->error("Not implemented DEX::DVMTypes::TYPE yet");
                    auto src = make_none_type();
                    assignment_instr = std::make_shared<MJOLNIR::IRAssign>(dest_reg, src);
                    break;
                }
                case DEX::DVMTypes::FIELD:
                {
                    auto src = make_field(instr->get_source_static_field());
                    assignment_instr = std::make_shared<MJOLNIR::IRAssign>(dest_reg, src);

                    switch (src->get_type())
                    {
                    case MJOLNIR::IRField::BOOLEAN_F:
                        cast_instr = std::make_shared<MJOLNIR::IRUnaryOp>(MJOLNIR::IRUnaryOp::CAST_OP_T, MJOLNIR::IRUnaryOp::TO_BOOLEAN, dest_reg, dest_reg);
                        break;
                    case MJOLNIR::IRField::BYTE_F:
                        cast_instr = std::make_shared<MJOLNIR::IRUnaryOp>(MJOLNIR::IRUnaryOp::CAST_OP_T, MJOLNIR::IRUnaryOp::TO_BYTE, dest_reg, dest_reg);
                        break;
                    case MJOLNIR::IRField::CHAR_F:
                        cast_instr = std::make_shared<MJOLNIR::IRUnaryOp>(MJOLNIR::IRUnaryOp::CAST_OP_T, MJOLNIR::IRUnaryOp::TO_CHAR, dest_reg, dest_reg);
                        break;
                    case MJOLNIR::IRField::SHORT_F:
                        cast_instr = std::make_shared<MJOLNIR::IRUnaryOp>(MJOLNIR::IRUnaryOp::CAST_OP_T, MJOLNIR::IRUnaryOp::TO_SHORT, dest_reg, dest_reg);
                        break;
                    case MJOLNIR::IRField::LONG_F:
                        cast_instr = std::make_shared<MJOLNIR::IRUnaryOp>(MJOLNIR::IRUnaryOp::CAST_OP_T, MJOLNIR::IRUnaryOp::TO_LONG, dest_reg, dest_reg);
                        break;
                    case MJOLNIR::IRField::INT_F:
                        cast_instr = std::make_shared<MJOLNIR::IRUnaryOp>(MJOLNIR::IRUnaryOp::CAST_OP_T, MJOLNIR::IRUnaryOp::TO_INT, dest_reg, dest_reg);
                        break;
                    case MJOLNIR::IRField::FLOAT_F:
                        cast_instr = std::make_shared<MJOLNIR::IRUnaryOp>(MJOLNIR::IRUnaryOp::CAST_OP_T, MJOLNIR::IRUnaryOp::TO_FLOAT, dest_reg, dest_reg);
                        break;
                    case MJOLNIR::IRField::DOUBLE_F:
                        cast_instr = std::make_shared<MJOLNIR::IRUnaryOp>(MJOLNIR::IRUnaryOp::CAST_OP_T, MJOLNIR::IRUnaryOp::TO_DOUBLE, dest_reg, dest_reg);
                        break;
                    case MJOLNIR::IRField::CLASS_F:
                        cast_instr = std::make_shared<MJOLNIR::IRUnaryOp>(MJOLNIR::IRUnaryOp::CAST_OP_T, MJOLNIR::IRUnaryOp::TO_ADDR, dest_reg, dest_reg);
                        break;
                    case MJOLNIR::IRField::VOID_F:
                        cast_instr = std::make_shared<MJOLNIR::IRUnaryOp>(MJOLNIR::IRUnaryOp::CAST_OP_T, MJOLNIR::IRUnaryOp::TO_VOID, dest_reg, dest_reg);
                        break;
                    case MJOLNIR::IRField::ARRAY_F:
                        cast_instr = std::make_shared<MJOLNIR::IRUnaryOp>(MJOLNIR::IRUnaryOp::CAST_OP_T, MJOLNIR::IRUnaryOp::TO_ARRAY, dest_reg, dest_reg);
                        break;
                    default:
                        throw exceptions::LifterException("lift_android_instruction: case DEX::DVMTypes::FIELD src->get_type() not implemented.");
                    } // src->get_type()

                    break;
                }
                case DEX::DVMTypes::PROTO:
                {
                    logger->error("Not implemented DEX::DVMTypes::PROTO yet");
                    auto src = make_none_type();
                    assignment_instr = std::make_shared<MJOLNIR::IRAssign>(dest_reg, src);
                    break;
                }
                default:
                    throw exceptions::LifterException("lift_android_instruction: instr->get_source_kind() value not implemented");
                }

                bb->append_statement_to_block(assignment_instr);

                if (cast_instr != nullptr)
                    bb->append_statement_to_block(cast_instr);

                break;
            }
            case DEX::DVMTypes::Opcode::OP_SPUT:
            case DEX::DVMTypes::Opcode::OP_SPUT_WIDE:
            case DEX::DVMTypes::Opcode::OP_SPUT_OBJECT:
            case DEX::DVMTypes::Opcode::OP_SPUT_BOOLEAN:
            case DEX::DVMTypes::Opcode::OP_SPUT_BYTE:
            case DEX::DVMTypes::Opcode::OP_SPUT_CHAR:
            case DEX::DVMTypes::Opcode::OP_SPUT_SHORT:
            {
                MJOLNIR::irstmnt_t assignment_instr;

                auto instr = std::dynamic_pointer_cast<DEX::Instruction21c>(instruction);

                // Instruction PUT follows the same instruction format
                // than GET, it follows same codification, but here
                // we have: SPUT Field, Regsister
                // here what we call  "destination" is the source of the data.
                // and source is a Field destination
                auto src = instr->get_destination();
                auto src_reg = make_android_register(src);

                auto dst = make_field(instr->get_source_static_field());

                assignment_instr = std::make_shared<MJOLNIR::IRAssign>(dst, src_reg);

                bb->append_statement_to_block(assignment_instr);

                break;
            }
            case DEX::DVMTypes::Opcode::OP_CONST_STRING_JUMBO:
            {
                MJOLNIR::irstmnt_t assignment_instr;

                auto instr = std::dynamic_pointer_cast<DEX::Instruction31c>(instruction);

                auto dest = instr->get_destination();
                auto dest_reg = make_android_register(dest);

                auto src = make_str(*instr->get_source_str());
                assignment_instr = std::make_shared<MJOLNIR::IRAssign>(dest_reg, src);

                bb->append_statement_to_block(assignment_instr);

                break;
            }
            case DEX::DVMTypes::Opcode::OP_IGET:
            case DEX::DVMTypes::Opcode::OP_IGET_WIDE:
            case DEX::DVMTypes::Opcode::OP_IGET_OBJECT:
            case DEX::DVMTypes::Opcode::OP_IGET_BOOLEAN:
            case DEX::DVMTypes::Opcode::OP_IGET_BYTE:
            case DEX::DVMTypes::Opcode::OP_IGET_CHAR:
            case DEX::DVMTypes::Opcode::OP_IGET_SHORT:
            {
                MJOLNIR::irstmnt_t assignment_instr;
                MJOLNIR::irunaryop_t cast_instr = nullptr;

                auto instr = std::dynamic_pointer_cast<DEX::Instruction22c>(instruction);

                auto dest_reg = make_android_register(instr->get_first_operand());
                auto src_field = make_field(instr->get_third_operand_FieldId());

                assignment_instr = std::make_shared<MJOLNIR::IRAssign>(dest_reg, src_field);

                switch (src_field->get_type())
                {
                case MJOLNIR::IRField::BOOLEAN_F:
                    cast_instr = std::make_shared<MJOLNIR::IRUnaryOp>(MJOLNIR::IRUnaryOp::CAST_OP_T, MJOLNIR::IRUnaryOp::TO_BOOLEAN, dest_reg, dest_reg);
                    break;
                case MJOLNIR::IRField::BYTE_F:
                    cast_instr = std::make_shared<MJOLNIR::IRUnaryOp>(MJOLNIR::IRUnaryOp::CAST_OP_T, MJOLNIR::IRUnaryOp::TO_BYTE, dest_reg, dest_reg);
                    break;
                case MJOLNIR::IRField::CHAR_F:
                    cast_instr = std::make_shared<MJOLNIR::IRUnaryOp>(MJOLNIR::IRUnaryOp::CAST_OP_T, MJOLNIR::IRUnaryOp::TO_CHAR, dest_reg, dest_reg);
                    break;
                case MJOLNIR::IRField::SHORT_F:
                    cast_instr = std::make_shared<MJOLNIR::IRUnaryOp>(MJOLNIR::IRUnaryOp::CAST_OP_T, MJOLNIR::IRUnaryOp::TO_SHORT, dest_reg, dest_reg);
                    break;
                case MJOLNIR::IRField::INT_F:
                    cast_instr = std::make_shared<MJOLNIR::IRUnaryOp>(MJOLNIR::IRUnaryOp::CAST_OP_T, MJOLNIR::IRUnaryOp::TO_INT, dest_reg, dest_reg);
                    break;
                case MJOLNIR::IRField::DOUBLE_F:
                    cast_instr = std::make_shared<MJOLNIR::IRUnaryOp>(MJOLNIR::IRUnaryOp::CAST_OP_T, MJOLNIR::IRUnaryOp::TO_DOUBLE, dest_reg, dest_reg);
                    break;
                case MJOLNIR::IRField::ARRAY_F:
                case MJOLNIR::IRField::CLASS_F:
                    cast_instr = std::make_shared<MJOLNIR::IRUnaryOp>(MJOLNIR::IRUnaryOp::CAST_OP_T, MJOLNIR::IRUnaryOp::TO_ADDR, dest_reg, dest_reg);
                    break;
                case MJOLNIR::IRField::FLOAT_F:
                    cast_instr = std::make_shared<MJOLNIR::IRUnaryOp>(MJOLNIR::IRUnaryOp::CAST_OP_T, MJOLNIR::IRUnaryOp::TO_FLOAT, dest_reg, dest_reg);
                    break;
                case MJOLNIR::IRField::LONG_F:
                    cast_instr = std::make_shared<MJOLNIR::IRUnaryOp>(MJOLNIR::IRUnaryOp::CAST_OP_T, MJOLNIR::IRUnaryOp::TO_LONG, dest_reg, dest_reg);
                    break;
                default:
                    throw exceptions::LifterException("lift_android_instruction: DEX::DVMTypes::Opcode::OP_IGET_SHORT src_field->get_type() not implemented");
                } // src_field->get_type()

                bb->append_statement_to_block(assignment_instr);

                if (cast_instr != nullptr)
                    bb->append_statement_to_block(cast_instr);

                break;
            }
            case DEX::DVMTypes::Opcode::OP_IPUT:
            case DEX::DVMTypes::Opcode::OP_IPUT_WIDE:
            case DEX::DVMTypes::Opcode::OP_IPUT_OBJECT:
            case DEX::DVMTypes::Opcode::OP_IPUT_BOOLEAN:
            case DEX::DVMTypes::Opcode::OP_IPUT_BYTE:
            case DEX::DVMTypes::Opcode::OP_IPUT_CHAR:
            case DEX::DVMTypes::Opcode::OP_IPUT_SHORT:
            {
                MJOLNIR::irstmnt_t assignment_instr;

                auto instr = std::dynamic_pointer_cast<DEX::Instruction22c>(instruction);

                auto src_reg = make_android_register(instr->get_first_operand());
                auto dst_field = make_field(instr->get_third_operand_FieldId());

                assignment_instr = std::make_shared<MJOLNIR::IRAssign>(dst_field, src_reg);

                bb->append_statement_to_block(assignment_instr);

                break;
            }
            case DEX::DVMTypes::Opcode::OP_ADD_INT:
            {
                lift_instruction23x_binary_instruction(instruction, MJOLNIR::IRBinOp::ADD_OP_T, MJOLNIR::IRUnaryOp::TO_INT, bb);
                break;
            }
            case DEX::DVMTypes::Opcode::OP_ADD_LONG:
            {
                lift_instruction23x_binary_instruction(instruction, MJOLNIR::IRBinOp::ADD_OP_T, MJOLNIR::IRUnaryOp::TO_LONG, bb);
                break;
            }
            case DEX::DVMTypes::Opcode::OP_ADD_FLOAT:
            {
                lift_instruction23x_binary_instruction(instruction, MJOLNIR::IRBinOp::ADD_OP_T, MJOLNIR::IRUnaryOp::TO_FLOAT, bb);
                break;
            }
            case DEX::DVMTypes::Opcode::OP_ADD_DOUBLE:
            {
                lift_instruction23x_binary_instruction(instruction, MJOLNIR::IRBinOp::ADD_OP_T, MJOLNIR::IRUnaryOp::TO_DOUBLE, bb);
                break;
            }
            case DEX::DVMTypes::Opcode::OP_SUB_INT:
            {
                lift_instruction23x_binary_instruction(instruction, MJOLNIR::IRBinOp::SUB_OP_T, MJOLNIR::IRUnaryOp::TO_INT, bb);
                break;
            }
            case DEX::DVMTypes::Opcode::OP_SUB_LONG:
            {
                lift_instruction23x_binary_instruction(instruction, MJOLNIR::IRBinOp::SUB_OP_T, MJOLNIR::IRUnaryOp::TO_LONG, bb);
                break;
            }
            case DEX::DVMTypes::Opcode::OP_SUB_FLOAT:
            {
                lift_instruction23x_binary_instruction(instruction, MJOLNIR::IRBinOp::SUB_OP_T, MJOLNIR::IRUnaryOp::TO_FLOAT, bb);
                break;
            }
            case DEX::DVMTypes::Opcode::OP_SUB_DOUBLE:
            {
                lift_instruction23x_binary_instruction(instruction, MJOLNIR::IRBinOp::SUB_OP_T, MJOLNIR::IRUnaryOp::TO_DOUBLE, bb);
                break;
            }
            case DEX::DVMTypes::Opcode::OP_MUL_INT:
            {
                lift_instruction23x_binary_instruction(instruction, MJOLNIR::IRBinOp::S_MUL_OP_T, MJOLNIR::IRUnaryOp::TO_INT, bb);
                break;
            }
            case DEX::DVMTypes::Opcode::OP_MUL_LONG:
            {
                lift_instruction23x_binary_instruction(instruction, MJOLNIR::IRBinOp::S_MUL_OP_T, MJOLNIR::IRUnaryOp::TO_LONG, bb);
                break;
            }
            case DEX::DVMTypes::Opcode::OP_MUL_FLOAT:
            {
                lift_instruction23x_binary_instruction(instruction, MJOLNIR::IRBinOp::S_MUL_OP_T, MJOLNIR::IRUnaryOp::TO_FLOAT, bb);
                break;
            }
            case DEX::DVMTypes::Opcode::OP_MUL_DOUBLE:
            {
                lift_instruction23x_binary_instruction(instruction, MJOLNIR::IRBinOp::S_MUL_OP_T, MJOLNIR::IRUnaryOp::TO_DOUBLE, bb);
                break;
            }
            case DEX::DVMTypes::Opcode::OP_DIV_INT:
            {
                lift_instruction23x_binary_instruction(instruction, MJOLNIR::IRBinOp::S_DIV_OP_T, MJOLNIR::IRUnaryOp::TO_INT, bb);
                break;
            }
            case DEX::DVMTypes::Opcode::OP_DIV_LONG:
            {
                lift_instruction23x_binary_instruction(instruction, MJOLNIR::IRBinOp::S_DIV_OP_T, MJOLNIR::IRUnaryOp::TO_LONG, bb);
                break;
            }
            case DEX::DVMTypes::Opcode::OP_DIV_FLOAT:
            {
                lift_instruction23x_binary_instruction(instruction, MJOLNIR::IRBinOp::S_DIV_OP_T, MJOLNIR::IRUnaryOp::TO_FLOAT, bb);
                break;
            }
            case DEX::DVMTypes::Opcode::OP_DIV_DOUBLE:
            {
                lift_instruction23x_binary_instruction(instruction, MJOLNIR::IRBinOp::S_DIV_OP_T, MJOLNIR::IRUnaryOp::TO_DOUBLE, bb);
                break;
            }
            case DEX::DVMTypes::Opcode::OP_REM_INT:
            {
                lift_instruction23x_binary_instruction(instruction, MJOLNIR::IRBinOp::MOD_OP_T, MJOLNIR::IRUnaryOp::TO_INT, bb);
                break;
            }
            case DEX::DVMTypes::Opcode::OP_REM_LONG:
            {
                lift_instruction23x_binary_instruction(instruction, MJOLNIR::IRBinOp::MOD_OP_T, MJOLNIR::IRUnaryOp::TO_LONG, bb);
                break;
            }
            case DEX::DVMTypes::Opcode::OP_REM_FLOAT:
            {
                lift_instruction23x_binary_instruction(instruction, MJOLNIR::IRBinOp::MOD_OP_T, MJOLNIR::IRUnaryOp::TO_FLOAT, bb);
                break;
            }
            case DEX::DVMTypes::Opcode::OP_REM_DOUBLE:
            {
                lift_instruction23x_binary_instruction(instruction, MJOLNIR::IRBinOp::MOD_OP_T, MJOLNIR::IRUnaryOp::TO_DOUBLE, bb);
                break;
            }
            case DEX::DVMTypes::Opcode::OP_AND_INT:
            {
                lift_instruction23x_binary_instruction(instruction, MJOLNIR::IRBinOp::AND_OP_T, MJOLNIR::IRUnaryOp::TO_INT, bb);
                break;
            }
            case DEX::DVMTypes::Opcode::OP_AND_LONG:
            {
                lift_instruction23x_binary_instruction(instruction, MJOLNIR::IRBinOp::AND_OP_T, MJOLNIR::IRUnaryOp::TO_LONG, bb);
                break;
            }
            case DEX::DVMTypes::Opcode::OP_OR_INT:
            {
                lift_instruction23x_binary_instruction(instruction, MJOLNIR::IRBinOp::OR_OP_T, MJOLNIR::IRUnaryOp::TO_INT, bb);
                break;
            }
            case DEX::DVMTypes::Opcode::OP_OR_LONG:
            {
                lift_instruction23x_binary_instruction(instruction, MJOLNIR::IRBinOp::OR_OP_T, MJOLNIR::IRUnaryOp::TO_LONG, bb);
                break;
            }
            case DEX::DVMTypes::Opcode::OP_XOR_INT:
            {
                lift_instruction23x_binary_instruction(instruction, MJOLNIR::IRBinOp::XOR_OP_T, MJOLNIR::IRUnaryOp::TO_INT, bb);
                break;
            }
            case DEX::DVMTypes::Opcode::OP_XOR_LONG:
            {
                lift_instruction23x_binary_instruction(instruction, MJOLNIR::IRBinOp::XOR_OP_T, MJOLNIR::IRUnaryOp::TO_LONG, bb);
                break;
            }
            case DEX::DVMTypes::Opcode::OP_SHL_INT:
            {
                lift_instruction23x_binary_instruction(instruction, MJOLNIR::IRBinOp::SHL_OP_T, MJOLNIR::IRUnaryOp::TO_INT, bb);
                break;
            }
            case DEX::DVMTypes::Opcode::OP_SHL_LONG:
            {
                lift_instruction23x_binary_instruction(instruction, MJOLNIR::IRBinOp::SHL_OP_T, MJOLNIR::IRUnaryOp::TO_LONG, bb);
                break;
            }
            case DEX::DVMTypes::Opcode::OP_SHR_INT:
            {
                lift_instruction23x_binary_instruction(instruction, MJOLNIR::IRBinOp::SHR_OP_T, MJOLNIR::IRUnaryOp::TO_INT, bb);
                break;
            }
            case DEX::DVMTypes::Opcode::OP_SHR_LONG:
            {
                lift_instruction23x_binary_instruction(instruction, MJOLNIR::IRBinOp::SHR_OP_T, MJOLNIR::IRUnaryOp::TO_LONG, bb);
                break;
            }
            case DEX::DVMTypes::Opcode::OP_USHR_INT:
            {
                lift_instruction23x_binary_instruction(instruction, MJOLNIR::IRBinOp::USHR_OP_T, MJOLNIR::IRUnaryOp::TO_INT, bb);
                break;
            }
            case DEX::DVMTypes::Opcode::OP_USHR_LONG:
            {
                lift_instruction23x_binary_instruction(instruction, MJOLNIR::IRBinOp::USHR_OP_T, MJOLNIR::IRUnaryOp::TO_LONG, bb);
                break;
            }
            case DEX::DVMTypes::Opcode::OP_ADD_INT_2ADDR:
            case DEX::DVMTypes::Opcode::OP_ADD_LONG_2ADDR:
            case DEX::DVMTypes::Opcode::OP_ADD_FLOAT_2ADDR:
            case DEX::DVMTypes::Opcode::OP_ADD_DOUBLE_2ADDR:
            {
                lift_instruction12x_binary_instruction(instruction, MJOLNIR::IRBinOp::ADD_OP_T, MJOLNIR::IRUnaryOp::TO_ADDR, bb);
                break;
            }
            case DEX::DVMTypes::Opcode::OP_SUB_INT_2ADDR:
            case DEX::DVMTypes::Opcode::OP_SUB_LONG_2ADDR:
            case DEX::DVMTypes::Opcode::OP_SUB_FLOAT_2ADDR:
            case DEX::DVMTypes::Opcode::OP_SUB_DOUBLE_2ADDR:
            {
                lift_instruction12x_binary_instruction(instruction, MJOLNIR::IRBinOp::SUB_OP_T, MJOLNIR::IRUnaryOp::TO_ADDR, bb);
                break;
            }
            case DEX::DVMTypes::Opcode::OP_MUL_INT_2ADDR:
            case DEX::DVMTypes::Opcode::OP_MUL_LONG_2ADDR:
            case DEX::DVMTypes::Opcode::OP_MUL_FLOAT_2ADDR:
            case DEX::DVMTypes::Opcode::OP_MUL_DOUBLE_2ADDR:
            {
                lift_instruction12x_binary_instruction(instruction, MJOLNIR::IRBinOp::S_MUL_OP_T, MJOLNIR::IRUnaryOp::TO_ADDR, bb);
                break;
            }
            case DEX::DVMTypes::Opcode::OP_DIV_INT_2ADDR:
            case DEX::DVMTypes::Opcode::OP_DIV_LONG_2ADDR:
            case DEX::DVMTypes::Opcode::OP_DIV_FLOAT_2ADDR:
            case DEX::DVMTypes::Opcode::OP_DIV_DOUBLE_2ADDR:
            {
                lift_instruction12x_binary_instruction(instruction, MJOLNIR::IRBinOp::S_DIV_OP_T, MJOLNIR::IRUnaryOp::TO_ADDR, bb);
                break;
            }
            case DEX::DVMTypes::Opcode::OP_REM_INT_2ADDR:
            case DEX::DVMTypes::Opcode::OP_REM_LONG_2ADDR:
            case DEX::DVMTypes::Opcode::OP_REM_FLOAT_2ADDR:
            case DEX::DVMTypes::Opcode::OP_REM_DOUBLE_2ADDR:
            {
                lift_instruction12x_binary_instruction(instruction, MJOLNIR::IRBinOp::MOD_OP_T, MJOLNIR::IRUnaryOp::TO_ADDR, bb);
                break;
            }
            case DEX::DVMTypes::Opcode::OP_AND_INT_2ADDR:
            case DEX::DVMTypes::Opcode::OP_AND_LONG_2ADDR:
            {
                lift_instruction12x_binary_instruction(instruction, MJOLNIR::IRBinOp::AND_OP_T, MJOLNIR::IRUnaryOp::TO_ADDR, bb);
                break;
            }
            case DEX::DVMTypes::Opcode::OP_OR_INT_2ADDR:
            case DEX::DVMTypes::Opcode::OP_OR_LONG_2ADDR:
            {
                lift_instruction12x_binary_instruction(instruction, MJOLNIR::IRBinOp::OR_OP_T, MJOLNIR::IRUnaryOp::TO_ADDR, bb);
                break;
            }
            case DEX::DVMTypes::Opcode::OP_XOR_INT_2ADDR:
            case DEX::DVMTypes::Opcode::OP_XOR_LONG_2ADDR:
            {
                lift_instruction12x_binary_instruction(instruction, MJOLNIR::IRBinOp::XOR_OP_T, MJOLNIR::IRUnaryOp::TO_ADDR, bb);
                break;
            }
            case DEX::DVMTypes::Opcode::OP_SHL_INT_2ADDR:
            case DEX::DVMTypes::Opcode::OP_SHL_LONG_2ADDR:
            {
                lift_instruction12x_binary_instruction(instruction, MJOLNIR::IRBinOp::SHL_OP_T, MJOLNIR::IRUnaryOp::TO_ADDR, bb);
                break;
            }
            case DEX::DVMTypes::Opcode::OP_SHR_INT_2ADDR:
            case DEX::DVMTypes::Opcode::OP_SHR_LONG_2ADDR:
            {
                lift_instruction12x_binary_instruction(instruction, MJOLNIR::IRBinOp::SHR_OP_T, MJOLNIR::IRUnaryOp::TO_ADDR, bb);
                break;
            }
            case DEX::DVMTypes::Opcode::OP_USHR_INT_2ADDR:
            case DEX::DVMTypes::Opcode::OP_USHR_LONG_2ADDR:
            {
                lift_instruction12x_binary_instruction(instruction, MJOLNIR::IRBinOp::USHR_OP_T, MJOLNIR::IRUnaryOp::TO_ADDR, bb);
                break;
            }
            case DEX::DVMTypes::Opcode::OP_ADD_INT_LIT16:
            {
                lift_instruction22s_binary_instruction(instruction, MJOLNIR::IRBinOp::ADD_OP_T, MJOLNIR::IRUnaryOp::TO_INT, bb);
                break;
            }
            case DEX::DVMTypes::Opcode::OP_RSUB_INT:
            {
                lift_instruction22s_binary_instruction(instruction, MJOLNIR::IRBinOp::SUB_OP_T, MJOLNIR::IRUnaryOp::TO_INT, bb);
                break;
            }
            case DEX::DVMTypes::Opcode::OP_MUL_INT_LIT16:
            {
                lift_instruction22s_binary_instruction(instruction, MJOLNIR::IRBinOp::S_MUL_OP_T, MJOLNIR::IRUnaryOp::TO_INT, bb);
                break;
            }
            case DEX::DVMTypes::Opcode::OP_DIV_INT_LIT16:
            {
                lift_instruction22s_binary_instruction(instruction, MJOLNIR::IRBinOp::S_DIV_OP_T, MJOLNIR::IRUnaryOp::TO_INT, bb);
                break;
            }
            case DEX::DVMTypes::Opcode::OP_REM_INT_LIT16:
            {
                lift_instruction22s_binary_instruction(instruction, MJOLNIR::IRBinOp::MOD_OP_T, MJOLNIR::IRUnaryOp::TO_INT, bb);
                break;
            }
            case DEX::DVMTypes::Opcode::OP_AND_INT_LIT16:
            {
                lift_instruction22s_binary_instruction(instruction, MJOLNIR::IRBinOp::AND_OP_T, MJOLNIR::IRUnaryOp::TO_INT, bb);
                break;
            }
            case DEX::DVMTypes::Opcode::OP_OR_INT_LIT16:
            {
                lift_instruction22s_binary_instruction(instruction, MJOLNIR::IRBinOp::OR_OP_T, MJOLNIR::IRUnaryOp::TO_INT, bb);
                break;
            }
            case DEX::DVMTypes::Opcode::OP_XOR_INT_LIT16:
            {
                lift_instruction22s_binary_instruction(instruction, MJOLNIR::IRBinOp::XOR_OP_T, MJOLNIR::IRUnaryOp::TO_INT, bb);
                break;
            }
            case DEX::DVMTypes::Opcode::OP_ADD_INT_LIT8:
            {
                lift_instruction22b_binary_instruction(instruction, MJOLNIR::IRBinOp::ADD_OP_T, MJOLNIR::IRUnaryOp::TO_INT, bb);
                break;
            }
            case DEX::DVMTypes::Opcode::OP_RSUB_INT_LIT8:
            {
                lift_instruction22b_binary_instruction(instruction, MJOLNIR::IRBinOp::SUB_OP_T, MJOLNIR::IRUnaryOp::TO_INT, bb);
                break;
            }
            case DEX::DVMTypes::Opcode::OP_MUL_INT_LIT8:
            {
                lift_instruction22b_binary_instruction(instruction, MJOLNIR::IRBinOp::S_MUL_OP_T, MJOLNIR::IRUnaryOp::TO_INT, bb);
                break;
            }
            case DEX::DVMTypes::Opcode::OP_DIV_INT_LIT8:
            {
                lift_instruction22b_binary_instruction(instruction, MJOLNIR::IRBinOp::S_DIV_OP_T, MJOLNIR::IRUnaryOp::TO_INT, bb);
                break;
            }
            case DEX::DVMTypes::Opcode::OP_REM_INT_LIT8:
            {
                lift_instruction22b_binary_instruction(instruction, MJOLNIR::IRBinOp::MOD_OP_T, MJOLNIR::IRUnaryOp::TO_INT, bb);
                break;
            }
            case DEX::DVMTypes::Opcode::OP_AND_INT_LIT8:
            {
                lift_instruction22b_binary_instruction(instruction, MJOLNIR::IRBinOp::AND_OP_T, MJOLNIR::IRUnaryOp::TO_INT, bb);
                break;
            }
            case DEX::DVMTypes::Opcode::OP_OR_INT_LIT8:
            {
                lift_instruction22b_binary_instruction(instruction, MJOLNIR::IRBinOp::OR_OP_T, MJOLNIR::IRUnaryOp::TO_INT, bb);
                break;
            }
            case DEX::DVMTypes::Opcode::OP_XOR_INT_LIT8:
            {
                lift_instruction22b_binary_instruction(instruction, MJOLNIR::IRBinOp::XOR_OP_T, MJOLNIR::IRUnaryOp::TO_INT, bb);
                break;
            }
            case DEX::DVMTypes::Opcode::OP_SHL_INT_LIT8:
            {
                lift_instruction22b_binary_instruction(instruction, MJOLNIR::IRBinOp::SHL_OP_T, MJOLNIR::IRUnaryOp::TO_INT, bb);
                break;
            }
            case DEX::DVMTypes::Opcode::OP_SHR_INT_LIT8:
            {
                lift_instruction22b_binary_instruction(instruction, MJOLNIR::IRBinOp::SHR_OP_T, MJOLNIR::IRUnaryOp::TO_INT, bb);
                break;
            }
            case DEX::DVMTypes::Opcode::OP_USHR_INT_LIT8:
            {
                lift_instruction22b_binary_instruction(instruction, MJOLNIR::IRBinOp::USHR_OP_T, MJOLNIR::IRUnaryOp::TO_INT, bb);
                break;
            }
            case DEX::DVMTypes::Opcode::OP_NEG_INT:
            {
                lift_instruction12x_unary_instruction(instruction, MJOLNIR::IRUnaryOp::NEG_OP_T, MJOLNIR::IRUnaryOp::TO_INT, bb);
                break;
            }
            case DEX::DVMTypes::Opcode::OP_NEG_LONG:
            {
                lift_instruction12x_unary_instruction(instruction, MJOLNIR::IRUnaryOp::NEG_OP_T, MJOLNIR::IRUnaryOp::TO_LONG, bb);
                break;
            }
            case DEX::DVMTypes::Opcode::OP_NEG_FLOAT:
            {
                lift_instruction12x_unary_instruction(instruction, MJOLNIR::IRUnaryOp::NEG_OP_T, MJOLNIR::IRUnaryOp::TO_FLOAT, bb);
                break;
            }
            case DEX::DVMTypes::Opcode::OP_NEG_DOUBLE:
            {
                lift_instruction12x_unary_instruction(instruction, MJOLNIR::IRUnaryOp::NEG_OP_T, MJOLNIR::IRUnaryOp::TO_DOUBLE, bb);
                break;
            }
            case DEX::DVMTypes::Opcode::OP_NOT_INT:
            {
                lift_instruction12x_unary_instruction(instruction, MJOLNIR::IRUnaryOp::NOT_OP_T, MJOLNIR::IRUnaryOp::TO_INT, bb);
                break;
            }
            case DEX::DVMTypes::Opcode::OP_NOT_LONG:
            {
                lift_instruction12x_unary_instruction(instruction, MJOLNIR::IRUnaryOp::NOT_OP_T, MJOLNIR::IRUnaryOp::TO_LONG, bb);
                break;
            }
            case DEX::DVMTypes::Opcode::OP_INT_TO_LONG:
            case DEX::DVMTypes::Opcode::OP_FLOAT_TO_LONG:
            case DEX::DVMTypes::Opcode::OP_DOUBLE_TO_LONG:
            {
                lift_instruction12x_unary_instruction(instruction, MJOLNIR::IRUnaryOp::NONE_UNARY_OP_T, MJOLNIR::IRUnaryOp::TO_LONG, bb);
                break;
            }
            case DEX::DVMTypes::Opcode::OP_INT_TO_FLOAT:
            case DEX::DVMTypes::Opcode::OP_LONG_TO_FLOAT:
            case DEX::DVMTypes::Opcode::OP_DOUBLE_TO_FLOAT:
            {
                lift_instruction12x_unary_instruction(instruction, MJOLNIR::IRUnaryOp::NONE_UNARY_OP_T, MJOLNIR::IRUnaryOp::TO_FLOAT, bb);
                break;
            }
            case DEX::DVMTypes::Opcode::OP_INT_TO_DOUBLE:
            case DEX::DVMTypes::Opcode::OP_LONG_TO_DOUBLE:
            case DEX::DVMTypes::Opcode::OP_FLOAT_TO_DOUBLE:
            {
                lift_instruction12x_unary_instruction(instruction, MJOLNIR::IRUnaryOp::NONE_UNARY_OP_T, MJOLNIR::IRUnaryOp::TO_DOUBLE, bb);
                break;
            }
            case DEX::DVMTypes::Opcode::OP_LONG_TO_INT:
            case DEX::DVMTypes::Opcode::OP_FLOAT_TO_INT:
            case DEX::DVMTypes::Opcode::OP_DOUBLE_TO_INT:
            {
                lift_instruction12x_unary_instruction(instruction, MJOLNIR::IRUnaryOp::NONE_UNARY_OP_T, MJOLNIR::IRUnaryOp::TO_INT, bb);
                break;
            }
            case DEX::DVMTypes::Opcode::OP_INT_TO_BYTE:
            {
                lift_instruction12x_unary_instruction(instruction, MJOLNIR::IRUnaryOp::NONE_UNARY_OP_T, MJOLNIR::IRUnaryOp::TO_BYTE, bb);
                break;
            }
            case DEX::DVMTypes::Opcode::OP_INT_TO_CHAR:
            {
                lift_instruction12x_unary_instruction(instruction, MJOLNIR::IRUnaryOp::NONE_UNARY_OP_T, MJOLNIR::IRUnaryOp::TO_CHAR, bb);
                break;
            }
            case DEX::DVMTypes::Opcode::OP_INT_TO_SHORT:
            {
                lift_instruction12x_unary_instruction(instruction, MJOLNIR::IRUnaryOp::NONE_UNARY_OP_T, MJOLNIR::IRUnaryOp::TO_SHORT, bb);
                break;
            }
            case DEX::DVMTypes::Opcode::OP_RETURN_VOID:
            {
                std::shared_ptr<MJOLNIR::IRRet> ret_instr;

                auto none = make_none_type();

                ret_instr = std::make_shared<MJOLNIR::IRRet>(none);

                bb->append_statement_to_block(ret_instr);

                break;
            }
            case DEX::DVMTypes::Opcode::OP_RETURN:
            case DEX::DVMTypes::Opcode::OP_RETURN_WIDE:
            case DEX::DVMTypes::Opcode::OP_RETURN_OBJECT:
            {
                std::shared_ptr<MJOLNIR::IRRet> ret_instr;

                auto instr = std::dynamic_pointer_cast<DEX::Instruction11x>(instruction);

                auto reg = instr->get_destination();

                auto ir_reg = make_android_register(reg);

                ret_instr = std::make_shared<MJOLNIR::IRRet>(ir_reg);

                bb->append_statement_to_block(ret_instr);

                break;
            }
            case DEX::DVMTypes::Opcode::OP_CMPL_FLOAT:
            {
                lift_comparison_instruction(instruction, MJOLNIR::IRUnaryOp::TO_FLOAT, MJOLNIR::IRBComp::LOWER_T, bb);
                break;
            }
            case DEX::DVMTypes::Opcode::OP_CMPG_FLOAT:
            {
                lift_comparison_instruction(instruction, MJOLNIR::IRUnaryOp::TO_FLOAT, MJOLNIR::IRBComp::GREATER_T, bb);
                break;
            }
            case DEX::DVMTypes::Opcode::OP_CMPL_DOUBLE:
            {
                lift_comparison_instruction(instruction, MJOLNIR::IRUnaryOp::TO_DOUBLE, MJOLNIR::IRBComp::LOWER_T, bb);
                break;
            }
            case DEX::DVMTypes::Opcode::OP_CMPG_DOUBLE:
            {
                lift_comparison_instruction(instruction, MJOLNIR::IRUnaryOp::TO_DOUBLE, MJOLNIR::IRBComp::GREATER_T, bb);
                break;
            }
            case DEX::DVMTypes::Opcode::OP_CMP_LONG:
            {
                lift_comparison_instruction(instruction, MJOLNIR::IRUnaryOp::TO_LONG, MJOLNIR::IRBComp::EQUAL_T, bb);
                break;
            }
            case DEX::DVMTypes::Opcode::OP_IF_EQ:
            {
                lift_jcc_instruction22t(instruction, MJOLNIR::IRBComp::EQUAL_T, bb);
                break;
            }
            case DEX::DVMTypes::Opcode::OP_IF_NE:
            {
                lift_jcc_instruction22t(instruction, MJOLNIR::IRBComp::NOT_EQUAL_T, bb);
                break;
            }
            case DEX::DVMTypes::Opcode::OP_IF_LT:
            {
                lift_jcc_instruction22t(instruction, MJOLNIR::IRBComp::LOWER_T, bb);
                break;
            }
            case DEX::DVMTypes::Opcode::OP_IF_GE:
            {
                lift_jcc_instruction22t(instruction, MJOLNIR::IRBComp::GREATER_EQUAL_T, bb);
                break;
            }
            case DEX::DVMTypes::Opcode::OP_IF_GT:
            {
                lift_jcc_instruction22t(instruction, MJOLNIR::IRBComp::GREATER_T, bb);
                break;
            }
            case DEX::DVMTypes::Opcode::OP_IF_LE:
            {
                lift_jcc_instruction22t(instruction, MJOLNIR::IRBComp::LOWER_EQUAL_T, bb);
                break;
            }
            case DEX::DVMTypes::Opcode::OP_IF_EQZ:
            {
                lift_jcc_instruction21t(instruction, MJOLNIR::IRZComp::EQUAL_ZERO_T, bb);
                break;
            }
            case DEX::DVMTypes::Opcode::OP_IF_NEZ:
            {
                lift_jcc_instruction21t(instruction, MJOLNIR::IRZComp::NOT_EQUAL_ZERO_T, bb);
                break;
            }
            case DEX::DVMTypes::Opcode::OP_IF_LTZ:
            {
                lift_jcc_instruction21t(instruction, MJOLNIR::IRZComp::LOWER_ZERO_T, bb);
                break;
            }
            case DEX::DVMTypes::Opcode::OP_IF_GEZ:
            {
                lift_jcc_instruction21t(instruction, MJOLNIR::IRZComp::GREATER_EQUAL_ZERO, bb);
                break;
            }
            case DEX::DVMTypes::Opcode::OP_IF_GTZ:
            {
                lift_jcc_instruction21t(instruction, MJOLNIR::IRZComp::GREATER_ZERO_T, bb);
                break;
            }
            case DEX::DVMTypes::Opcode::OP_IF_LEZ:
            {
                lift_jcc_instruction21t(instruction, MJOLNIR::IRZComp::LOWER_EQUAL_ZERO, bb);
                break;
            }
            case DEX::DVMTypes::Opcode::OP_GOTO:
            {
                std::shared_ptr<MJOLNIR::IRUJmp> ujmp;

                auto jmp = std::dynamic_pointer_cast<DEX::Instruction10t>(instruction);

                auto addr = current_idx + (jmp->get_offset() * 2);

                ujmp = std::make_shared<MJOLNIR::IRUJmp>(addr, nullptr);

                bb->append_statement_to_block(ujmp);

                break;
            }
            case DEX::DVMTypes::Opcode::OP_GOTO_16:
            {
                std::shared_ptr<MJOLNIR::IRUJmp> ujmp;

                auto jmp = std::dynamic_pointer_cast<DEX::Instruction20t>(instruction);

                auto addr = current_idx + (jmp->get_offset() * 2);

                ujmp = std::make_shared<MJOLNIR::IRUJmp>(addr, nullptr);

                bb->append_statement_to_block(ujmp);

                break;
            }
            case DEX::DVMTypes::Opcode::OP_GOTO_32:
            {
                std::shared_ptr<MJOLNIR::IRUJmp> ujmp;

                auto jmp = std::dynamic_pointer_cast<DEX::Instruction30t>(instruction);

                auto addr = current_idx + (jmp->get_offset() * 2);

                ujmp = std::make_shared<MJOLNIR::IRUJmp>(addr, nullptr);

                bb->append_statement_to_block(ujmp);

                break;
            }
            case DEX::DVMTypes::Opcode::OP_INVOKE_VIRTUAL:
            case DEX::DVMTypes::Opcode::OP_INVOKE_SUPER:
            case DEX::DVMTypes::Opcode::OP_INVOKE_DIRECT:
            case DEX::DVMTypes::Opcode::OP_INVOKE_STATIC:
            case DEX::DVMTypes::Opcode::OP_INVOKE_INTERFACE:
            {
                std::shared_ptr<MJOLNIR::IRCall> call = nullptr;
                std::shared_ptr<MJOLNIR::IRExpr> callee = nullptr;

                MJOLNIR::IRCall::call_type_t call_type = MJOLNIR::IRCall::INTERNAL_CALL_T;

                auto op_code = static_cast<DEX::DVMTypes::Opcode>(instruction->get_OP());

                auto call_inst = std::dynamic_pointer_cast<DEX::Instruction35c>(instruction);
                std::vector<std::shared_ptr<MJOLNIR::IRExpr>> parameters;

                size_t p_size = call_inst->get_array_size();

                for (size_t i = 0; i < p_size; i++)
                    parameters.push_back(make_android_register(call_inst->get_operand_register(i)));

                // as it is a call to a method, we can safely retrieve the operand as method
                auto method_called = call_inst->get_operands_kind_method();

                // Get the values for the Callee
                std::string method_name = *method_called->get_method_name();

                std::string class_name = "";

                auto type = method_called->get_method_class();

                switch (type->get_type())
                {
                case DEX::Type::type_e::ARRAY:
                    class_name = std::dynamic_pointer_cast<DEX::Array>(type)->get_raw();
                    break;
                case DEX::Type::type_e::CLASS:
                    class_name = std::dynamic_pointer_cast<DEX::Class>(type)->get_raw();
                    break;
                default:
                    throw exceptions::LifterException("lift_android_instruction: DEX::DVMTypes::Opcode::OP_INVOKE_* type->get_type() not implemented");
                }

                std::string proto = method_called->get_method_prototype()->get_proto_str();

                if (this->android_analysis)
                {
                    auto method_analysis = this->android_analysis->get_method_analysis_by_name(class_name, method_name, proto);

                    if (method_analysis == nullptr || method_analysis->external() || method_analysis->is_android_api())
                    {
                        call_type = MJOLNIR::IRCall::EXTERNAL_CALL_T;
                    }
                }

                callee = std::make_shared<MJOLNIR::IRCallee>(0, method_name, class_name, p_size, proto, class_name + "->" + method_name + proto, ADDR_S);
                call = std::make_shared<MJOLNIR::IRCall>(callee, call_type, parameters);

                bb->append_statement_to_block(call);

                break;
            }
            case DEX::DVMTypes::Opcode::OP_INVOKE_VIRTUAL_RANGE:
            case DEX::DVMTypes::Opcode::OP_INVOKE_SUPER_RANGE:
            case DEX::DVMTypes::Opcode::OP_INVOKE_DIRECT_RANGE:
            case DEX::DVMTypes::Opcode::OP_INVOKE_STATIC_RANGE:
            case DEX::DVMTypes::Opcode::OP_INVOKE_INTERFACE_RANGE:
            {
                std::shared_ptr<MJOLNIR::IRCall> call = nullptr;
                std::shared_ptr<MJOLNIR::IRExpr> callee = nullptr;

                MJOLNIR::IRCall::call_type_t call_type = MJOLNIR::IRCall::INTERNAL_CALL_T;

                auto call_inst = std::dynamic_pointer_cast<DEX::Instruction3rc>(instruction);
                std::vector<std::shared_ptr<MJOLNIR::IRExpr>> parameters;

                size_t p_size = call_inst->get_array_size();

                for (size_t i = 0; i < p_size; i++)
                    parameters.push_back(make_android_register(call_inst->get_operand_register(i)));

                auto method_called = call_inst->get_operands_method();

                std::string method_name = *method_called->get_method_name();
                std::string class_name = std::dynamic_pointer_cast<DEX::Class>(method_called->get_method_class())->get_name();
                std::string proto = method_called->get_method_prototype()->get_proto_str();

                if (this->android_analysis)
                {
                    auto method_analysis = this->android_analysis->get_method_analysis_by_name(class_name, method_name, proto);

                    if (method_analysis == nullptr || method_analysis->external() || method_analysis->is_android_api())
                    {
                        call_type = MJOLNIR::IRCall::EXTERNAL_CALL_T;
                    }
                }

                callee = std::make_shared<MJOLNIR::IRCallee>(0, method_name, class_name, p_size, proto, class_name + "->" + method_name + proto, ADDR_S);
                call = std::make_shared<MJOLNIR::IRCall>(callee, call_type, parameters);

                bb->append_statement_to_block(call);

                break;
            }
            case DEX::DVMTypes::Opcode::OP_AGET:
            {
                lift_load_instruction(instruction, DWORD_S, MJOLNIR::IRUnaryOp::NONE_CAST, bb);
                break;
            }
            case DEX::DVMTypes::Opcode::OP_AGET_WIDE:
            {
                lift_load_instruction(instruction, QWORD_S, MJOLNIR::IRUnaryOp::NONE_CAST, bb);
                break;
            }
            case DEX::DVMTypes::Opcode::OP_AGET_OBJECT:
            {
                lift_load_instruction(instruction, ADDR_S, MJOLNIR::IRUnaryOp::TO_CLASS, bb);
                break;
            }
            case DEX::DVMTypes::Opcode::OP_AGET_BOOLEAN:
            {
                lift_load_instruction(instruction, DWORD_S, MJOLNIR::IRUnaryOp::TO_BOOLEAN, bb);
                break;
            }
            case DEX::DVMTypes::Opcode::OP_AGET_BYTE:
            {
                lift_load_instruction(instruction, BYTE_S, MJOLNIR::IRUnaryOp::TO_BYTE, bb);
                break;
            }
            case DEX::DVMTypes::Opcode::OP_AGET_CHAR:
            {
                lift_load_instruction(instruction, WORD_S, MJOLNIR::IRUnaryOp::TO_CHAR, bb);
                break;
            }
            case DEX::DVMTypes::Opcode::OP_AGET_SHORT:
            {
                lift_load_instruction(instruction, WORD_S, MJOLNIR::IRUnaryOp::TO_SHORT, bb);
                break;
            }
            case DEX::DVMTypes::Opcode::OP_APUT:
            {
                lift_store_instruction(instruction, DWORD_S, bb);
                break;
            }
            case DEX::DVMTypes::Opcode::OP_APUT_WIDE:
            {
                lift_store_instruction(instruction, QWORD_S, bb);
                break;
            }
            case DEX::DVMTypes::Opcode::OP_APUT_OBJECT:
            {
                lift_store_instruction(instruction, ADDR_S, bb);
                break;
            }
            case DEX::DVMTypes::Opcode::OP_APUT_BOOLEAN:
            {
                lift_store_instruction(instruction, DWORD_S, bb);
                break;
            }
            case DEX::DVMTypes::Opcode::OP_APUT_BYTE:
            {
                lift_store_instruction(instruction, BYTE_S, bb);
                break;
            }
            case DEX::DVMTypes::Opcode::OP_APUT_CHAR:
            {
                lift_store_instruction(instruction, WORD_S, bb);
                break;
            }
            case DEX::DVMTypes::Opcode::OP_APUT_SHORT:
            {
                lift_store_instruction(instruction, WORD_S, bb);
                break;
            }
            case DEX::DVMTypes::Opcode::OP_NEW_INSTANCE:
            {
                std::shared_ptr<MJOLNIR::IRStmnt> new_instr;
                auto op_code = static_cast<DEX::DVMTypes::Opcode>(instruction->get_OP());

                auto instr = std::dynamic_pointer_cast<DEX::Instruction21c>(instruction);

                auto dst_reg = make_android_register(instr->get_destination());

                auto class_ = make_class(std::dynamic_pointer_cast<DEX::Class>(instr->get_source_typeid()));

                new_instr = std::make_shared<MJOLNIR::IRNew>(dst_reg, class_);

                bb->append_statement_to_block(new_instr);

                break;
            }
            case DEX::DVMTypes::Opcode::OP_PACKED_SWITCH:
            {
                auto instr = std::dynamic_pointer_cast<DEX::Instruction31t>(instruction);
                auto condition = make_android_register(instr->get_array_ref());

                std::vector<std::int32_t> targets;
                std::vector<std::int32_t> checks;

                auto packed_switch = instr->get_packed_switch();
                auto switch_targets = packed_switch->get_targets();

                for (auto target : switch_targets)
                {
                    targets.push_back(current_idx + (target * 2));
                }

                auto switch_instr = std::make_shared<MJOLNIR::IRSwitch>(targets, condition, checks);
                bb->append_statement_to_block(switch_instr);

                break;
            }
            case DEX::DVMTypes::Opcode::OP_SPARSE_SWITCH:
            {
                auto instr = std::dynamic_pointer_cast<DEX::Instruction31t>(instruction);
                auto condition = make_android_register(instr->get_array_ref());

                std::vector<std::int32_t> targets;
                std::vector<std::int32_t> checks;

                auto sparse_switcht = instr->get_sparse_switch();

                auto key_targets = sparse_switcht->get_keys_targets();

                for (auto key_target : key_targets)
                {
                    checks.push_back(std::get<0>(key_target));
                    targets.push_back(current_idx + (std::get<1>(key_target) * 2));
                }

                auto switch_instr = std::make_shared<MJOLNIR::IRSwitch>(targets, condition, checks);
                bb->append_statement_to_block(switch_instr);

                break;
            }
            default:
                logger->error("Invalid instruction '{}'", std::to_string(op_code));
                return false;
            }
            return true;
        }

        /***
         * Private methods.
         */
        void LifterAndroid::lift_instruction23x_binary_instruction(KUNAI::DEX::instruction_t &instruction,
                                                                   KUNAI::MJOLNIR::IRBinOp::bin_op_t bin_op,
                                                                   MJOLNIR::IRUnaryOp::cast_type_t cast_type,
                                                                   MJOLNIR::irblock_t &bb)
        {
            MJOLNIR::irstmnt_t arith_logc_instr;
            MJOLNIR::irunaryop_t cast_instr;

            auto instr = std::dynamic_pointer_cast<DEX::Instruction23x>(instruction);

            auto dest = instr->get_destination();
            auto src1 = instr->get_first_source();
            auto src2 = instr->get_second_source();

            auto dest_reg = make_android_register(dest);
            auto src1_reg = make_android_register(src1);
            auto src2_reg = make_android_register(src2);

            arith_logc_instr = std::make_shared<MJOLNIR::IRBinOp>(bin_op, dest_reg, src1_reg, src2_reg);
            cast_instr = std::make_shared<MJOLNIR::IRUnaryOp>(MJOLNIR::IRUnaryOp::CAST_OP_T, cast_type, dest_reg, dest_reg);

            bb->append_statement_to_block(arith_logc_instr);
            bb->append_statement_to_block(cast_instr);
        }

        void LifterAndroid::lift_instruction12x_binary_instruction(KUNAI::DEX::instruction_t &instruction,
                                                                   KUNAI::MJOLNIR::IRBinOp::bin_op_t bin_op,
                                                                   MJOLNIR::IRUnaryOp::cast_type_t cast_type,
                                                                   MJOLNIR::irblock_t &bb)
        {
            MJOLNIR::irstmnt_t arith_logc_instr;
            MJOLNIR::irunaryop_t cast_instr;

            auto instr = std::dynamic_pointer_cast<DEX::Instruction12x>(instruction);

            auto dest = instr->get_destination();
            auto src = instr->get_source();

            auto dest_reg = make_android_register(dest);
            auto src_reg = make_android_register(src);

            arith_logc_instr = std::make_shared<MJOLNIR::IRBinOp>(bin_op, dest_reg, dest_reg, src_reg);
            cast_instr = std::make_shared<MJOLNIR::IRUnaryOp>(MJOLNIR::IRUnaryOp::CAST_OP_T, cast_type, dest_reg, dest_reg);

            bb->append_statement_to_block(arith_logc_instr);
            bb->append_statement_to_block(cast_instr);
        }

        void LifterAndroid::lift_instruction22s_binary_instruction(KUNAI::DEX::instruction_t &instruction,
                                                                   KUNAI::MJOLNIR::IRBinOp::bin_op_t bin_op,
                                                                   MJOLNIR::IRUnaryOp::cast_type_t cast_type,
                                                                   MJOLNIR::irblock_t &bb)
        {
            MJOLNIR::irstmnt_t arith_logc_instr;
            MJOLNIR::irunaryop_t cast_instr;

            auto instr = std::dynamic_pointer_cast<DEX::Instruction22s>(instruction);

            auto dest = instr->get_destination();
            auto src1 = instr->get_source();
            auto src2 = instr->get_number();

            auto dest_reg = make_android_register(dest);
            auto src1_reg = make_android_register(src1);
            auto src2_num = make_int(src2, true, WORD_S);

            arith_logc_instr = std::make_shared<MJOLNIR::IRBinOp>(bin_op, dest_reg, src1_reg, src2_num);
            cast_instr = std::make_shared<MJOLNIR::IRUnaryOp>(MJOLNIR::IRUnaryOp::CAST_OP_T, cast_type, dest_reg, dest_reg);

            bb->append_statement_to_block(arith_logc_instr);
            bb->append_statement_to_block(cast_instr);
        }

        void LifterAndroid::lift_instruction22b_binary_instruction(KUNAI::DEX::instruction_t &instruction,
                                                                   KUNAI::MJOLNIR::IRBinOp::bin_op_t bin_op,
                                                                   MJOLNIR::IRUnaryOp::cast_type_t cast_type,
                                                                   MJOLNIR::irblock_t &bb)
        {
            MJOLNIR::irstmnt_t arith_logc_instr;
            MJOLNIR::irunaryop_t cast_instr;

            auto instr = std::dynamic_pointer_cast<DEX::Instruction22b>(instruction);

            auto dest = instr->get_destination();
            auto src1 = instr->get_source();
            auto src2 = instr->get_number();

            auto dest_reg = make_android_register(dest);
            auto src1_reg = make_android_register(src1);
            auto src2_num = make_int(src2, true, BYTE_S);

            arith_logc_instr = std::make_shared<MJOLNIR::IRBinOp>(bin_op, dest_reg, src1_reg, src2_num);
            cast_instr = std::make_shared<MJOLNIR::IRUnaryOp>(MJOLNIR::IRUnaryOp::CAST_OP_T, cast_type, dest_reg, dest_reg);

            bb->append_statement_to_block(arith_logc_instr);
            bb->append_statement_to_block(cast_instr);
        }

        void LifterAndroid::lift_instruction12x_unary_instruction(KUNAI::DEX::instruction_t &instruction,
                                                                  KUNAI::MJOLNIR::IRUnaryOp::unary_op_t unary_op,
                                                                  MJOLNIR::IRUnaryOp::cast_type_t cast_type,
                                                                  MJOLNIR::irblock_t &bb)
        {
            MJOLNIR::irstmnt_t arith_logc_instr;
            MJOLNIR::irunaryop_t cast_instr;

            auto instr = std::dynamic_pointer_cast<DEX::Instruction12x>(instruction);

            auto dest = instr->get_destination();
            auto src = instr->get_source();

            auto dest_reg = make_android_register(dest);
            auto src_reg = make_android_register(src);

            if (unary_op != MJOLNIR::IRUnaryOp::NONE_UNARY_OP_T)
                arith_logc_instr = std::make_shared<MJOLNIR::IRUnaryOp>(unary_op, dest_reg, src_reg);

            cast_instr = std::make_shared<MJOLNIR::IRUnaryOp>(MJOLNIR::IRUnaryOp::CAST_OP_T, cast_type, dest_reg, dest_reg);

            if (unary_op != MJOLNIR::IRUnaryOp::NONE_UNARY_OP_T)
                bb->append_statement_to_block(arith_logc_instr);
            bb->append_statement_to_block(cast_instr);
        }

        void LifterAndroid::lift_comparison_instruction(KUNAI::DEX::instruction_t &instruction,
                                                        MJOLNIR::IRUnaryOp::cast_type_t cast_type,
                                                        MJOLNIR::IRBComp::comp_t comparison,
                                                        MJOLNIR::irblock_t &bb)
        {
            std::shared_ptr<MJOLNIR::IRBComp> ir_comp;

            auto op_code = static_cast<DEX::DVMTypes::Opcode>(instruction->get_OP());
            auto instr = std::dynamic_pointer_cast<DEX::Instruction23x>(instruction);

            auto reg1 = make_android_register(instr->get_first_source());
            auto reg2 = make_android_register(instr->get_second_source());
            auto result = make_android_register(instr->get_destination());

            auto cast1 = std::make_shared<MJOLNIR::IRUnaryOp>(MJOLNIR::IRUnaryOp::CAST_OP_T, cast_type, reg1, reg1);
            auto cast2 = std::make_shared<MJOLNIR::IRUnaryOp>(MJOLNIR::IRUnaryOp::CAST_OP_T, cast_type, reg2, reg2);

            ir_comp = std::make_shared<MJOLNIR::IRBComp>(comparison, result, reg1, reg2);

            bb->append_statement_to_block(cast1);
            bb->append_statement_to_block(cast2);
            bb->append_statement_to_block(ir_comp);
        }

        void LifterAndroid::lift_jcc_instruction22t(KUNAI::DEX::instruction_t &instruction,
                                                    MJOLNIR::IRBComp::comp_t comparison,
                                                    MJOLNIR::irblock_t &bb)
        {
            auto instr = std::dynamic_pointer_cast<DEX::Instruction22t>(instruction);

            auto temp_reg = make_temporal_register();

            auto reg1 = make_android_register(instr->get_first_check_reg());
            auto reg2 = make_android_register(instr->get_second_check_reg());

            uint64_t target = current_idx + (instr->get_ref() * 2);

            auto bcomp = std::make_shared<MJOLNIR::IRBComp>(comparison, temp_reg, reg1, reg2);
            auto ir_cond = std::make_shared<MJOLNIR::IRCJmp>(target, temp_reg, nullptr, nullptr);

            bb->append_statement_to_block(bcomp);
            bb->append_statement_to_block(ir_cond);
        }

        void LifterAndroid::lift_jcc_instruction21t(KUNAI::DEX::instruction_t &instruction,
                                                    MJOLNIR::IRZComp::zero_comp_t comparison,
                                                    MJOLNIR::irblock_t &bb)
        {
            auto instr = std::dynamic_pointer_cast<DEX::Instruction21t>(instruction);
            auto temp_reg = make_temporal_register();
            auto reg = make_android_register(instr->get_check_reg());

            uint64_t target = current_idx + (instr->get_ref() * 2);

            auto zcomp = std::make_shared<MJOLNIR::IRZComp>(comparison, temp_reg, reg);
            auto ir_cond = std::make_shared<MJOLNIR::IRCJmp>(target, temp_reg, nullptr, nullptr);

            bb->append_statement_to_block(zcomp);
            bb->append_statement_to_block(ir_cond);
        }

        void LifterAndroid::lift_load_instruction(DEX::instruction_t instruction, size_t size, MJOLNIR::IRUnaryOp::cast_type_t cast_type, MJOLNIR::irblock_t bb)
        {
            MJOLNIR::irexpr_t load_instr;
            MJOLNIR::irunaryop_t cast_instr;
            auto op_code = static_cast<DEX::DVMTypes::Opcode>(instruction->get_OP());

            auto inst = std::dynamic_pointer_cast<DEX::Instruction23x>(instruction);

            auto dst = make_android_register(inst->get_destination());
            auto source = make_android_register(inst->get_first_source());
            auto index = make_android_register(inst->get_second_source());

            load_instr = std::make_shared<MJOLNIR::IRLoad>(dst, source, index, size);
            bb->append_statement_to_block(load_instr);

            if (cast_type != MJOLNIR::IRUnaryOp::NONE_CAST)
            {
                cast_instr = std::make_shared<MJOLNIR::IRUnaryOp>(MJOLNIR::IRUnaryOp::CAST_OP_T, cast_type, dst, dst);
                bb->append_statement_to_block(cast_instr);
            }
        }

        void LifterAndroid::lift_store_instruction(DEX::instruction_t instruction, size_t size, MJOLNIR::irblock_t bb)
        {
            std::shared_ptr<MJOLNIR::IRExpr> store_instr;
            auto inst = std::dynamic_pointer_cast<DEX::Instruction23x>(instruction);
            auto op_code = static_cast<DEX::DVMTypes::Opcode>(instruction->get_OP());

            auto dst = make_android_register(inst->get_destination());
            auto source = make_android_register(inst->get_first_source());
            auto index = make_android_register(inst->get_second_source());

            store_instr = std::make_shared<MJOLNIR::IRStore>(dst, source, index, size);
            bb->append_statement_to_block(store_instr);
        }

        MJOLNIR::irreg_t LifterAndroid::make_android_register(std::uint32_t reg_id)
        {
            // check if was already created
            // in that case return the already created one
            if (created_registers.find(reg_id) != created_registers.end())
                return created_registers[reg_id];

            created_registers[reg_id] = std::make_shared<MJOLNIR::IRReg>(reg_id, MJOLNIR::dalvik_arch, "v" + std::to_string(reg_id), DWORD_S);
            return created_registers[reg_id];
        }

        MJOLNIR::irtempreg_t LifterAndroid::make_temporal_register()
        {
            auto temp_reg = std::make_shared<MJOLNIR::IRTempReg>(temp_reg_id, "t" + std::to_string(temp_reg_id), DWORD_S);
            temp_reg_id++;
            return temp_reg;
        }

        MJOLNIR::irtype_t LifterAndroid::make_none_type()
        {
            return std::make_shared<MJOLNIR::IRType>(MJOLNIR::IRType::NONE_TYPE, MJOLNIR::IRStmnt::NONE_OP_T, "", 0);
        }

        MJOLNIR::irconstint_t LifterAndroid::make_int(std::uint64_t value, bool is_signed, size_t type_size)
        {
            std::string int_representation;

            if (is_signed)
                int_representation = std::to_string(static_cast<std::int64_t>(value));
            else
                int_representation = std::to_string(value);

            return std::make_shared<MJOLNIR::IRConstInt>(value, is_signed, MJOLNIR::IRType::LE_ACCESS, int_representation, type_size);
        }

        MJOLNIR::irstring_t LifterAndroid::make_str(std::string value)
        {
            return std::make_shared<MJOLNIR::IRString>(value, value, value.length());
        }

        MJOLNIR::irclass_t LifterAndroid::make_class(DEX::class_t value)
        {
            return std::make_shared<MJOLNIR::IRClass>(value->get_name(), value->get_name(), 0);
        }

        MJOLNIR::irfield_t LifterAndroid::make_field(DEX::fieldid_t field)
        {
            DEX::class_t class_idx = std::dynamic_pointer_cast<DEX::Class>(field->get_class_idx());
            std::string class_name = class_idx->get_name();
            MJOLNIR::IRField::field_t field_type;
            std::string field_type_class = "";
            size_t type_size;
            std::stringstream type_name;
            std::string field_name = *field->get_name_idx();

            if (field->get_type_idx()->get_type() == DEX::Type::FUNDAMENTAL)
            {
                DEX::fundamental_t fundamental_idx = std::dynamic_pointer_cast<DEX::Fundamental>(field->get_type_idx());

                switch (fundamental_idx->get_fundamental_type())
                {
                case DEX::Fundamental::BOOLEAN:
                    field_type = MJOLNIR::IRField::BOOLEAN_F;
                    type_size = DWORD_S;
                    type_name << "BOOLEAN ";
                    break;
                case DEX::Fundamental::BYTE:
                    field_type = MJOLNIR::IRField::BYTE_F;
                    type_size = BYTE_S;
                    type_name << "byte ";
                    break;
                case DEX::Fundamental::CHAR:
                    field_type = MJOLNIR::IRField::CHAR_F;
                    type_size = WORD_S; // in Java Char is 2 bytes
                    type_name << "char ";
                    break;
                case DEX::Fundamental::DOUBLE:
                    field_type = MJOLNIR::IRField::DOUBLE_F;
                    type_size = QWORD_S;
                    type_name << "double ";
                    break;
                case DEX::Fundamental::FLOAT:
                    field_type = MJOLNIR::IRField::FLOAT_F;
                    type_size = DWORD_S;
                    type_name << "float ";
                    break;
                case DEX::Fundamental::INT:
                    field_type = MJOLNIR::IRField::INT_F;
                    type_size = DWORD_S;
                    type_name << "int ";
                    break;
                case DEX::Fundamental::LONG:
                    field_type = MJOLNIR::IRField::LONG_F;
                    type_size = QWORD_S;
                    type_name << "long ";
                    break;
                case DEX::Fundamental::SHORT:
                    field_type = MJOLNIR::IRField::SHORT_F;
                    type_size = WORD_S;
                    type_name << "short ";
                    break;
                case DEX::Fundamental::VOID:
                    field_type = MJOLNIR::IRField::VOID_F;
                    type_size = 0;
                    type_name << "void ";
                    break;
                }

                type_name << class_name << "." << field_name;

                return std::make_shared<MJOLNIR::IRField>(class_name, field_type, field_name, type_name.str(), type_size);
            }
            else if (field->get_type_idx()->get_type() == DEX::Type::CLASS)
            {
                DEX::class_t type_idx = std::dynamic_pointer_cast<DEX::Class>(field->get_type_idx());
                field_type = MJOLNIR::IRField::CLASS_F;
                field_type_class = type_idx->get_name();
                type_size = ADDR_S;
                type_name << field_type_class << " " << class_name << "." << field_name;

                return std::make_shared<MJOLNIR::IRField>(class_name, field_type_class, field_name, type_name.str(), type_size);
            }
            else if (field->get_type_idx()->get_type() == DEX::Type::ARRAY)
            {
                field_type = MJOLNIR::IRField::ARRAY_F;
                type_size = 0;

                type_name << "ARRAY " << class_name << "." << field_name;
                return std::make_shared<MJOLNIR::IRField>(class_name, field_type, field_name, type_name.str(), type_size);
            }

            return nullptr;
        }

        void LifterAndroid::jump_target_analysis(std::vector<std::shared_ptr<KUNAI::DEX::DVMBasicBlock>>& bbs, MJOLNIR::irgraph_t method_graph)
        {
            for (auto bb : bbs)
            {
                auto next_bbs = bb->get_next();

                auto current_bb = lifted_blocks[bb.get()];

                if (current_bb == nullptr || current_bb->get_number_of_statements() == 0) // security check
                    continue;

                // now set some interesting stuff for instructions.
                // now set some interesting stuff for instructions.
                auto last_instr = current_bb->get_statements().back();

                if (auto jmp = MJOLNIR::unconditional_jump_ir(last_instr))
                {
                    if (next_bbs.size() == 1)
                    {
                        auto block = std::get<2>(next_bbs[0]);
                        jmp->set_jump_target(lifted_blocks[block]);

                        method_graph->add_uniq_edge(current_bb, lifted_blocks[block]);
                    }
                }
                else if (auto jcc = MJOLNIR::conditional_jump_ir(last_instr))
                {
                    if (next_bbs.size() == 2)
                    {
                        auto bb1 = std::get<2>(next_bbs[0]);
                        auto bb2 = std::get<2>(next_bbs[1]);

                        if (bb1->get_start() == jcc->get_addr()) // if bb1 is target of jump
                        {
                            jcc->set_jump_target(lifted_blocks[bb1]);
                            jcc->set_fallthrough_Target(lifted_blocks[bb2]);
                        }
                        else
                        {
                            jcc->set_jump_target(lifted_blocks[bb2]);
                            jcc->set_fallthrough_Target(lifted_blocks[bb1]);
                        }
                    }
                }
            }
        }

    }
}