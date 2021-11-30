#include "lifter_android.hpp"

namespace KUNAI
{
    namespace LIFTER
    {

        /**
         * @brief Constructor of lifter.
         */
        LifterAndroid::LifterAndroid()
        {
            temp_reg_id = 0;
            current_idx = 0;
        }

        /**
         * @brief Lift a given method from a method_analysis object.
         * 
         * @param method_analysis: method from Android to lift.
         * @return std::shared_ptr<MJOLNIR::IRGraph>
         */
        std::shared_ptr<MJOLNIR::IRGraph> LifterAndroid::lift_android_method(std::shared_ptr<DEX::MethodAnalysis> method_analysis)
        {
            auto bbs = method_analysis->get_basic_blocks()->get_basic_blocks();
            size_t n_bbs = bbs.size();
            // graph returnedd by
            std::shared_ptr<MJOLNIR::IRGraph> method_graph = std::make_shared<MJOLNIR::IRGraph>();

            // first of all lift all the blocks
            for (auto bb : bbs)
            {
                std::shared_ptr<MJOLNIR::IRBlock> lifted_bb = std::make_shared<MJOLNIR::IRBlock>();

                this->lift_android_basic_block(bb, lifted_bb);

                lifted_blocks[bb] = lifted_bb;

                method_graph->add_node(lifted_bb);
            }

            // Create Control Flow Graph using the children nodes
            // from the method blocks.
            for (auto bb : bbs)
            {
                auto next_bbs = bb->get_next();

                auto current_bb = lifted_blocks[bb];

                for (auto next_bb : next_bbs)
                {
                    auto block = std::get<2>(next_bb);
                    method_graph->add_edge(current_bb, lifted_blocks[block]);
                }
            }

            this->jump_target_analysis(bbs);
            this->fallthrough_target_analysis(method_graph);

            return method_graph;
        }

        /**
         * @brief Lift an android basic block instructions to IR instructions.
         * 
         * @param basic_block: basic block with Android instructions. 
         * @param bb: IR Basic Block.
         * @return true 
         * @return false 
         */
        bool LifterAndroid::lift_android_basic_block(std::shared_ptr<DEX::DVMBasicBlock> basic_block, std::shared_ptr<MJOLNIR::IRBlock> bb)
        {
            auto instructions = basic_block->get_instructions();
            auto next = basic_block->get_next();

            for (auto instruction : instructions)
            {
                if (androidinstructions.move_result_instruction.find(static_cast<DEX::DVMTypes::Opcode>(instruction->get_OP())) != androidinstructions.move_result_instruction.end())
                {

                    if (bb->get_number_of_statements() == 0) // security check if there are or not statements
                        continue;

                    auto last_instr = bb->get_statements().back();

                    if (MJOLNIR::is_call(last_instr))
                    {
                        auto ir_call = std::dynamic_pointer_cast<MJOLNIR::IRCall>(last_instr);
                        this->lift_move_result_instruction(instruction, ir_call);
                    }
                }
                else
                    this->lift_android_instruction(instruction, bb);
                
                current_idx += instruction->get_length();
            }

            bb->set_start_idx(basic_block->get_start());
            bb->set_end_idx(basic_block->get_end());

            return true;
        }

        /**
         * @brief 
         * 
         * @param instruction: instruction from android to lift.
         * @param bb: IR Basic Block.
         * @return true 
         * @return false 
         */
        bool LifterAndroid::lift_android_instruction(std::shared_ptr<DEX::Instruction> instruction, std::shared_ptr<MJOLNIR::IRBlock> bb)
        {
            auto op_code = static_cast<DEX::DVMTypes::Opcode>(instruction->get_OP());

            if (androidinstructions.assignment_instruction.find(op_code) != androidinstructions.assignment_instruction.end())
                this->lift_assignment_instruction(instruction, bb);
            else if (androidinstructions.arithmetic_logic_instruction.find(op_code) != androidinstructions.arithmetic_logic_instruction.end())
                this->lift_arithmetic_logic_instruction(instruction, bb);
            else if (androidinstructions.ret_instruction.find(op_code) != androidinstructions.ret_instruction.end())
                this->lift_ret_instruction(instruction, bb);
            else if (androidinstructions.cmp_instruction.find(op_code) != androidinstructions.cmp_instruction.end())
                this->lift_comparison_instruction(instruction, bb);
            else if (androidinstructions.jcc_instruction.find(op_code) != androidinstructions.jcc_instruction.end())
                this->lift_conditional_jump_instruction(instruction, bb);
            else if (androidinstructions.jmp_instruction.find(op_code) != androidinstructions.jmp_instruction.end())
                this->lift_unconditional_jump_instruction(instruction, bb);
            else if (androidinstructions.call_instructions.find(op_code) != androidinstructions.call_instructions.end())
                this->lift_call_instruction(instruction, bb);

            return true;
        }

        /***
         * Private methods.
         */

        /**
         * @brief Create a new IRReg with the id of the register,
         *        using different objects will be useful for having
         *        different registers in different methods.
         * @param reg_id: id of the register to create.
         * @return std::shared_ptr<MJOLNIR::IRReg>
         */
        std::shared_ptr<MJOLNIR::IRReg> LifterAndroid::make_android_register(std::uint32_t reg_id)
        {
            return std::make_shared<MJOLNIR::IRReg>(reg_id, MJOLNIR::dalvik_arch, "v" + std::to_string(reg_id), DWORD_S);
        }

        /**
         * @brief Return a temporal register, used for operations like
         *        conditional jumps to have some place where to store result
         *        of comparison.
         * 
         * @return std::shared_ptr<MJOLNIR::IRTempReg> 
         */
        std::shared_ptr<MJOLNIR::IRTempReg> LifterAndroid::make_temporal_register()
        {
            auto temp_reg = std::make_shared<MJOLNIR::IRTempReg>(temp_reg_id, "t" + std::to_string(temp_reg_id), DWORD_S);
            temp_reg_id++;
            return temp_reg;
        }

        /**
         * @brief Create a NONE type useful for cases where one of
         *        the parameters does not exists.
         * @return std::shared_ptr<MJOLNIR::IRType>
         */
        std::shared_ptr<MJOLNIR::IRType> LifterAndroid::make_none_type()
        {
            return std::make_shared<MJOLNIR::IRType>(MJOLNIR::IRType::NONE_TYPE, "", 0);
        }

        /**
         * @brief Create an Int type used for the DEX instructions
         * @param value: value of the integer as an unsigned of 64 bits
         * @param is_signed: boolean saying if the value must be signed or unsigned.
         * @param type_size: size of the integer type.
         * @return std::shared_ptr<MJOLNIR::IRConstInt>
         */
        std::shared_ptr<MJOLNIR::IRConstInt> LifterAndroid::make_int(std::uint64_t value, bool is_signed, size_t type_size)
        {
            std::string int_representation;

            if (is_signed)
                int_representation = std::to_string(static_cast<std::int64_t>(value));
            else
                int_representation = std::to_string(value);

            return std::make_shared<MJOLNIR::IRConstInt>(value, is_signed, MJOLNIR::IRType::LE_ACCESS, int_representation, type_size);
        }

        /**
         * @brief Generate a IRString type for Android, this will have the string
         *        and the size of the string.
         * @param value: string to generate the object.
         * @return std::shared_ptr<MJOLNIR::IRString>
         */
        std::shared_ptr<MJOLNIR::IRString> LifterAndroid::make_str(std::string value)
        {
            return std::make_shared<MJOLNIR::IRString>(value, value, value.length());
        }

        /**
         * @brief Generate a IRClass type for Android, this will be nothing more
         *        than the complete name of the class.
         * @param value: class to generate the object.
         * @return std::shared_ptr<MJOLNIR::IRClass>
         */
        std::shared_ptr<MJOLNIR::IRClass> LifterAndroid::make_class(DEX::Class *value)
        {
            return std::make_shared<MJOLNIR::IRClass>(value->get_name(), value->get_name(), 0);
        }

        /**
         * @brief Generate a IRField type for Android, this has the values from the
         *        FieldID.
         * @param field: FieldID to generate the object. 
         * @return std::shared_ptr<MJOLNIR::IRField> 
         */
        std::shared_ptr<MJOLNIR::IRField> LifterAndroid::make_field(DEX::FieldID *field)
        {
            DEX::Class *class_idx = reinterpret_cast<DEX::Class *>(field->get_class_idx());
            std::string class_name = class_idx->get_name();
            MJOLNIR::IRField::field_t field_type;
            std::string field_type_class = "";
            size_t type_size;
            std::stringstream type_name;
            std::string field_name = *field->get_name_idx();

            if (field->get_type_idx()->get_type() == DEX::Type::FUNDAMENTAL)
            {
                DEX::Fundamental *fundamental_idx = reinterpret_cast<DEX::Fundamental *>(field->get_type_idx());

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
                DEX::Class *type_idx = reinterpret_cast<DEX::Class *>(field->get_type_idx());
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

        /**
         * @brief Generate a IRAssign instruction from the IR, this will
         *        be any instruction for assigning strings, classes or registers
         *        to registers.
         * @param instruction: instruction to lift.
         * @param bb: basic block where to insert the instruction.
         * @return void.
         */
        void LifterAndroid::lift_assignment_instruction(std::shared_ptr<DEX::Instruction> instruction, std::shared_ptr<MJOLNIR::IRBlock> bb)
        {
            std::shared_ptr<MJOLNIR::IRStmnt> assignment_instr;
            std::shared_ptr<MJOLNIR::IRUnaryOp> cast_instr = nullptr;
            auto op_code = static_cast<DEX::DVMTypes::Opcode>(instruction->get_OP());

            // Instruction12x types
            if (androidinstructions.assignment_instruction12x.find(op_code) != androidinstructions.assignment_instruction12x.end())
            {
                auto instr = std::dynamic_pointer_cast<DEX::Instruction12x>(instruction);
                auto dest = instr->get_destination();
                auto src = instr->get_source();

                auto dest_reg = make_android_register(dest);
                auto src_reg = make_android_register(src);

                assignment_instr = std::make_shared<MJOLNIR::IRAssign>(dest_reg, src_reg, nullptr, nullptr);
            }
            // Instruction22x types
            else if (androidinstructions.assignment_instruction22x.find(op_code) != androidinstructions.assignment_instruction22x.end())
            {
                auto instr = std::dynamic_pointer_cast<DEX::Instruction22x>(instruction);
                auto dest = instr->get_destination();
                auto src = instr->get_source();

                auto dest_reg = make_android_register(dest);
                auto src_reg = make_android_register(src);

                assignment_instr = std::make_shared<MJOLNIR::IRAssign>(dest_reg, src_reg, nullptr, nullptr);
            }
            // Instruction32x types
            else if (androidinstructions.assignment_instruction32x.find(op_code) != androidinstructions.assignment_instruction32x.end())
            {
                auto instr = std::dynamic_pointer_cast<DEX::Instruction32x>(instruction);

                auto dest = instr->get_destination();
                auto src = instr->get_source();

                auto dest_reg = make_android_register(dest);
                auto src_reg = make_android_register(src);

                assignment_instr = std::make_shared<MJOLNIR::IRAssign>(dest_reg, src_reg, nullptr, nullptr);
            }
            else if (androidinstructions.assigment_instruction11n.find(op_code) != androidinstructions.assigment_instruction11n.end())
            {
                auto instr = std::dynamic_pointer_cast<DEX::Instruction11n>(instruction);

                auto dest = instr->get_destination();
                auto src = instr->get_source();

                auto dest_reg = make_android_register(dest);
                auto src_int = make_int(src, true, NIBBLE_S);

                assignment_instr = std::make_shared<MJOLNIR::IRAssign>(dest_reg, src_int, nullptr, nullptr);
            }
            // Instruction21s types
            else if (androidinstructions.assigment_instruction21s.find(op_code) != androidinstructions.assigment_instruction21s.end())
            {
                auto instr = std::dynamic_pointer_cast<DEX::Instruction21s>(instruction);

                auto dest = instr->get_destination();
                auto src = instr->get_source();

                auto dest_reg = make_android_register(dest);
                auto src_int = make_int(src, true, WORD_S);

                assignment_instr = std::make_shared<MJOLNIR::IRAssign>(dest_reg, src_int, nullptr, nullptr);
            }
            // Instrution31i types
            else if (androidinstructions.assigment_instruction31i.find(op_code) != androidinstructions.assigment_instruction31i.end())
            {
                auto instr = std::dynamic_pointer_cast<DEX::Instruction31i>(instruction);

                auto dest = instr->get_destination();
                auto src = instr->get_source();

                auto dest_reg = make_android_register(dest);
                auto src_int = make_int(src, true, DWORD_S);

                assignment_instr = std::make_shared<MJOLNIR::IRAssign>(dest_reg, src_int, nullptr, nullptr);
            }
            // Instruction21h types
            else if (androidinstructions.assigment_instruction21h.find(op_code) != androidinstructions.assigment_instruction21h.end())
            {
                auto instr = std::dynamic_pointer_cast<DEX::Instruction21h>(instruction);

                auto dest = instr->get_destination();
                auto src = instr->get_source();

                auto dest_reg = make_android_register(dest);
                std::shared_ptr<KUNAI::MJOLNIR::IRConstInt> src_int = nullptr;

                if (op_code == DEX::DVMTypes::Opcode::OP_CONST_HIGH16)
                    src_int = make_int(src << 16, true, QWORD_S);
                else if (op_code == DEX::DVMTypes::Opcode::OP_CONST_WIDE_HIGH16)
                    src_int = make_int(src << 48, true, QWORD_S);

                assignment_instr = std::make_shared<MJOLNIR::IRAssign>(dest_reg, src_int, nullptr, nullptr);
            }
            // Instruction 51l types
            else if (androidinstructions.assigment_instruction51l.find(op_code) != androidinstructions.assigment_instruction51l.end())
            {
                auto instr = std::dynamic_pointer_cast<DEX::Instruction51l>(instruction);

                auto dest = instr->get_destination();
                auto src = instr->get_source();

                auto dest_reg = make_android_register(dest);
                auto src_int = make_int(src, true, QWORD_S);

                assignment_instr = std::make_shared<MJOLNIR::IRAssign>(dest_reg, src_int, nullptr, nullptr);
            }
            // Instruction21c types
            else if (androidinstructions.assigment_instruction21c.find(op_code) != androidinstructions.assigment_instruction21c.end())
            {
                auto instr = std::dynamic_pointer_cast<DEX::Instruction21c>(instruction);

                auto dest = instr->get_destination();
                auto dest_reg = make_android_register(dest);

                if (op_code == DEX::DVMTypes::Opcode::OP_CONST_STRING)
                {
                    auto src = make_str(*instr->get_source_str());
                    assignment_instr = std::make_shared<MJOLNIR::IRAssign>(dest_reg, src, nullptr, nullptr);
                }
                else if (op_code == DEX::DVMTypes::Opcode::OP_CONST_CLASS)
                {
                    if (instr->get_source_typeid()->get_type() != DEX::Type::CLASS)
                    {
                        // ToDo generate an exception
                        return;
                    }
                    auto src_class = dynamic_cast<DEX::Class *>(instr->get_source_typeid());
                    auto src = make_class(src_class);

                    assignment_instr = std::make_shared<MJOLNIR::IRAssign>(dest_reg, src, nullptr, nullptr);
                }
                else
                {
                    switch (instr->get_source_kind())
                    {
                    case DEX::DVMTypes::METH:
                        /* Todo */
                        {
                            std::cerr << "Not implemented DEX::DVMTypes::METH yet" << std::endl;
                            auto src = make_none_type();
                            assignment_instr = std::make_shared<MJOLNIR::IRAssign>(dest_reg, src, nullptr, nullptr);
                        }
                        break;
                    case DEX::DVMTypes::STRING:
                    {
                        auto src = make_str(*instr->get_source_str());
                        assignment_instr = std::make_shared<MJOLNIR::IRAssign>(dest_reg, src, nullptr, nullptr);
                    }
                    break;
                    case DEX::DVMTypes::TYPE:
                        /* Todo */
                        {
                            std::cerr << "Not implemented DEX::DVMTypes::TYPE yet" << std::endl;
                            auto src = make_none_type();
                            assignment_instr = std::make_shared<MJOLNIR::IRAssign>(dest_reg, src, nullptr, nullptr);
                        }
                        break;
                    case DEX::DVMTypes::FIELD:
                    {
                        auto src = make_field(instr->get_source_static_field());
                        assignment_instr = std::make_shared<MJOLNIR::IRAssign>(dest_reg, src, nullptr, nullptr);

                        if (src->get_type() == MJOLNIR::IRField::BOOLEAN_F)
                            cast_instr = std::make_shared<MJOLNIR::IRUnaryOp>(MJOLNIR::IRUnaryOp::CAST_OP_T, MJOLNIR::IRUnaryOp::TO_BOOLEAN, dest_reg, dest_reg, nullptr, nullptr);
                        else if (src->get_type() == MJOLNIR::IRField::BYTE_F)
                            cast_instr = std::make_shared<MJOLNIR::IRUnaryOp>(MJOLNIR::IRUnaryOp::CAST_OP_T, MJOLNIR::IRUnaryOp::TO_BYTE, dest_reg, dest_reg, nullptr, nullptr);
                        else if (src->get_type() == MJOLNIR::IRField::CHAR_F)
                            cast_instr = std::make_shared<MJOLNIR::IRUnaryOp>(MJOLNIR::IRUnaryOp::CAST_OP_T, MJOLNIR::IRUnaryOp::TO_CHAR, dest_reg, dest_reg, nullptr, nullptr);
                        else if (src->get_type() == MJOLNIR::IRField::SHORT_F)
                            cast_instr = std::make_shared<MJOLNIR::IRUnaryOp>(MJOLNIR::IRUnaryOp::CAST_OP_T, MJOLNIR::IRUnaryOp::TO_SHORT, dest_reg, dest_reg, nullptr, nullptr);
                        else if (src->get_type() == MJOLNIR::IRField::INT_F)
                            cast_instr = std::make_shared<MJOLNIR::IRUnaryOp>(MJOLNIR::IRUnaryOp::CAST_OP_T, MJOLNIR::IRUnaryOp::TO_INT, dest_reg, dest_reg, nullptr, nullptr);
                        else if (src->get_type() == MJOLNIR::IRField::DOUBLE_F)
                            cast_instr = std::make_shared<MJOLNIR::IRUnaryOp>(MJOLNIR::IRUnaryOp::CAST_OP_T, MJOLNIR::IRUnaryOp::TO_DOUBLE, dest_reg, dest_reg, nullptr, nullptr);
                    }
                    break;
                    case DEX::DVMTypes::PROTO:
                        /* Todo */
                        {
                            std::cerr << "Not implemented DEX::DVMTypes::PROTO yet" << std::endl;
                            auto src = make_none_type();
                            assignment_instr = std::make_shared<MJOLNIR::IRAssign>(dest_reg, src, nullptr, nullptr);
                        }
                        break;
                    }
                }
            }
            else if (androidinstructions.assigment_instruction21_put.find(op_code) != androidinstructions.assigment_instruction21_put.end())
            {
                auto instr = std::dynamic_pointer_cast<DEX::Instruction21c>(instruction);

                // Instruction PUT follows the same instruction format
                // than GET, it follows same codification, but here
                // we have: SPUT Field, Regsister
                // here what we call  "destination" is the source of the data.
                // and source is a Field destination
                auto src = instr->get_destination();
                auto src_reg = make_android_register(src);

                auto dst = make_field(instr->get_source_static_field());

                assignment_instr = std::make_shared<MJOLNIR::IRAssign>(dst, src_reg, nullptr, nullptr);
            }
            // Instruction31c types
            else if (androidinstructions.assigment_instruction31c.find(op_code) != androidinstructions.assigment_instruction31c.end())
            {
                auto instr = std::dynamic_pointer_cast<DEX::Instruction31c>(instruction);

                auto dest = instr->get_destination();
                auto dest_reg = make_android_register(dest);

                auto src = make_str(*instr->get_source_str());
                assignment_instr = std::make_shared<MJOLNIR::IRAssign>(dest_reg, src, nullptr, nullptr);
            }

            // push instruction in the block
            bb->append_statement_to_block(assignment_instr);

            if (cast_instr != nullptr)
                bb->append_statement_to_block(cast_instr);
        }

        void LifterAndroid::lift_load_instruction(std::shared_ptr<DEX::Instruction> instruction, std::shared_ptr<MJOLNIR::IRBlock> bb)
        {
            std::shared_ptr<MJOLNIR::IRExpr> load_instr;
            std::shared_ptr<MJOLNIR::IRUnaryOp> cast_instr;
            auto op_code = static_cast<DEX::DVMTypes::Opcode>(instruction->get_OP());
        }

        /**
         * @brief Generate a IRBinOp instruction or IRUnaryOp instruction
         *        which represent any arithmetic logic instruction.
         * @param instruction: instruction to lift from arithmetic logic instructions.
         * @param bb: basic block where to insert the instructions.
         * @return void.
         */
        void LifterAndroid::lift_arithmetic_logic_instruction(std::shared_ptr<DEX::Instruction> instruction, std::shared_ptr<MJOLNIR::IRBlock> bb)
        {
            std::shared_ptr<MJOLNIR::IRExpr> arith_logc_instr;
            std::shared_ptr<MJOLNIR::IRUnaryOp> cast_instr = nullptr;
            auto op_code = static_cast<DEX::DVMTypes::Opcode>(instruction->get_OP());

            KUNAI::MJOLNIR::IRBinOp::bin_op_t bin_operation;
            KUNAI::MJOLNIR::IRUnaryOp::unary_op_t unary_operation;

            KUNAI::MJOLNIR::IRUnaryOp::cast_type_t cast_type;
            KUNAI::MJOLNIR::IRUnaryOp::unary_op_t cast_op = MJOLNIR::IRUnaryOp::CAST_OP_T;

            // first decide the type of operation
            if (androidinstructions.add_instruction.find(op_code) != androidinstructions.add_instruction.end())
                bin_operation = MJOLNIR::IRBinOp::ADD_OP_T;
            else if (androidinstructions.sub_instruction.find(op_code) != androidinstructions.sub_instruction.end())
                bin_operation = MJOLNIR::IRBinOp::SUB_OP_T;
            else if (androidinstructions.mul_instruction.find(op_code) != androidinstructions.mul_instruction.end())
                bin_operation = MJOLNIR::IRBinOp::S_MUL_OP_T;
            else if (androidinstructions.div_instruction.find(op_code) != androidinstructions.div_instruction.end())
                bin_operation = MJOLNIR::IRBinOp::S_DIV_OP_T;
            else if (androidinstructions.mod_instruction.find(op_code) != androidinstructions.mod_instruction.end())
                bin_operation = MJOLNIR::IRBinOp::MOD_OP_T;
            else if (androidinstructions.and_instruction.find(op_code) != androidinstructions.and_instruction.end())
                bin_operation = MJOLNIR::IRBinOp::AND_OP_T;
            else if (androidinstructions.or_instruction.find(op_code) != androidinstructions.or_instruction.end())
                bin_operation = MJOLNIR::IRBinOp::OR_OP_T;
            else if (androidinstructions.xor_instruction.find(op_code) != androidinstructions.xor_instruction.end())
                bin_operation = MJOLNIR::IRBinOp::XOR_OP_T;
            else if (androidinstructions.shl_instruction.find(op_code) != androidinstructions.shl_instruction.end())
                bin_operation = MJOLNIR::IRBinOp::SHL_OP_T;
            else if (androidinstructions.shr_instruction.find(op_code) != androidinstructions.shr_instruction.end())
                bin_operation = MJOLNIR::IRBinOp::SHR_OP_T;
            else if (androidinstructions.ushr_instruction.find(op_code) != androidinstructions.ushr_instruction.end())
                bin_operation = MJOLNIR::IRBinOp::USHR_OP_T;
            else if (androidinstructions.neg_instruction.find(op_code) != androidinstructions.neg_instruction.end())
                unary_operation = KUNAI::MJOLNIR::IRUnaryOp::NEG_OP_T;
            else if (androidinstructions.not_instruction.find(op_code) != androidinstructions.not_instruction.end())
                unary_operation = KUNAI::MJOLNIR::IRUnaryOp::NOT_OP_T;
            else if (androidinstructions.cast_instruction.find(op_code) != androidinstructions.cast_instruction.end())
                unary_operation = KUNAI::MJOLNIR::IRUnaryOp::CAST_OP_T;
            // type of cast todo
            switch (op_code)
            {
            case DEX::DVMTypes::Opcode::OP_INT_TO_CHAR:
                cast_type = MJOLNIR::IRUnaryOp::TO_CHAR;
                break;
            case DEX::DVMTypes::Opcode::OP_INT_TO_BYTE:
                cast_type = MJOLNIR::IRUnaryOp::TO_BYTE;
                break;
            case DEX::DVMTypes::Opcode::OP_INT_TO_SHORT:
                cast_type = MJOLNIR::IRUnaryOp::TO_SHORT;
                break;
            case DEX::DVMTypes::Opcode::OP_ADD_INT:
            case DEX::DVMTypes::Opcode::OP_SUB_INT:
            case DEX::DVMTypes::Opcode::OP_MUL_INT:
            case DEX::DVMTypes::Opcode::OP_DIV_INT:
            case DEX::DVMTypes::Opcode::OP_REM_INT:
            case DEX::DVMTypes::Opcode::OP_AND_INT:
            case DEX::DVMTypes::Opcode::OP_OR_INT:
            case DEX::DVMTypes::Opcode::OP_XOR_INT:
            case DEX::DVMTypes::Opcode::OP_SHL_INT:
            case DEX::DVMTypes::Opcode::OP_SHR_INT:
            case DEX::DVMTypes::Opcode::OP_USHR_INT:
            case DEX::DVMTypes::Opcode::OP_NEG_INT:
            case DEX::DVMTypes::Opcode::OP_NOT_INT:
            case DEX::DVMTypes::Opcode::OP_ADD_INT_LIT16:
            case DEX::DVMTypes::Opcode::OP_ADD_INT_LIT8:
            case DEX::DVMTypes::Opcode::OP_RSUB_INT:
            case DEX::DVMTypes::Opcode::OP_RSUB_INT_LIT8:
            case DEX::DVMTypes::Opcode::OP_MUL_INT_LIT16:
            case DEX::DVMTypes::Opcode::OP_MUL_INT_LIT8:
            case DEX::DVMTypes::Opcode::OP_DIV_INT_LIT16:
            case DEX::DVMTypes::Opcode::OP_DIV_INT_LIT8:
            case DEX::DVMTypes::Opcode::OP_REM_INT_LIT16:
            case DEX::DVMTypes::Opcode::OP_REM_INT_LIT8:
            case DEX::DVMTypes::Opcode::OP_AND_INT_LIT16:
            case DEX::DVMTypes::Opcode::OP_AND_INT_LIT8:
            case DEX::DVMTypes::Opcode::OP_OR_INT_LIT16:
            case DEX::DVMTypes::Opcode::OP_OR_INT_LIT8:
            case DEX::DVMTypes::Opcode::OP_XOR_INT_LIT16:
            case DEX::DVMTypes::Opcode::OP_XOR_INT_LIT8:
            case DEX::DVMTypes::Opcode::OP_SHL_INT_LIT8:
            case DEX::DVMTypes::Opcode::OP_SHR_INT_LIT8:
            case DEX::DVMTypes::Opcode::OP_USHR_INT_LIT8:
            case DEX::DVMTypes::Opcode::OP_LONG_TO_INT:
            case DEX::DVMTypes::Opcode::OP_FLOAT_TO_INT:
            case DEX::DVMTypes::Opcode::OP_DOUBLE_TO_INT:
                cast_type = MJOLNIR::IRUnaryOp::TO_INT;
                break;
            case DEX::DVMTypes::Opcode::OP_ADD_LONG:
            case DEX::DVMTypes::Opcode::OP_SUB_LONG:
            case DEX::DVMTypes::Opcode::OP_MUL_LONG:
            case DEX::DVMTypes::Opcode::OP_DIV_LONG:
            case DEX::DVMTypes::Opcode::OP_REM_LONG:
            case DEX::DVMTypes::Opcode::OP_AND_LONG:
            case DEX::DVMTypes::Opcode::OP_OR_LONG:
            case DEX::DVMTypes::Opcode::OP_XOR_LONG:
            case DEX::DVMTypes::Opcode::OP_SHL_LONG:
            case DEX::DVMTypes::Opcode::OP_SHR_LONG:
            case DEX::DVMTypes::Opcode::OP_USHR_LONG:
            case DEX::DVMTypes::Opcode::OP_NEG_LONG:
            case DEX::DVMTypes::Opcode::OP_NOT_LONG:
            case DEX::DVMTypes::Opcode::OP_INT_TO_LONG:
            case DEX::DVMTypes::Opcode::OP_FLOAT_TO_LONG:
            case DEX::DVMTypes::Opcode::OP_DOUBLE_TO_LONG:
                cast_type = MJOLNIR::IRUnaryOp::TO_LONG;
                break;
            case DEX::DVMTypes::Opcode::OP_ADD_FLOAT:
            case DEX::DVMTypes::Opcode::OP_SUB_FLOAT:
            case DEX::DVMTypes::Opcode::OP_MUL_FLOAT:
            case DEX::DVMTypes::Opcode::OP_DIV_FLOAT:
            case DEX::DVMTypes::Opcode::OP_REM_FLOAT:
            case DEX::DVMTypes::Opcode::OP_NEG_FLOAT:
            case DEX::DVMTypes::Opcode::OP_INT_TO_FLOAT:
            case DEX::DVMTypes::Opcode::OP_LONG_TO_FLOAT:
            case DEX::DVMTypes::Opcode::OP_DOUBLE_TO_FLOAT:
                cast_type = MJOLNIR::IRUnaryOp::TO_FLOAT;
                break;
            case DEX::DVMTypes::Opcode::OP_ADD_DOUBLE:
            case DEX::DVMTypes::Opcode::OP_SUB_DOUBLE:
            case DEX::DVMTypes::Opcode::OP_MUL_DOUBLE:
            case DEX::DVMTypes::Opcode::OP_DIV_DOUBLE:
            case DEX::DVMTypes::Opcode::OP_REM_DOUBLE:
            case DEX::DVMTypes::Opcode::OP_NEG_DOUBLE:
            case DEX::DVMTypes::Opcode::OP_INT_TO_DOUBLE:
            case DEX::DVMTypes::Opcode::OP_LONG_TO_DOUBLE:
            case DEX::DVMTypes::Opcode::OP_FLOAT_TO_DOUBLE:
                cast_type = MJOLNIR::IRUnaryOp::TO_DOUBLE;
                break;
            case DEX::DVMTypes::Opcode::OP_ADD_INT_2ADDR:
            case DEX::DVMTypes::Opcode::OP_SUB_INT_2ADDR:
            case DEX::DVMTypes::Opcode::OP_MUL_INT_2ADDR:
            case DEX::DVMTypes::Opcode::OP_DIV_INT_2ADDR:
            case DEX::DVMTypes::Opcode::OP_REM_INT_2ADDR:
            case DEX::DVMTypes::Opcode::OP_AND_INT_2ADDR:
            case DEX::DVMTypes::Opcode::OP_OR_INT_2ADDR:
            case DEX::DVMTypes::Opcode::OP_XOR_INT_2ADDR:
            case DEX::DVMTypes::Opcode::OP_SHL_INT_2ADDR:
            case DEX::DVMTypes::Opcode::OP_SHR_INT_2ADDR:
            case DEX::DVMTypes::Opcode::OP_USHR_INT_2ADDR:
            case DEX::DVMTypes::Opcode::OP_ADD_LONG_2ADDR:
            case DEX::DVMTypes::Opcode::OP_SUB_LONG_2ADDR:
            case DEX::DVMTypes::Opcode::OP_MUL_LONG_2ADDR:
            case DEX::DVMTypes::Opcode::OP_DIV_LONG_2ADDR:
            case DEX::DVMTypes::Opcode::OP_REM_LONG_2ADDR:
            case DEX::DVMTypes::Opcode::OP_AND_LONG_2ADDR:
            case DEX::DVMTypes::Opcode::OP_OR_LONG_2ADDR:
            case DEX::DVMTypes::Opcode::OP_XOR_LONG_2ADDR:
            case DEX::DVMTypes::Opcode::OP_SHL_LONG_2ADDR:
            case DEX::DVMTypes::Opcode::OP_SHR_LONG_2ADDR:
            case DEX::DVMTypes::Opcode::OP_USHR_LONG_2ADDR:
            case DEX::DVMTypes::Opcode::OP_ADD_FLOAT_2ADDR:
            case DEX::DVMTypes::Opcode::OP_SUB_FLOAT_2ADDR:
            case DEX::DVMTypes::Opcode::OP_MUL_FLOAT_2ADDR:
            case DEX::DVMTypes::Opcode::OP_DIV_FLOAT_2ADDR:
            case DEX::DVMTypes::Opcode::OP_REM_FLOAT_2ADDR:
            case DEX::DVMTypes::Opcode::OP_ADD_DOUBLE_2ADDR:
            case DEX::DVMTypes::Opcode::OP_SUB_DOUBLE_2ADDR:
            case DEX::DVMTypes::Opcode::OP_MUL_DOUBLE_2ADDR:
            case DEX::DVMTypes::Opcode::OP_DIV_DOUBLE_2ADDR:
            case DEX::DVMTypes::Opcode::OP_REM_DOUBLE_2ADDR:
                cast_type = MJOLNIR::IRUnaryOp::TO_ADDR;
                break;
            default:
                cast_type = MJOLNIR::IRUnaryOp::NONE_CAST;
                break;
            }

            if (androidinstructions.instruction23x_binary_instruction.find(op_code) != androidinstructions.instruction23x_binary_instruction.end())
            {
                auto instr = std::dynamic_pointer_cast<DEX::Instruction23x>(instruction);

                auto dest = instr->get_destination();
                auto src1 = instr->get_first_source();
                auto src2 = instr->get_second_source();

                auto dest_reg = make_android_register(dest);
                auto src1_reg = make_android_register(src1);
                auto src2_reg = make_android_register(src2);

                arith_logc_instr = std::make_shared<MJOLNIR::IRBinOp>(bin_operation, dest_reg, src1_reg, src2_reg, nullptr, nullptr);

                cast_instr = std::make_shared<MJOLNIR::IRUnaryOp>(cast_op, cast_type, dest_reg, dest_reg, nullptr, nullptr);
            }

            else if (androidinstructions.instruction12x_binary_instruction.find(op_code) != androidinstructions.instruction12x_binary_instruction.end())
            {
                auto instr = std::dynamic_pointer_cast<DEX::Instruction12x>(instruction);

                auto dest = instr->get_destination();
                auto src = instr->get_source();

                auto dest_reg = make_android_register(dest);
                auto src_reg = make_android_register(src);

                arith_logc_instr = std::make_shared<MJOLNIR::IRBinOp>(bin_operation, dest_reg, dest_reg, src_reg, nullptr, nullptr);
                cast_instr = std::make_shared<MJOLNIR::IRUnaryOp>(cast_op, cast_type, dest_reg, dest_reg, nullptr, nullptr);
            }

            else if (androidinstructions.instruction22s_binary_instruction.find(op_code) != androidinstructions.instruction22s_binary_instruction.end())
            {
                auto instr = std::dynamic_pointer_cast<DEX::Instruction22s>(instruction);

                auto dest = instr->get_destination();
                auto src1 = instr->get_source();
                auto src2 = instr->get_number();

                auto dest_reg = make_android_register(dest);
                auto src1_reg = make_android_register(src1);
                auto src2_num = make_int(src2, true, WORD_S);

                arith_logc_instr = std::make_shared<MJOLNIR::IRBinOp>(bin_operation, dest_reg, src1_reg, src2_num, nullptr, nullptr);
                cast_instr = std::make_shared<MJOLNIR::IRUnaryOp>(cast_op, cast_type, dest_reg, dest_reg, nullptr, nullptr);
            }

            else if (androidinstructions.instruction22b_binary_instruction.find(op_code) != androidinstructions.instruction22b_binary_instruction.end())
            {
                auto instr = std::dynamic_pointer_cast<DEX::Instruction22b>(instruction);

                auto dest = instr->get_destination();
                auto src1 = instr->get_source();
                auto src2 = instr->get_number();

                auto dest_reg = make_android_register(dest);
                auto src1_reg = make_android_register(src1);
                auto src2_num = make_int(src2, true, BYTE_S);

                arith_logc_instr = std::make_shared<MJOLNIR::IRBinOp>(bin_operation, dest_reg, src1_reg, src2_num, nullptr, nullptr);
                cast_instr = std::make_shared<MJOLNIR::IRUnaryOp>(cast_op, cast_type, dest_reg, dest_reg, nullptr, nullptr);
            }

            else if (androidinstructions.instruction12x_unary_instruction.find(op_code) != androidinstructions.instruction12x_unary_instruction.end())
            {
                auto instr = std::dynamic_pointer_cast<DEX::Instruction12x>(instruction);

                auto dest = instr->get_destination();
                auto src = instr->get_source();

                auto dest_reg = make_android_register(dest);
                auto src_reg = make_android_register(src);

                arith_logc_instr = std::make_shared<MJOLNIR::IRUnaryOp>(unary_operation, dest_reg, src_reg, nullptr, nullptr);
                cast_instr = std::make_shared<MJOLNIR::IRUnaryOp>(cast_op, cast_type, dest_reg, dest_reg, nullptr, nullptr);
            }

            else if (androidinstructions.cast_instruction.find(op_code) != androidinstructions.cast_instruction.end())
            {
                auto instr = std::dynamic_pointer_cast<DEX::Instruction12x>(instruction);

                auto dest = instr->get_destination();
                auto src = instr->get_source();

                auto dest_reg = make_android_register(dest);
                auto src_reg = make_android_register(src);

                arith_logc_instr = std::make_shared<MJOLNIR::IRUnaryOp>(unary_operation, cast_type, dest_reg, src_reg, nullptr, nullptr);
                // no cast instruction in cast instruction =D
            }

            bb->append_statement_to_block(arith_logc_instr);

            if (cast_instr != nullptr)
                bb->append_statement_to_block(cast_instr);
        }

        /**
         * @brief Generate a IRRet instruction which represent a ret instruction.
         * 
         * @param instruction: instruction to lift.
         * @param bb: basic block where to insert the instructions.
         * @return void
         */
        void LifterAndroid::lift_ret_instruction(std::shared_ptr<DEX::Instruction> instruction, std::shared_ptr<MJOLNIR::IRBlock> bb)
        {
            std::shared_ptr<MJOLNIR::IRRet> ret_instr;
            auto op_code = static_cast<DEX::DVMTypes::Opcode>(instruction->get_OP());

            if (op_code == DEX::DVMTypes::Opcode::OP_RETURN_VOID)
            {
                auto none = make_none_type();

                ret_instr = std::make_shared<MJOLNIR::IRRet>(none);
            }
            else
            {
                auto instr = std::dynamic_pointer_cast<DEX::Instruction11x>(instruction);

                auto reg = instr->get_destination();

                auto ir_reg = make_android_register(reg);

                ret_instr = std::make_shared<MJOLNIR::IRRet>(ir_reg);
            }

            bb->append_statement_to_block(ret_instr);
        }

        /**
         * @brief Generate a IRBComp instruction which represent a comparison between two values.
         * 
         * @param instruction: instruction to lift.
         * @param bb: basic block where to insert the instructions.
         * @return void
         */
        void LifterAndroid::lift_comparison_instruction(std::shared_ptr<DEX::Instruction> instruction, std::shared_ptr<MJOLNIR::IRBlock> bb)
        {
            auto op_code = static_cast<DEX::DVMTypes::Opcode>(instruction->get_OP());
            auto instr = std::dynamic_pointer_cast<DEX::Instruction23x>(instruction);
            std::shared_ptr<MJOLNIR::IRBComp> ir_comp;

            auto reg1 = make_android_register(instr->get_first_source());
            auto reg2 = make_android_register(instr->get_second_source());
            auto result = make_android_register(instr->get_destination());

            if (op_code == DEX::DVMTypes::Opcode::OP_CMPL_FLOAT ||
                op_code == DEX::DVMTypes::Opcode::OP_CMPG_FLOAT)
            {
                auto cast1 = std::make_shared<MJOLNIR::IRUnaryOp>(MJOLNIR::IRUnaryOp::CAST_OP_T, MJOLNIR::IRUnaryOp::TO_FLOAT, reg1, reg1, nullptr, nullptr);
                auto cast2 = std::make_shared<MJOLNIR::IRUnaryOp>(MJOLNIR::IRUnaryOp::CAST_OP_T, MJOLNIR::IRUnaryOp::TO_FLOAT, reg2, reg2, nullptr, nullptr);

                if (op_code == DEX::DVMTypes::Opcode::OP_CMPL_FLOAT)
                    ir_comp = std::make_shared<MJOLNIR::IRBComp>(MJOLNIR::IRBComp::LOWER_T, result, reg1, reg2, nullptr, nullptr);
                else
                    ir_comp = std::make_shared<MJOLNIR::IRBComp>(MJOLNIR::IRBComp::GREATER_T, result, reg1, reg2, nullptr, nullptr);

                bb->append_statement_to_block(cast1);
                bb->append_statement_to_block(cast2);
                bb->append_statement_to_block(ir_comp);
            }
            else if (op_code == DEX::DVMTypes::Opcode::OP_CMPL_DOUBLE ||
                     op_code == DEX::DVMTypes::Opcode::OP_CMPG_DOUBLE)
            {
                auto cast1 = std::make_shared<MJOLNIR::IRUnaryOp>(MJOLNIR::IRUnaryOp::CAST_OP_T, MJOLNIR::IRUnaryOp::TO_DOUBLE, reg1, reg1, nullptr, nullptr);
                auto cast2 = std::make_shared<MJOLNIR::IRUnaryOp>(MJOLNIR::IRUnaryOp::CAST_OP_T, MJOLNIR::IRUnaryOp::TO_DOUBLE, reg2, reg2, nullptr, nullptr);

                if (op_code == DEX::DVMTypes::Opcode::OP_CMPL_DOUBLE)
                    ir_comp = std::make_shared<MJOLNIR::IRBComp>(MJOLNIR::IRBComp::LOWER_T, result, reg1, reg2, nullptr, nullptr);
                else
                    ir_comp = std::make_shared<MJOLNIR::IRBComp>(MJOLNIR::IRBComp::GREATER_T, result, reg1, reg2, nullptr, nullptr);

                bb->append_statement_to_block(cast1);
                bb->append_statement_to_block(cast2);
                bb->append_statement_to_block(ir_comp);
            }
            else
            {
                auto cast1 = std::make_shared<MJOLNIR::IRUnaryOp>(MJOLNIR::IRUnaryOp::CAST_OP_T, MJOLNIR::IRUnaryOp::TO_LONG, reg1, reg1, nullptr, nullptr);
                auto cast2 = std::make_shared<MJOLNIR::IRUnaryOp>(MJOLNIR::IRUnaryOp::CAST_OP_T, MJOLNIR::IRUnaryOp::TO_LONG, reg2, reg2, nullptr, nullptr);
                ir_comp = std::make_shared<MJOLNIR::IRBComp>(MJOLNIR::IRBComp::EQUAL_T, result, reg1, reg2, nullptr, nullptr);

                bb->append_statement_to_block(cast1);
                bb->append_statement_to_block(cast2);
                bb->append_statement_to_block(ir_comp);
            }
        }

        /**
         * @brief Generate a IRCJmp instruction which represent a conditional jump.
         * 
         * @param instruction: instruction to lift.
         * @param bb: basic block where to insert the instructions.
         * @return void
         */
        void LifterAndroid::lift_conditional_jump_instruction(std::shared_ptr<DEX::Instruction> instruction, std::shared_ptr<MJOLNIR::IRBlock> bb)
        {
            auto op_code = static_cast<DEX::DVMTypes::Opcode>(instruction->get_OP());

            if (androidinstructions.jcc_instruction22t.find(op_code) != androidinstructions.jcc_instruction22t.end())
            {
                auto instr = std::dynamic_pointer_cast<DEX::Instruction22t>(instruction);
                auto temp_reg = make_temporal_register();
                auto reg1 = make_android_register(instr->get_first_check_reg());
                auto reg2 = make_android_register(instr->get_second_check_reg());
                MJOLNIR::IRBComp::comp_t comparison;

                uint64_t target = current_idx + (instr->get_ref() * 2) + instruction->get_length();

                switch (op_code)
                {
                case DEX::DVMTypes::Opcode::OP_IF_EQ:
                    comparison = MJOLNIR::IRBComp::EQUAL_T;
                    break;
                case DEX::DVMTypes::Opcode::OP_IF_NE:
                    comparison = MJOLNIR::IRBComp::NOT_EQUAL_T;
                    break;
                case DEX::DVMTypes::Opcode::OP_IF_LT:
                    comparison = MJOLNIR::IRBComp::LOWER_T;
                    break;
                case DEX::DVMTypes::Opcode::OP_IF_GE:
                    comparison = MJOLNIR::IRBComp::GREATER_EQUAL_T;
                    break;
                case DEX::DVMTypes::Opcode::OP_IF_GT:
                    comparison = MJOLNIR::IRBComp::GREATER_T;
                    break;
                case DEX::DVMTypes::Opcode::OP_IF_LE:
                    comparison = MJOLNIR::IRBComp::LOWER_EQUAL_T;
                    break;
                }

                auto bcomp = std::make_shared<MJOLNIR::IRBComp>(comparison, temp_reg, reg1, reg2, nullptr, nullptr);
                auto ir_ucond = std::make_shared<MJOLNIR::IRCJmp>(target, temp_reg, nullptr, nullptr);

                bb->append_statement_to_block(bcomp);
                bb->append_statement_to_block(ir_ucond);
            }
            else if (androidinstructions.jcc_instruction21t.find(op_code) != androidinstructions.jcc_instruction21t.end())
            {
                auto instr = std::dynamic_pointer_cast<DEX::Instruction21t>(instruction);
                auto temp_reg = make_temporal_register();
                auto reg = make_android_register(instr->get_check_reg());
                MJOLNIR::IRZComp::zero_comp_t comparison;

                uint64_t target = current_idx + (instr->get_ref() * 2)  + instruction->get_length();

                switch (op_code)
                {
                case DEX::DVMTypes::Opcode::OP_IF_EQZ:
                    comparison = MJOLNIR::IRZComp::EQUAL_ZERO_T;
                    break;
                case DEX::DVMTypes::Opcode::OP_IF_NEZ:
                    comparison = MJOLNIR::IRZComp::NOT_EQUAL_ZERO_T;
                    break;
                case DEX::DVMTypes::Opcode::OP_IF_LTZ:
                    comparison = MJOLNIR::IRZComp::LOWER_ZERO_T;
                    break;
                case DEX::DVMTypes::Opcode::OP_IF_GEZ:
                    comparison = MJOLNIR::IRZComp::GREATER_EQUAL_ZERO;
                    break;
                case DEX::DVMTypes::Opcode::OP_IF_GTZ:
                    comparison = MJOLNIR::IRZComp::GREATER_ZERO_T;
                    break;
                case DEX::DVMTypes::Opcode::OP_IF_LEZ:
                    comparison = MJOLNIR::IRZComp::LOWER_EQUAL_ZERO;
                    break;
                }

                auto zcomp = std::make_shared<MJOLNIR::IRZComp>(comparison, temp_reg, reg, nullptr, nullptr);
                auto ir_ucond = std::make_shared<MJOLNIR::IRCJmp>(target, temp_reg, nullptr, nullptr);
                bb->append_statement_to_block(zcomp);
                bb->append_statement_to_block(ir_ucond);
            }
        }

        /**
         * @brief Generate a IRUJmp instruction which represent an unconditional jump.
         * 
         * @param instruction 
         * @param bb 
         * @return void
         */
        void LifterAndroid::lift_unconditional_jump_instruction(std::shared_ptr<DEX::Instruction> instruction, std::shared_ptr<MJOLNIR::IRBlock> bb)
        {
            auto op_code = static_cast<DEX::DVMTypes::Opcode>(instruction->get_OP());
            std::shared_ptr<MJOLNIR::IRUJmp> ujmp = nullptr;

            if (DEX::DVMTypes::Opcode::OP_GOTO == op_code)
            {
                auto jmp = std::dynamic_pointer_cast<DEX::Instruction10t>(instruction);

                auto addr = current_idx + (jmp->get_offset() * 2) + instruction->get_length();

                ujmp = std::make_shared<MJOLNIR::IRUJmp>(addr, nullptr);
            }
            else if (DEX::DVMTypes::Opcode::OP_GOTO_16 == op_code)
            {
                auto jmp = std::dynamic_pointer_cast<DEX::Instruction20t>(instruction);

                auto addr = current_idx + (jmp->get_offset() * 2) + instruction->get_length();

                ujmp = std::make_shared<MJOLNIR::IRUJmp>(addr, nullptr);
            }
            else if (DEX::DVMTypes::Opcode::OP_GOTO_32 == op_code)
            {
                auto jmp = std::dynamic_pointer_cast<DEX::Instruction30t>(instruction);

                auto addr = current_idx + (jmp->get_offset() * 2) + instruction->get_length();

                ujmp = std::make_shared<MJOLNIR::IRUJmp>(addr, nullptr);
            }

            if (ujmp != nullptr)
                bb->append_statement_to_block(ujmp);
        }

        /**
         * @brief Generate the IRCall instruction which represent any kind of call function/method instruction.
         * 
         * @param instruction 
         * @param bb 
         */
        void LifterAndroid::lift_call_instruction(std::shared_ptr<DEX::Instruction> instruction, std::shared_ptr<MJOLNIR::IRBlock> bb)
        {
            auto op_code = static_cast<DEX::DVMTypes::Opcode>(instruction->get_OP());
            std::shared_ptr<MJOLNIR::IRCall> call = nullptr;
            std::shared_ptr<MJOLNIR::IRExpr> callee = nullptr;

            if (androidinstructions.call_instruction35c.find(op_code) != androidinstructions.call_instruction35c.end())
            {
                auto call_inst = std::dynamic_pointer_cast<DEX::Instruction35c>(instruction);
                std::vector<std::shared_ptr<MJOLNIR::IRExpr>> parameters;

                size_t p_size = call_inst->get_array_size();

                for (size_t i = 0; i < p_size; i++)
                    parameters.push_back(make_android_register(call_inst->get_operand_register(i)));

                // as it is a call to a method, we can safely retrieve the operand as method
                auto method_called = call_inst->get_operands_kind_method();

                std::string method_name = *method_called->get_method_name();
                std::string class_name = reinterpret_cast<DEX::Class *>(method_called->get_method_class())->get_name();
                std::string proto = method_called->get_method_prototype()->get_proto_str();

                callee = std::make_shared<MJOLNIR::IRCallee>(0, method_name, class_name, p_size, proto, class_name + "->" + method_name + proto, ADDR_S);
                call = std::make_shared<MJOLNIR::IRCall>(callee, parameters, nullptr, nullptr);
            }

            if (call)
                bb->append_statement_to_block(call);
        }

        /**
         * @brief Fix for every jump instruction at the end of a basic block,
         *        its target and in case of conditional jump its fallthrough,
         *        this will give for each one, the basic block where it points
         *        to.
         * 
         * @param bbs: basic blocks from the method.
         */
        void LifterAndroid::jump_target_analysis(std::vector<std::shared_ptr<KUNAI::DEX::DVMBasicBlock>> bbs)
        {
            for (auto bb : bbs)
            {
                auto next_bbs = bb->get_next();

                auto current_bb = lifted_blocks[bb];

                if (current_bb->get_number_of_statements() == 0) // security check
                    continue;

                // now set some interesting stuff for instructions.
                // now set some interesting stuff for instructions.
                auto last_instr = current_bb->get_statements().back();

                if (MJOLNIR::is_unconditional_jump(last_instr))
                {
                    auto jmp = std::dynamic_pointer_cast<MJOLNIR::IRUJmp>(last_instr);

                    if (next_bbs.size() == 1)
                    {
                        auto block = std::get<2>(next_bbs[0]);
                        jmp->set_jump_target(lifted_blocks[block]);
                    }
                }
                else if (MJOLNIR::is_conditional_jump(last_instr))
                {
                    auto jcc = std::dynamic_pointer_cast<MJOLNIR::IRCJmp>(last_instr);

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

        /**
         * @brief Set for a call instruction its return register.
         *        this will be just for those cases where the current
         *        instruction is some kind of move_result and then
         *        the previos instruction is a call.
         * 
         * @param instruction 
         * @param call 
         */
        void LifterAndroid::lift_move_result_instruction(std::shared_ptr<DEX::Instruction> instruction, std::shared_ptr<MJOLNIR::IRCall> call)
        {
            auto move_result = std::dynamic_pointer_cast<DEX::Instruction11x>(instruction);

            call->set_ret_val(make_android_register(move_result->get_destination()));

            return;
        }

        /**
         * @brief Analyze the basic blocks of the graph in order to create the
         *        fallthrough edges between blocks which are from conditional
         *        jumps, these has no edges so for the moment one block looks
         *        like goes nowehere:
         * 
         *                          +----------------+
         *                          |                |
         *                          |  jcc           |
         *                          +----------------+
         *           fallthrough   /                  \  target
         *                        /                    \
         *               +----------------+        +----------------+
         *               | points         |        |                |
         *               | nowhere        |        |  jmp           |
         *               +----------------+        +----------------+
         *                                               |
         *                                               |
         *                                         +----------------+
         *                                         |                |
         *                                         |                |
         *                                         +----------------+
         * 
         *         The one on the left points nowhere but this is the real previous
         *         block before the last block, but last block is divided because
         *         there's a jump to it, so we will create an edge between, the one
         *         on the left, and the last one, as it's a continuation.
         * 
         *                          +----------------+
         *                          |                |
         *                          |  jcc           |
         *                          +----------------+
         *           fallthrough   /                  \  target
         *                        /                    \
         *               +----------------+        +----------------+
         *               | points         |        |                |
         *               | nowhere        |        |  jmp           |
         *               +----------------+        +----------------+
         *                      \                         /
         *                       \                       /
         *                        \                     /
         *                         \                   /
         *                          \                 /
         *                           +----------------+
         *                           |                |
         *                           |                |
         *                           +----------------+
         * @param ir_graph 
         */
        void LifterAndroid::fallthrough_target_analysis(std::shared_ptr<MJOLNIR::IRGraph> ir_graph)
        {
            auto nodes = ir_graph->get_nodes();

            for (auto node : nodes)
            {

                if (node->get_number_of_statements() == 0) // security check
                    continue;

                auto last_inst = node->get_statements().back();

                if (MJOLNIR::is_conditional_jump(last_inst) || MJOLNIR::is_unconditional_jump(last_inst) || MJOLNIR::is_ret(last_inst))
                    continue;

                for (auto aux : nodes)
                {
                    if (node->get_end_idx() == aux->get_start_idx())
                    {
                        ir_graph->add_uniq_edge(node, aux);
                    }
                }
            }
        }

    }
}