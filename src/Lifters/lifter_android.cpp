#include "lifter_android.hpp"

namespace KUNAI
{
    namespace LIFTER
    {
        // Assignment instructions of type Instruction12x
        std::set<DEX::DVMTypes::Opcode> assignment_instruction12x = {DEX::DVMTypes::Opcode::OP_MOVE, DEX::DVMTypes::Opcode::OP_MOVE_WIDE, DEX::DVMTypes::Opcode::OP_MOVE_OBJECT};
        // Assignment instructions of type Instruction22x
        std::set<DEX::DVMTypes::Opcode> assignment_instruction22x = {DEX::DVMTypes::Opcode::OP_MOVE_FROM16, DEX::DVMTypes::Opcode::OP_MOVE_WIDE_FROM16, DEX::DVMTypes::Opcode::OP_MOVE_OBJECT_FROM16};
        // Assignment instructions of type Instruction32x
        std::set<DEX::DVMTypes::Opcode> assignment_instruction32x = {DEX::DVMTypes::Opcode::OP_MOVE_16, DEX::DVMTypes::Opcode::OP_MOVE_WIDE_16, DEX::DVMTypes::Opcode::OP_MOVE_OBJECT_16};
        // Assignment instructions of type Instruction11x
        std::set<DEX::DVMTypes::Opcode> assignment_instruction11x = {DEX::DVMTypes::Opcode::OP_MOVE_RESULT, DEX::DVMTypes::Opcode::OP_MOVE_RESULT_WIDE, DEX::DVMTypes::Opcode::OP_MOVE_RESULT_OBJECT, DEX::DVMTypes::Opcode::OP_MOVE_EXCEPTION, DEX::DVMTypes::Opcode::OP_RETURN, DEX::DVMTypes::Opcode::OP_RETURN_WIDE, DEX::DVMTypes::Opcode::OP_RETURN_OBJECT};
        // Assignment instructions of type Instruction11n
        std::set<DEX::DVMTypes::Opcode> assigment_instruction11n = {DEX::DVMTypes::Opcode::OP_CONST_4};
        // Assignment instructions of type Instruction21s
        std::set<DEX::DVMTypes::Opcode> assigment_instruction21s = {DEX::DVMTypes::Opcode::OP_CONST_16, DEX::DVMTypes::Opcode::OP_CONST_WIDE_16};
        // Assignment instructions of type Instruction31i
        std::set<DEX::DVMTypes::Opcode> assigment_instruction31i = {DEX::DVMTypes::Opcode::OP_CONST, DEX::DVMTypes::Opcode::OP_CONST_WIDE_32};
        // Assignment instructions of type Instruction21h
        std::set<DEX::DVMTypes::Opcode> assigment_instruction21h = {DEX::DVMTypes::Opcode::OP_CONST_HIGH16, DEX::DVMTypes::Opcode::OP_CONST_WIDE_HIGH16};
        // Assignment instructions of type Instruction51l
        std::set<DEX::DVMTypes::Opcode> assigment_instruction51l = {DEX::DVMTypes::Opcode::OP_CONST_WIDE};
        // Assignment instructions of type Instruction21c
        std::set<DEX::DVMTypes::Opcode> assigment_instruction21c = {DEX::DVMTypes::Opcode::OP_CONST_STRING, DEX::DVMTypes::Opcode::OP_CONST_CLASS};
        // Assignment instructions of type Instruction31c
        std::set<DEX::DVMTypes::Opcode> assigment_instruction31c = {DEX::DVMTypes::Opcode::OP_CONST_STRING_JUMBO};
        /**
         * @brief Create a new IRReg with the id of the register,
         *        using different objects will be useful for having
         *        different registers in different methods.
         * @param reg_id: id of the register to create.
         * @return std::shared_ptr<MJOLNIR::IRReg>
         */
        std::shared_ptr<MJOLNIR::IRReg> LifterAndroid::make_android_register(std::uint32_t reg_id)
        {
            return std::make_shared<MJOLNIR::IRReg>(reg_id, "v" + std::to_string(reg_id), DWORD);
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
        std::shared_ptr<MJOLNIR::IRConstInt> make_int(std::uint64_t value, bool is_signed, size_t type_size)
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
        std::shared_ptr<MJOLNIR::IRClass> LifterAndroid::make_class(DEX::Class* value)
        {
            return std::make_shared<MJOLNIR::IRClass>(value->get_name(), value->get_name(), 0);
        }


        std::shared_ptr<MJOLNIR::IRExpr> LifterAndroid::lift_assignment_instruction(std::shared_ptr<DEX::Instruction> instruction)
        {
            std::shared_ptr<MJOLNIR::IRExpr> assignment_instr;
            auto op_code = static_cast<DEX::DVMTypes::Opcode>(instruction->get_OP());

            // Instruction12x types
            if (assignment_instruction12x.find(op_code) != assignment_instruction12x.end())
            {
                auto instr = std::dynamic_pointer_cast<DEX::Instruction12x>(instruction);
                auto dest = instr->get_destination();
                auto src = instr->get_source();

                auto dest_reg = make_android_register(dest);
                auto src_reg = make_android_register(src);

                assignment_instr = std::make_shared<MJOLNIR::IRAssign>(dest_reg, src_reg, nullptr, nullptr);
            }
            // Instruction22x types
            else if (assignment_instruction22x.find(op_code) != assignment_instruction22x.end())
            {
                auto instr = std::dynamic_pointer_cast<DEX::Instruction22x>(instruction);
                auto dest = instr->get_destination();
                auto src = instr->get_source();

                auto dest_reg = make_android_register(dest);
                auto src_reg = make_android_register(src);

                assignment_instr = std::make_shared<MJOLNIR::IRAssign>(dest_reg, src_reg, nullptr, nullptr);
            }
            // Instruction32x types
            else if (assignment_instruction32x.find(op_code) != assignment_instruction32x.end())
            {
                auto instr = std::dynamic_pointer_cast<DEX::Instruction32x>(instruction);

                auto dest = instr->get_destination();
                auto src = instr->get_source();

                auto dest_reg = make_android_register(dest);
                auto src_reg = make_android_register(src);

                assignment_instr = std::make_shared<MJOLNIR::IRAssign>(dest_reg, src_reg, nullptr, nullptr);
            }
            // Instruction11x types
            else if (assignment_instruction11x.find(op_code) != assignment_instruction11x.end())
            {
                auto instr = std::dynamic_pointer_cast<DEX::Instruction11x>(instruction);

                auto dest = instr->get_destination();

                auto dest_reg = make_android_register(dest);
                auto none = make_none_type();

                assignment_instr = std::make_shared<MJOLNIR::IRAssign>(dest_reg, none, nullptr, nullptr);
            }
            // Instruction11n types
            else if (assigment_instruction11n.find(op_code) != assigment_instruction11n.end())
            {
                auto instr = std::dynamic_pointer_cast<DEX::Instruction11n>(instruction);

                auto dest = instr->get_destination();
                auto src = instr->get_source();

                auto dest_reg = make_android_register(dest);
                auto src_int = make_int(src, true, NIBBLE);

                assignment_instr = std::make_shared<MJOLNIR::IRAssign>(dest_reg, src_int, nullptr, nullptr);
            }
            // Instruction21s types
            else if (assigment_instruction21s.find(op_code) != assigment_instruction21s.end())
            {
                auto instr = std::dynamic_pointer_cast<DEX::Instruction21s>(instruction);

                auto dest = instr->get_destination();
                auto src = instr->get_source();

                auto dest_reg = make_android_register(dest);
                auto src_int = make_int(src, true, WORD);

                assignment_instr = std::make_shared<MJOLNIR::IRAssign>(dest_reg, src_int, nullptr, nullptr);
            }
            // Instrution31i types
            else if (assigment_instruction31i.find(op_code) != assigment_instruction31i.end())
            {
                auto instr = std::dynamic_pointer_cast<DEX::Instruction31i>(instruction);   

                auto dest = instr->get_destination();
                auto src = instr->get_source();

                auto dest_reg = make_android_register(dest);
                auto src_int = make_int(src, true, DWORD);

                assignment_instr = std::make_shared<MJOLNIR::IRAssign>(dest_reg, src_int, nullptr, nullptr);
            }
            // Instruction21h types
            else if (assigment_instruction21h.find(op_code) != assigment_instruction21h.end())
            {
                auto instr = std::dynamic_pointer_cast<DEX::Instruction21h>(instruction);

                auto dest = instr->get_destination();
                auto src = instr->get_source();

                auto dest_reg = make_android_register(dest);
                std::shared_ptr<KUNAI::MJOLNIR::IRConstInt> src_int = nullptr;

                if (op_code == DEX::DVMTypes::Opcode::OP_CONST_WIDE_16)
                    src_int = make_int(src << 16, true, QWORD);
                else if (op_code == DEX::DVMTypes::Opcode::OP_CONST_WIDE_HIGH16)
                    src_int = make_int(src << 48, true, QWORD);
                

                assignment_instr = std::make_shared<MJOLNIR::IRAssign>(dest_reg, src_int, nullptr, nullptr);
            }
            // Instruction 51l types
            else if (assigment_instruction51l.find(op_code) != assigment_instruction51l.end())
            {
                auto instr = std::dynamic_pointer_cast<DEX::Instruction51l>(instruction);   

                auto dest = instr->get_destination();
                auto src = instr->get_source();

                auto dest_reg = make_android_register(dest);
                auto src_int = make_int(src, true, QWORD);

                assignment_instr = std::make_shared<MJOLNIR::IRAssign>(dest_reg, src_int, nullptr, nullptr);
            }
            // Instruction21c types
            else if (assigment_instruction21c.find(op_code) != assigment_instruction21c.end())
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
                    auto src_class = dynamic_cast<DEX::Class*>(instr->get_source_typeid());
                    auto src = make_class(src_class);

                    assignment_instr = std::make_shared<MJOLNIR::IRAssign>(dest_reg, src, nullptr, nullptr);
                }
            }
            // Instruction31c types
            else if (assigment_instruction31c.find(op_code) != assigment_instruction31c.end())
            {
                auto instr = std::dynamic_pointer_cast<DEX::Instruction31c>(instruction);

                auto dest = instr->get_destination();
                auto dest_reg = make_android_register(dest);

                auto src = make_str(*instr->get_source_str());
                assignment_instr = std::make_shared<MJOLNIR::IRAssign>(dest_reg, src, nullptr, nullptr);
            }
        
            return assignment_instr;
        }
    }
}