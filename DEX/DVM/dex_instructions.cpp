#include "dex_instructions.hpp"

namespace KUNAI
{
    namespace DEX
    {

        /**
         * Instruction
         */
        Instruction::Instruction(std::shared_ptr<DalvikOpcodes> dalvik_opcodes, std::istream &input_file)
        {
            this->dalvik_opcodes = dalvik_opcodes;
            this->length = 0;
            this->OP = 0;
        }

        Instruction::~Instruction() {}

        DVMTypes::Kind Instruction::get_kind()
        {
            return dalvik_opcodes->get_instruction_type(OP);
        }

        std::string Instruction::get_translated_kind()
        {
            return dalvik_opcodes->get_instruction_type_str(OP);
        }

        std::string Instruction::get_name()
        {
            return dalvik_opcodes->get_instruction_name(OP);
        }

        std::uint32_t Instruction::get_length()
        {
            return length;
        }

        std::uint32_t Instruction::get_OP()
        {
            return OP;
        }

        void Instruction::set_length(std::uint32_t length)
        {
            this->length = length;
        }

        void Instruction::set_OP(std::uint32_t OP)
        {
            this->OP = OP;
        }

        void Instruction::show_instruction()
        {
            std::cout << std::left << std::setfill(' ') << std::setw(25) << get_name() << get_output();
        }

        std::string Instruction::get_output()
        {
            return "";
        }

        std::uint64_t Instruction::get_raw()
        {
            return this->OP;
        }

        std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> Instruction::get_operands()
        {
            std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> operands;

            return operands;
        }

        std::shared_ptr<DalvikOpcodes> Instruction::get_dalvik_opcodes()
        {
            return dalvik_opcodes;
        }
        /**
         * Instruction00x
         */
        Instruction00x::Instruction00x(std::shared_ptr<DalvikOpcodes> dalvik_opcodes, std::istream &input_file) : Instruction(dalvik_opcodes, input_file)
        {
            this->set_length(0);
        }

        Instruction00x::~Instruction00x() {}
        /**
         * Instruction10x
         */
        Instruction10x::Instruction10x(std::shared_ptr<DalvikOpcodes> dalvik_opcodes, std::istream &input_file) : Instruction(dalvik_opcodes, input_file)
        {
            std::uint8_t instruction[2];
            this->set_length(2);

            if (!KUNAI::read_data_file<std::uint8_t[2]>(instruction, this->get_length(), input_file))
                throw exceptions::DisassemblerException("Error disassembling Instruction10x");

            if (instruction[1] != 0)
                throw exceptions::InvalidInstruction("Instruction10x high byte should be 0");

            this->set_OP(instruction[0]);
        }

        Instruction10x::~Instruction10x() {}

        std::uint64_t Instruction10x::get_raw()
        {
            return this->get_OP();
        }
        /**
         * Instruction12x
         */
        Instruction12x::Instruction12x(std::shared_ptr<DalvikOpcodes> dalvik_opcodes, std::istream &input_file) : Instruction(dalvik_opcodes, input_file)
        {
            std::uint8_t instruction[2];
            this->set_length(2);

            if (!KUNAI::read_data_file<std::uint8_t[2]>(instruction, this->get_length(), input_file))
                throw exceptions::DisassemblerException("Error disassembling Instruction12x");

            this->set_OP(instruction[0]);
            this->vA = (instruction[1] & 0x0F);
            this->vB = (instruction[1] & 0xF0) >> 4;
        }

        Instruction12x::~Instruction12x() {}

        std::string Instruction12x::get_output()
        {
            return "v" + std::to_string(vA) + ", v" + std::to_string(vB);
        }

        std::uint64_t Instruction12x::get_raw()
        {
            return (get_OP() | vA << 8 | vB << 12);
        }

        std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> Instruction12x::get_operands()
        {
            std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> operands = {
                {DVMTypes::Operand::REGISTER, vA},
                {DVMTypes::Operand::REGISTER, vB}};

            return operands;
        }

        DVMTypes::Operand Instruction12x::get_source_type()
        {
            return DVMTypes::Operand::REGISTER;
        }

        std::uint8_t Instruction12x::get_source()
        {
            return vB;
        }

        DVMTypes::Operand Instruction12x::get_destination_type()
        {
            return DVMTypes::Operand::REGISTER;
        }

        std::uint8_t Instruction12x::get_destination()
        {
            return vA;
        }
        /**
         * Instruction11n
         */
        Instruction11n::Instruction11n(std::shared_ptr<DalvikOpcodes> dalvik_opcodes, std::istream &input_file) : Instruction(dalvik_opcodes, input_file)
        {
            std::uint8_t instruction[2];
            this->set_length(2);

            if (!KUNAI::read_data_file<std::uint8_t[2]>(instruction, this->get_length(), input_file))
                throw exceptions::DisassemblerException("Error disassembling Instruction11n");

            this->set_OP(instruction[0]);
            this->vA = instruction[1] & 0xF;
            this->nB = (instruction[1] & 0xF0) >> 4;
        }

        Instruction11n::~Instruction11n() {}

        std::string Instruction11n::get_output()
        {
            return "v" + std::to_string(vA) + ", " + std::to_string(nB);
        }

        std::uint64_t Instruction11n::get_raw()
        {
            return (get_OP() | vA << 8 | nB << 12);
        }

        std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> Instruction11n::get_operands()
        {
            std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> operands = {
                {DVMTypes::Operand::REGISTER, vA},
                {DVMTypes::Operand::LITERAL, nB}};

            return operands;
        }

        DVMTypes::Operand Instruction11n::get_source_type()
        {
            return DVMTypes::Operand::LITERAL;
        }

        std::int8_t Instruction11n::get_source()
        {
            return nB;
        }

        DVMTypes::Operand Instruction11n::get_destination_type()
        {
            return DVMTypes::Operand::REGISTER;
        }

        std::uint8_t Instruction11n::get_destination()
        {
            return vA;
        }
        /**
         * Instruction11x
         */
        Instruction11x::Instruction11x(std::shared_ptr<DalvikOpcodes> dalvik_opcodes, std::istream &input_file) : Instruction(dalvik_opcodes, input_file)
        {
            std::uint8_t instruction[2];
            this->set_length(2);

            if (!KUNAI::read_data_file<std::uint8_t[2]>(instruction, this->get_length(), input_file))
                throw exceptions::DisassemblerException("Error disassembling Instruction32x");

            this->set_OP(instruction[0]);
            this->vAA = instruction[1];
        }

        Instruction11x::~Instruction11x() {}

        std::string Instruction11x::get_output()
        {
            return "v" + std::to_string(vAA);
        }

        std::uint64_t Instruction11x::get_raw()
        {
            return (get_OP() | vAA << 8);
        }

        std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> Instruction11x::get_operands()
        {
            std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> operands = {
                {DVMTypes::Operand::REGISTER, vAA}};

            return operands;
        }

        DVMTypes::Operand Instruction11x::get_destination_type()
        {
            return DVMTypes::Operand::REGISTER;
        }

        std::uint8_t Instruction11x::get_destination()
        {
            return vAA;
        }
        /***
         * Instruction10t
         */
        Instruction10t::Instruction10t(std::shared_ptr<DalvikOpcodes> dalvik_opcodes, std::istream &input_file) : Instruction(dalvik_opcodes, input_file)
        {
            std::uint8_t instruction[2];
            this->set_length(2);

            if (!KUNAI::read_data_file<std::uint8_t[2]>(instruction, this->get_length(), input_file))
                throw exceptions::DisassemblerException("Error disassembling Instruction10t");

            if (instruction[1] == 0)
                throw exceptions::InvalidInstruction("Error reading Instruction10t offset cannot be 0");

            this->set_OP(instruction[0]);
            this->nAA = static_cast<std::int8_t>(instruction[1]);
        }

        Instruction10t::~Instruction10t() {}

        std::string Instruction10t::get_output()
        {
            return std::to_string(nAA);
        }

        std::uint64_t Instruction10t::get_raw()
        {
            return get_OP() | nAA << 8;
        }

        std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> Instruction10t::get_operands()
        {
            std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> operands = {
                {DVMTypes::Operand::OFFSET, nAA}};

            return operands;
        }

        DVMTypes::Operand Instruction10t::get_offset_type()
        {
            return DVMTypes::Operand::OFFSET;
        }

        std::int8_t Instruction10t::get_offset()
        {
            return nAA;
        }
        /**
         * Instruction20t
         */
        Instruction20t::Instruction20t(std::shared_ptr<DalvikOpcodes> dalvik_opcodes, std::istream &input_file) : Instruction(dalvik_opcodes, input_file)
        {
            std::uint8_t instruction[4];
            this->set_length(4);

            if (!KUNAI::read_data_file<std::uint8_t[4]>(instruction, this->get_length(), input_file))
                throw exceptions::DisassemblerException("Error disassembling Instruction20t");

            if (instruction[1] != 0)
                throw exceptions::InvalidInstruction("Error reading Instruction20t padding must be 0");

            this->set_OP(instruction[0]);
            this->nAAAA = *(reinterpret_cast<std::uint16_t *>(&instruction[2]));

            if (this->nAAAA == 0)
                throw exceptions::InvalidInstruction("Error reading Instruction20t offset cannot be 0");
        }

        Instruction20t::~Instruction20t() {}

        std::string Instruction20t::get_output()
        {
            return std::to_string(nAAAA);
        }

        std::uint64_t Instruction20t::get_raw()
        {
            return static_cast<std::uint64_t>(get_OP()) | static_cast<std::uint64_t>(nAAAA) << 16;
        }

        std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> Instruction20t::get_operands()
        {
            std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> operands = {
                {DVMTypes::Operand::OFFSET, nAAAA}};

            return operands;
        }

        DVMTypes::Operand Instruction20t::get_offset_type()
        {
            return DVMTypes::Operand::OFFSET;
        }

        std::int16_t Instruction20t::get_offset()
        {
            return nAAAA;
        }
        /**
         * Instruction20bc
         */
        Instruction20bc::Instruction20bc(std::shared_ptr<DalvikOpcodes> dalvik_opcodes, std::istream &input_file) : Instruction(dalvik_opcodes, input_file)
        {
            std::uint8_t instruction[4];
            this->set_length(4);

            if (!KUNAI::read_data_file<std::uint8_t[4]>(instruction, this->get_length(), input_file))
                throw exceptions::DisassemblerException("Error disassembling Instruction20bc");

            this->set_OP(instruction[0]);
            this->nAA = instruction[1];
            this->nBBBB = *(reinterpret_cast<std::uint16_t *>(&instruction[2]));
        }

        Instruction20bc::~Instruction20bc() {}

        std::string Instruction20bc::get_output()
        {
            return std::to_string(nBBBB) + ", " + std::to_string(nAA);
        }

        std::uint64_t Instruction20bc::get_raw()
        {
            return get_OP() | nAA << 8 | nBBBB << 16;
        }

        std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> Instruction20bc::get_operands()
        {
            std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> operands = {
                {DVMTypes::Operand::LITERAL, nAA},
                {DVMTypes::Operand::LITERAL, nBBBB}};

            return operands;
        }

        DVMTypes::Operand Instruction20bc::get_type_of_error_data_type()
        {
            return DVMTypes::Operand::LITERAL;
        }

        std::uint8_t Instruction20bc::get_type_of_error()
        {
            return nAA;
        }

        DVMTypes::Operand Instruction20bc::get_index_table_data_type()
        {
            return DVMTypes::Operand::LITERAL;
        }

        std::uint16_t Instruction20bc::get_index_table()
        {
            return nBBBB;
        }
        /**
         * Instruction22x
         */
        Instruction22x::Instruction22x(std::shared_ptr<DalvikOpcodes> dalvik_opcodes, std::istream &input_file) : Instruction(dalvik_opcodes, input_file)
        {
            std::uint8_t instruction[4];
            this->set_length(4);

            if (!KUNAI::read_data_file<std::uint8_t[4]>(instruction, this->get_length(), input_file))
                throw exceptions::DisassemblerException("Error disassembling Instruction22x");

            this->set_OP(instruction[0]);
            this->vAA = instruction[1];
            this->vBBBB = *(reinterpret_cast<std::uint16_t *>(&instruction[2]));
        }

        Instruction22x::~Instruction22x() {}

        std::string Instruction22x::get_output()
        {
            return "v" + std::to_string(vAA) + ", v" + std::to_string(vBBBB);
        }

        std::uint64_t Instruction22x::get_raw()
        {
            return (this->get_OP() | vAA << 8 | vBBBB << 16);
        }

        std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> Instruction22x::get_operands()
        {
            std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> operands = {
                {DVMTypes::Operand::REGISTER, vAA},
                {DVMTypes::Operand::REGISTER, vBBBB}};

            return operands;
        }

        DVMTypes::Operand Instruction22x::get_source_type()
        {
            return DVMTypes::Operand::REGISTER;
        }

        std::uint16_t Instruction22x::get_source()
        {
            return vBBBB;
        }

        DVMTypes::Operand Instruction22x::get_destination_type()
        {
            return DVMTypes::Operand::REGISTER;
        }

        std::uint8_t Instruction22x::get_destination()
        {
            return vAA;
        }
        /**
         * Instruction21t
         */
        Instruction21t::Instruction21t(std::shared_ptr<DalvikOpcodes> dalvik_opcodes, std::istream &input_file) : Instruction(dalvik_opcodes, input_file)
        {
            std::uint8_t instruction[4];
            this->set_length(4);

            if (!KUNAI::read_data_file<std::uint8_t[4]>(instruction, this->get_length(), input_file))
                throw exceptions::DisassemblerException("Error disassembling Instruction21t");

            this->set_OP(instruction[0]);
            this->vAA = instruction[1];
            this->nBBBB = *(reinterpret_cast<std::int16_t *>(&instruction[2]));

            if (this->nBBBB == 0)
                throw exceptions::InvalidInstruction("Error reading Instruction21t offset cannot be 0");
        }

        Instruction21t::~Instruction21t() {}

        std::string Instruction21t::get_output()
        {
            return "v" + std::to_string(vAA) + ", " + std::to_string(nBBBB);
        }

        std::uint64_t Instruction21t::get_raw()
        {
            return get_OP() | vAA << 8 | nBBBB << 16;
        }

        std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> Instruction21t::get_operands()
        {
            std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> operands = {
                {DVMTypes::Operand::REGISTER, vAA},
                {DVMTypes::Operand::OFFSET, nBBBB}};

            return operands;
        }

        DVMTypes::Operand Instruction21t::get_check_reg_type()
        {
            return DVMTypes::Operand::REGISTER;
        }

        std::uint8_t Instruction21t::get_check_reg()
        {
            return vAA;
        }

        DVMTypes::Operand Instruction21t::get_ref_type()
        {
            return DVMTypes::Operand::OFFSET;
        }

        std::int16_t Instruction21t::get_ref()
        {
            return nBBBB;
        }
        /**
         * Instruction21s
         */
        Instruction21s::Instruction21s(std::shared_ptr<DalvikOpcodes> dalvik_opcodes, std::istream &input_file) : Instruction(dalvik_opcodes, input_file)
        {
            std::uint8_t instruction[4];
            this->set_length(4);

            if (!KUNAI::read_data_file<std::uint8_t[4]>(instruction, this->get_length(), input_file))
                throw exceptions::DisassemblerException("Error disassembling Instruction21s");

            this->set_OP(instruction[0]);
            this->vA = instruction[1];
            this->nBBBB = *(reinterpret_cast<std::int16_t *>(&instruction[2]));
        }

        Instruction21s::~Instruction21s() {}

        std::string Instruction21s::get_output()
        {
            return "v" + std::to_string(vA) + ", " + std::to_string(nBBBB);
        }

        std::uint64_t Instruction21s::get_raw()
        {
            return (get_OP() | vA << 8 | nBBBB << 16);
        }

        std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> Instruction21s::get_operands()
        {
            std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> operands = {
                {DVMTypes::Operand::REGISTER, vA},
                {DVMTypes::Operand::LITERAL, nBBBB}};

            return operands;
        }

        DVMTypes::Operand Instruction21s::get_source_type()
        {
            return DVMTypes::Operand::LITERAL;
        }

        std::int16_t Instruction21s::get_source()
        {
            return nBBBB;
        }

        DVMTypes::Operand Instruction21s::get_destination_type()
        {
            return DVMTypes::Operand::REGISTER;
        }

        std::uint8_t Instruction21s::get_destination()
        {
            return vA;
        }
        /**
         * Instruction21h
         */
        Instruction21h::Instruction21h(std::shared_ptr<DalvikOpcodes> dalvik_opcodes, std::istream &input_file) : Instruction(dalvik_opcodes, input_file)
        {
            std::uint8_t instruction[4];

            this->set_length(4);

            if (!KUNAI::read_data_file<std::uint8_t[4]>(instruction, this->get_length(), input_file))
                throw exceptions::DisassemblerException("Error disassembling Instruction21h");

            this->set_OP(instruction[0]);
            this->vAA = instruction[1];
            this->nBBBB_aux = *(reinterpret_cast<std::int16_t *>(&instruction[2]));

            switch (this->get_OP())
            {
            case 0x15:
                // const/high16 vAA, #+BBBB0000
                this->nBBBB = static_cast<std::int64_t>(this->nBBBB_aux) << 16;
                break;
            case 0x19:
                // const-wide/high16 vAA, #+BBBB000000000000
                this->nBBBB = static_cast<std::int64_t>(this->nBBBB_aux) << 48;
                break;
            default:
                this->nBBBB = static_cast<std::int64_t>(this->nBBBB_aux);
                break;
            }
        }

        Instruction21h::~Instruction21h() {}

        std::string Instruction21h::get_output()
        {
            return "v" + std::to_string(vAA) + ", " + std::to_string(nBBBB);
        }

        std::uint64_t Instruction21h::get_raw()
        {
            return (get_OP() | vAA << 8 | nBBBB << 16);
        }

        std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> Instruction21h::get_operands()
        {
            std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> operands = {
                {DVMTypes::Operand::REGISTER, vAA},
                {DVMTypes::Operand::LITERAL, nBBBB}};

            return operands;
        }

        DVMTypes::Operand Instruction21h::get_source_type()
        {
            return DVMTypes::Operand::LITERAL;
        }

        std::int64_t Instruction21h::get_source()
        {
            return nBBBB;
        }

        DVMTypes::Operand Instruction21h::get_destination_type()
        {
            return DVMTypes::Operand::REGISTER;
        }

        std::uint8_t Instruction21h::get_destination()
        {
            return vAA;
        }
        /**
         * Instruction21c
         */
        Instruction21c::Instruction21c(std::shared_ptr<DalvikOpcodes> dalvik_opcodes, std::istream &input_file) : Instruction(dalvik_opcodes, input_file)
        {
            std::uint8_t instruction[4];
            this->set_length(4);

            if (!KUNAI::read_data_file<std::uint8_t[4]>(instruction, this->get_length(), input_file))
                throw exceptions::DisassemblerException("Error disassembling Instruction21c");

            this->set_OP(instruction[0]);
            this->vAA = instruction[1];
            this->iBBBB = *(reinterpret_cast<std::uint16_t *>(&instruction[2]));
        }

        Instruction21c::~Instruction21c() {}

        std::string Instruction21c::get_output()
        {
            std::string str = "";

            switch (this->get_kind())
            {
            case DVMTypes::Kind::STRING:
                str = this->get_dalvik_opcodes()->get_dalvik_string_by_id_str(iBBBB);
                break;
            case DVMTypes::Kind::TYPE:
                str = this->get_dalvik_opcodes()->get_dalvik_type_by_id_str(iBBBB);
                break;
            case DVMTypes::Kind::FIELD:
                str = this->get_dalvik_opcodes()->get_dalvik_static_field_by_id_str(iBBBB);
                break;
            case DVMTypes::Kind::METH:
                str = this->get_dalvik_opcodes()->get_dalvik_method_by_id_str(iBBBB);
                break;
            case DVMTypes::Kind::PROTO:
                str = this->get_dalvik_opcodes()->get_dalvik_proto_by_id_str(iBBBB);
                break;
            default:
                str = std::to_string(iBBBB);
                break;
            }

            return "v" + std::to_string(vAA) + ", " + str;
        }

        std::uint64_t Instruction21c::get_raw()
        {
            return (get_OP() | vAA << 8 | iBBBB << 16);
        }

        std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> Instruction21c::get_operands()
        {
            std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> operands = {
                {DVMTypes::Operand::REGISTER, vAA},
                {DVMTypes::Operand::KIND, iBBBB}};

            return operands;
        }

        DVMTypes::Kind Instruction21c::get_source_kind()
        {
            return this->get_kind();
        }

        std::string *Instruction21c::get_source_str()
        {
            if (this->get_kind() == DVMTypes::Kind::STRING)
                return this->get_dalvik_opcodes()->get_dalvik_string_by_id(iBBBB);
            return nullptr;
        }
        Type *Instruction21c::get_source_typeid()
        {
            if (this->get_kind() == DVMTypes::Kind::TYPE)
                return this->get_dalvik_opcodes()->get_dalvik_Type_by_id(iBBBB);
            return nullptr;
        }
        FieldID *Instruction21c::get_source_static_field()
        {
            if (this->get_kind() == DVMTypes::Kind::FIELD)
                return this->get_dalvik_opcodes()->get_dalvik_field_by_id(iBBBB);
            return nullptr;
        }
        MethodID *Instruction21c::get_source_method()
        {
            if (this->get_kind() == DVMTypes::Kind::METH)
                return this->get_dalvik_opcodes()->get_dalvik_method_by_id(iBBBB);
            return nullptr;
        }
        ProtoID *Instruction21c::get_source_proto()
        {
            if (this->get_kind() == DVMTypes::Kind::PROTO)
                return this->get_dalvik_opcodes()->get_dalvik_proto_by_id(iBBBB);
            return nullptr;
        }

        DVMTypes::Operand Instruction21c::get_destination_type()
        {
            return DVMTypes::Operand::REGISTER;
        }

        std::uint8_t Instruction21c::get_destination()
        {
            return vAA;
        }
        /**
         * Instruction23x
         */
        Instruction23x::Instruction23x(std::shared_ptr<DalvikOpcodes> dalvik_opcodes, std::istream &input_file) : Instruction(dalvik_opcodes, input_file)
        {
            std::uint8_t instruction[4];
            this->set_length(4);

            if (!KUNAI::read_data_file<std::uint8_t[4]>(instruction, this->get_length(), input_file))
                throw exceptions::DisassemblerException("Error disassembling Instruction23x");

            this->set_OP(instruction[0]);
            this->vAA = instruction[1];
            this->vBB = instruction[2];
            this->vCC = instruction[3];
        }

        Instruction23x::~Instruction23x() {}

        std::string Instruction23x::get_output()
        {
            return "v" + std::to_string(vAA) + ", v" + std::to_string(vBB) + ", v" + std::to_string(vCC);
        }

        std::uint64_t Instruction23x::get_raw()
        {
            return get_OP() | vAA << 8 | vBB << 16 | vCC << 24;
        }

        std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> Instruction23x::get_operands()
        {
            std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> operands = {
                {DVMTypes::Operand::REGISTER, vAA},
                {DVMTypes::Operand::REGISTER, vBB},
                {DVMTypes::Operand::REGISTER, vCC},
            };

            return operands;
        }

        DVMTypes::Operand Instruction23x::get_destination_type()
        {
            return DVMTypes::Operand::REGISTER;
        }

        std::uint8_t Instruction23x::get_destination()
        {
            return vAA;
        }

        DVMTypes::Operand Instruction23x::get_first_source_type()
        {
            return DVMTypes::Operand::REGISTER;
        }

        std::uint8_t Instruction23x::get_first_source()
        {
            return vBB;
        }

        DVMTypes::Operand Instruction23x::get_second_source_type()
        {
            return DVMTypes::Operand::REGISTER;
        }

        std::uint8_t Instruction23x::get_second_source()
        {
            return vCC;
        }
        /**
         * Instruction22b
         */
        Instruction22b::Instruction22b(std::shared_ptr<DalvikOpcodes> dalvik_opcodes, std::istream &input_file) : Instruction(dalvik_opcodes, input_file)
        {
            std::uint8_t instruction[4];
            this->set_length(4);

            if (!KUNAI::read_data_file<std::uint8_t[4]>(instruction, this->get_length(), input_file))
                throw exceptions::DisassemblerException("Error disassembling Instruction22b");

            this->set_OP(instruction[0]);
            this->vAA = instruction[1];
            this->vBB = instruction[2];
            this->nCC = instruction[3];
        }

        Instruction22b::~Instruction22b() {}

        std::string Instruction22b::get_output()
        {
            return "v" + std::to_string(vAA) + ", v" + std::to_string(vBB) + ", " + std::to_string(nCC);
        }

        std::uint64_t Instruction22b::get_raw()
        {
            return get_OP() | vAA << 8 | vBB << 16 | nCC << 24;
        }

        std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> Instruction22b::get_operands()
        {
            std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> operators = {
                {DVMTypes::Operand::REGISTER, vAA},
                {DVMTypes::Operand::REGISTER, vBB},
                {DVMTypes::Operand::LITERAL, nCC}};

            return operators;
        }

        DVMTypes::Operand Instruction22b::get_destination_type()
        {
            return DVMTypes::Operand::REGISTER;
        }

        std::uint8_t Instruction22b::get_destination()
        {
            return vAA;
        }

        DVMTypes::Operand Instruction22b::get_source_type()
        {
            return DVMTypes::Operand::REGISTER;
        }

        std::uint8_t Instruction22b::get_source()
        {
            return vBB;
        }

        DVMTypes::Operand Instruction22b::get_number_type()
        {
            return DVMTypes::Operand::LITERAL;
        }

        std::int8_t Instruction22b::get_number()
        {
            return nCC;
        }
        /**
         * Instruction22t
         */
        Instruction22t::Instruction22t(std::shared_ptr<DalvikOpcodes> dalvik_opcodes, std::istream &input_file) : Instruction(dalvik_opcodes, input_file)
        {
            std::uint8_t instruction[4];
            this->set_length(4);

            if (!KUNAI::read_data_file<std::uint8_t[4]>(instruction, this->get_length(), input_file))
                throw exceptions::DisassemblerException("Error disassembling Instruction22t");

            this->set_OP(instruction[0]);
            this->vA = instruction[1] & 0x0F;
            this->vB = (instruction[1] & 0xF0) >> 4;
            this->nCCCC = *(reinterpret_cast<std::int16_t *>(&instruction[2]));

            if (this->nCCCC == 0)
                throw exceptions::InvalidInstruction("Error reading Instruction22t offset cannot be 0");
        }

        Instruction22t::~Instruction22t() {}

        std::string Instruction22t::get_output()
        {
            return "v" + std::to_string(vA) + ", v" + std::to_string(vB) + ", " + std::to_string(nCCCC);
        }

        std::uint64_t Instruction22t::get_raw()
        {
            return get_OP() | vA << 8 | vB << 12 | nCCCC << 16;
        }

        std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> Instruction22t::get_operands()
        {
            std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> operands = {
                {DVMTypes::Operand::REGISTER, vA},
                {DVMTypes::Operand::REGISTER, vB},
                {DVMTypes::Operand::OFFSET, nCCCC}};

            return operands;
        }

        DVMTypes::Operand Instruction22t::get_first_check_reg_type()
        {
            return DVMTypes::Operand::REGISTER;
        }

        std::uint8_t Instruction22t::get_first_check_reg()
        {
            return vA;
        }

        DVMTypes::Operand Instruction22t::get_second_check_reg_type()
        {
            return DVMTypes::Operand::REGISTER;
        }

        std::uint8_t Instruction22t::get_second_check_reg()
        {
            return vB;
        }

        DVMTypes::Operand Instruction22t::get_ref_type()
        {
            return DVMTypes::Operand::OFFSET;
        }

        std::int16_t Instruction22t::get_ref()
        {
            return nCCCC;
        }
        /**
         * Instruction22s
         */
        Instruction22s::Instruction22s(std::shared_ptr<DalvikOpcodes> dalvik_opcodes, std::istream &input_file) : Instruction(dalvik_opcodes, input_file)
        {
            std::uint8_t instruction[4];
            this->set_length(4);

            if (!KUNAI::read_data_file<std::uint8_t[4]>(instruction, this->get_length(), input_file))
                throw exceptions::DisassemblerException("Error disassembling Instruction22s");

            this->set_OP(instruction[0]);
            this->vA = instruction[1] & 0x0F;
            this->vB = (instruction[1] & 0xF0) >> 4;
            this->nCCCC = *(reinterpret_cast<std::int16_t *>(&instruction[2]));
        }

        Instruction22s::~Instruction22s() {}

        std::string Instruction22s::get_output()
        {
            return "v" + std::to_string(vA) + ", v" + std::to_string(vB) + ", " + std::to_string(nCCCC);
        }

        std::uint64_t Instruction22s::get_raw()
        {
            return get_OP() | vA << 8 | vB << 12 | nCCCC << 16;
        }

        std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> Instruction22s::get_operands()
        {
            std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> operands = {
                {DVMTypes::Operand::REGISTER, vA},
                {DVMTypes::Operand::REGISTER, vB},
                {DVMTypes::Operand::LITERAL, nCCCC}};

            return operands;
        }

        DVMTypes::Operand Instruction22s::get_destination_type()
        {
            return DVMTypes::Operand::REGISTER;
        }

        std::uint8_t Instruction22s::get_destination()
        {
            return vA;
        }

        DVMTypes::Operand Instruction22s::get_source_type()
        {
            return DVMTypes::Operand::REGISTER;
        }

        std::uint8_t Instruction22s::get_source()
        {
            return vB;
        }

        DVMTypes::Operand Instruction22s::get_number_type()
        {
            return DVMTypes::Operand::LITERAL;
        }

        std::int16_t Instruction22s::get_number()
        {
            return nCCCC;
        }
        /**
         * Instruction22c
         */
        Instruction22c::Instruction22c(std::shared_ptr<DalvikOpcodes> dalvik_opcodes, std::istream &input_file) : Instruction(dalvik_opcodes, input_file)
        {
            std::uint8_t instruction[4];
            this->set_length(4);

            if (!KUNAI::read_data_file<std::uint8_t[4]>(instruction, this->get_length(), input_file))
                throw exceptions::DisassemblerException("Error disassembling Instruction22c");

            this->set_OP(instruction[0]);
            this->vA = instruction[1] & 0xF;
            this->vB = (instruction[1] & 0xF0) >> 4;
            this->iCCCC = *(reinterpret_cast<std::uint16_t *>(&instruction[2]));
        }

        Instruction22c::~Instruction22c() {}

        std::string Instruction22c::get_output()
        {
            std::string str;

            switch (this->get_kind())
            {
            case DVMTypes::Kind::TYPE:
                str = this->get_dalvik_opcodes()->get_dalvik_type_by_id_str(iCCCC);
                break;
            case DVMTypes::Kind::FIELD:
                str = this->get_dalvik_opcodes()->get_dalvik_static_field_by_id_str(iCCCC);
                break;
            default:
                str = std::to_string(iCCCC);
                break;
            }

            return "v" + std::to_string(vA) + ", v" + std::to_string(vB) + ", " + str;
        }

        std::uint64_t Instruction22c::get_raw()
        {
            return (get_OP() | vA << 8 | vB << 12 | iCCCC << 16);
        }

        std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> Instruction22c::get_operands()
        {
            std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> operands = {
                {DVMTypes::Operand::REGISTER, vA},
                {DVMTypes::Operand::REGISTER, vB},
                {DVMTypes::Operand::KIND, iCCCC}};

            return operands;
        }

        DVMTypes::Operand Instruction22c::get_first_operand_type()
        {
            return DVMTypes::Operand::REGISTER;
        }

        std::uint8_t Instruction22c::get_first_operand()
        {
            return vA;
        }

        DVMTypes::Operand Instruction22c::get_second_operand_type()
        {
            return DVMTypes::Operand::REGISTER;
        }

        std::uint8_t Instruction22c::get_second_operand()
        {
            return vB;
        }

        DVMTypes::Operand Instruction22c::get_third_operand_type()
        {
            return DVMTypes::Operand::KIND;
        }

        std::uint16_t Instruction22c::get_third_operand()
        {
            return iCCCC;
        }

        DVMTypes::Kind Instruction22c::get_third_operand_kind()
        {
            return this->get_kind();
        }

        Type *Instruction22c::get_third_operand_typeId()
        {
            if (get_kind() == DVMTypes::Kind::TYPE)
                return this->get_dalvik_opcodes()->get_dalvik_Type_by_id(iCCCC);
            return nullptr;
        }

        FieldID *Instruction22c::get_third_operand_FieldId()
        {
            if (get_kind() == DVMTypes::Kind::FIELD)
                return this->get_dalvik_opcodes()->get_dalvik_field_by_id(iCCCC);
            return nullptr;
        }
        /**
         * Instruction22cs
         */
        Instruction22cs::Instruction22cs(std::shared_ptr<DalvikOpcodes> dalvik_opcodes, std::istream &input_file) : Instruction(dalvik_opcodes, input_file)
        {
            std::uint8_t instruction[4];
            this->set_length(4);

            if (!KUNAI::read_data_file<std::uint8_t[4]>(instruction, this->get_length(), input_file))
                throw exceptions::DisassemblerException("Error disassembling Instruction22c");

            this->set_OP(instruction[0]);
            this->vA = instruction[1] & 0xF;
            this->vB = (instruction[1] & 0xF0) >> 4;
            this->iCCCC = *(reinterpret_cast<std::uint16_t *>(&instruction[2]));
        }

        Instruction22cs::~Instruction22cs() {}

        std::string Instruction22cs::get_output()
        {
            std::string str;

            switch (this->get_kind())
            {
            case DVMTypes::Kind::TYPE:
                str = this->get_dalvik_opcodes()->get_dalvik_type_by_id_str(iCCCC);
                break;
            case DVMTypes::Kind::FIELD:
                str = this->get_dalvik_opcodes()->get_dalvik_static_field_by_id_str(iCCCC);
                break;
            default:
                str = std::to_string(iCCCC);
                break;
            }

            return "v" + std::to_string(vA) + ", v" + std::to_string(vB) + ", " + str;
        }

        std::uint64_t Instruction22cs::get_raw()
        {
            return (get_OP() | vA << 8 | vB << 12 | iCCCC << 16);
        }

        std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> Instruction22cs::get_operands()
        {
            std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> operands = {
                {DVMTypes::Operand::REGISTER, vA},
                {DVMTypes::Operand::REGISTER, vB},
                {DVMTypes::Operand::KIND, iCCCC}};

            return operands;
        }

        DVMTypes::Operand Instruction22cs::get_first_operand_type()
        {
            return DVMTypes::Operand::REGISTER;
        }

        std::uint8_t Instruction22cs::get_first_operand()
        {
            return vA;
        }

        DVMTypes::Operand Instruction22cs::get_second_operand_type()
        {
            return DVMTypes::Operand::REGISTER;
        }

        std::uint8_t Instruction22cs::get_second_operand()
        {
            return vB;
        }

        DVMTypes::Operand Instruction22cs::get_third_operand_type()
        {
            return DVMTypes::Operand::KIND;
        }

        std::uint16_t Instruction22cs::get_third_operand()
        {
            return iCCCC;
        }

        DVMTypes::Kind Instruction22cs::get_third_operand_kind()
        {
            return this->get_kind();
        }

        Type *Instruction22cs::get_third_operand_typeId()
        {
            if (get_kind() == DVMTypes::Kind::TYPE)
                return this->get_dalvik_opcodes()->get_dalvik_Type_by_id(iCCCC);
            return nullptr;
        }

        FieldID *Instruction22cs::get_third_operand_FieldId()
        {
            if (get_kind() == DVMTypes::Kind::FIELD)
                return this->get_dalvik_opcodes()->get_dalvik_field_by_id(iCCCC);
            return nullptr;
        }
        /**
         * Instruction30t
         */
        Instruction30t::Instruction30t(std::shared_ptr<DalvikOpcodes> dalvik_opcodes, std::istream &input_file) : Instruction(dalvik_opcodes, input_file)
        {
            std::uint8_t instruction[6];
            this->set_length(6);

            if (!KUNAI::read_data_file<std::uint8_t[6]>(instruction, this->get_length(), input_file))
                throw exceptions::DisassemblerException("Error disassembling Instruction30t");

            if (instruction[1] != 0)
                throw exceptions::InvalidInstruction("Error reading Instruction30t padding must be 0");

            this->set_OP(instruction[0]);
            this->nAAAAAAAA = *(reinterpret_cast<std::int32_t *>(&instruction[2]));

            if (this->nAAAAAAAA == 0)
                throw exceptions::InvalidInstruction("Error reading Instruction30t offset cannot be 0");
        }

        Instruction30t::~Instruction30t() {}

        std::string Instruction30t::get_output()
        {
            return std::to_string(nAAAAAAAA);
        }

        std::uint64_t Instruction30t::get_raw()
        {
            return get_OP() | nAAAAAAAA << 16;
        }

        std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> Instruction30t::get_operands()
        {
            std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> operands = {
                {DVMTypes::Operand::OFFSET, nAAAAAAAA}};

            return operands;
        }

        DVMTypes::Operand Instruction30t::get_offset_type()
        {
            return DVMTypes::Operand::OFFSET;
        }

        std::int32_t Instruction30t::get_offset()
        {
            return nAAAAAAAA;
        }
        /**
         * Instruction32x
         */
        Instruction32x::Instruction32x(std::shared_ptr<DalvikOpcodes> dalvik_opcodes, std::istream &input_file) : Instruction(dalvik_opcodes, input_file)
        {
            std::uint8_t instruction[6];

            this->set_length(6);

            if (!KUNAI::read_data_file<std::uint8_t[6]>(instruction, this->get_length(), input_file))
                throw exceptions::DisassemblerException("Error disassembling Instruction32x");

            if (instruction[1] != 0)
                throw exceptions::InvalidInstruction("Instruction32x OP code high byte should be 0");

            this->set_OP(instruction[0]);
            this->vAAAA = *(reinterpret_cast<std::uint16_t *>(&instruction[2]));
            this->vBBBB = *(reinterpret_cast<std::uint16_t *>(&instruction[4]));
        }

        Instruction32x::~Instruction32x() {}

        std::string Instruction32x::get_output()
        {
            return "v" + std::to_string(vAAAA) + ", v" + std::to_string(vBBBB);
        }

        std::uint64_t Instruction32x::get_raw()
        {
            return (get_OP() | vAAAA << 16 | vBBBB << 24);
        }

        std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> Instruction32x::get_operands()
        {
            std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> operands = {
                {DVMTypes::Operand::REGISTER, vAAAA},
                {DVMTypes::Operand::REGISTER, vBBBB}};

            return operands;
        }

        DVMTypes::Operand Instruction32x::get_source_type()
        {
            return DVMTypes::Operand::REGISTER;
        }

        std::uint16_t Instruction32x::get_source()
        {
            return vBBBB;
        }

        DVMTypes::Operand Instruction32x::get_destination_type()
        {
            return DVMTypes::Operand::REGISTER;
        }

        std::uint16_t Instruction32x::get_destination()
        {
            return vAAAA;
        }

        /**
         * Instruction31i
         */
        Instruction31i::Instruction31i(std::shared_ptr<DalvikOpcodes> dalvik_opcodes, std::istream &input_file) : Instruction(dalvik_opcodes, input_file)
        {
            this->set_length(6);
            std::uint8_t instruction[6];

            if (!KUNAI::read_data_file<std::uint8_t[6]>(instruction, this->get_length(), input_file))
                throw exceptions::DisassemblerException("Error disassembling Instruction31i");

            this->set_OP(instruction[0]);
            this->vAA = instruction[1];
            this->nBBBBBBBB = *(reinterpret_cast<std::uint32_t *>(&instruction[2]));
        }

        Instruction31i::~Instruction31i() {}

        std::string Instruction31i::get_output()
        {
            return "v" + std::to_string(vAA) + ", " + std::to_string(nBBBBBBBB);
        }

        std::uint64_t Instruction31i::get_raw()
        {
            return (get_OP() | vAA << 8 | nBBBBBBBB << 16);
        }

        std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> Instruction31i::get_operands()
        {
            std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> operands = {
                {DVMTypes::Operand::REGISTER, vAA},
                {DVMTypes::Operand::LITERAL, nBBBBBBBB}};

            return operands;
        }

        DVMTypes::Operand Instruction31i::get_source_type()
        {
            return DVMTypes::Operand::LITERAL;
        }

        std::int32_t Instruction31i::get_source()
        {
            return nBBBBBBBB;
        }

        DVMTypes::Operand Instruction31i::get_destination_type()
        {
            return DVMTypes::Operand::REGISTER;
        }

        std::uint8_t Instruction31i::get_destination()
        {
            return vAA;
        }
        /**
         * Instruction31t
         */
        Instruction31t::Instruction31t(std::shared_ptr<DalvikOpcodes> dalvik_opcodes, std::istream &input_file) : Instruction(dalvik_opcodes, input_file)
        {
            std::uint8_t instruction[6];
            this->set_length(6);

            if (!KUNAI::read_data_file<std::uint8_t[6]>(instruction, this->get_length(), input_file))
                throw exceptions::DisassemblerException("Error disassembling Instruction31t");

            this->set_OP(instruction[0]);
            this->vAA = instruction[1];
            this->nBBBBBBBB = *(reinterpret_cast<std::int32_t *>(&instruction[2]));
        }

        Instruction31t::~Instruction31t() {}

        std::string Instruction31t::get_output()
        {
            return "v" + std::to_string(vAA) + ", " + std::to_string(nBBBBBBBB);
        }

        std::uint64_t Instruction31t::get_raw()
        {
            return get_OP() | vAA << 8 | nBBBBBBBB << 16;
        }

        std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> Instruction31t::get_operands()
        {
            std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> operands = {
                {DVMTypes::Operand::REGISTER, vAA},
                {DVMTypes::Operand::OFFSET, nBBBBBBBB}};

            return operands;
        }

        DVMTypes::Operand Instruction31t::get_array_ref_type()
        {
            return DVMTypes::Operand::REGISTER;
        }

        std::uint8_t Instruction31t::get_array_ref()
        {
            return vAA;
        }

        DVMTypes::Operand Instruction31t::get_offset_type()
        {
            return DVMTypes::Operand::OFFSET;
        }

        std::int32_t Instruction31t::get_offset()
        {
            return nBBBBBBBB;
        }
        /**
         * Instruction31c
         */
        Instruction31c::Instruction31c(std::shared_ptr<DalvikOpcodes> dalvik_opcodes, std::istream &input_file) : Instruction(dalvik_opcodes, input_file)
        {
            std::uint8_t instruction[6];
            this->set_length(6);

            if (!KUNAI::read_data_file<std::uint8_t[6]>(instruction, this->get_length(), input_file))
                throw exceptions::DisassemblerException("Error disassembling Instruction31c");

            this->set_OP(instruction[0]);
            this->vAA = instruction[1];
            this->iBBBBBBBB = *(reinterpret_cast<std::uint32_t *>(&instruction[2]));
        }

        Instruction31c::~Instruction31c() {}

        std::string Instruction31c::get_output()
        {
            return "v" + std::to_string(vAA) + ", " + this->get_dalvik_opcodes()->get_dalvik_string_by_id_str(iBBBBBBBB);
        }

        std::uint64_t Instruction31c::get_raw()
        {
            return (get_OP() | vAA << 8 | iBBBBBBBB << 16);
        }

        std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> Instruction31c::get_operands()
        {
            std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> operands = {
                {DVMTypes::Operand::REGISTER, vAA},
                {DVMTypes::Operand::KIND, iBBBBBBBB}};

            return operands;
        }

        DVMTypes::Operand Instruction31c::get_source_type()
        {
            return DVMTypes::Operand::KIND;
        }

        std::uint16_t Instruction31c::get_source()
        {
            return iBBBBBBBB;
        }

        DVMTypes::Kind Instruction31c::get_source_kind()
        {
            return this->get_kind();
        }

        std::string *Instruction31c::get_source_str()
        {
            return this->get_dalvik_opcodes()->get_dalvik_string_by_id(iBBBBBBBB);
        }

        DVMTypes::Operand Instruction31c::get_destination_type()
        {
            return DVMTypes::Operand::REGISTER;
        }

        std::uint8_t Instruction31c::get_destination()
        {
            return vAA;
        }
        /**
         * Instruction35c
         */
        Instruction35c::Instruction35c(std::shared_ptr<DalvikOpcodes> dalvik_opcodes, std::istream &input_file) : Instruction(dalvik_opcodes, input_file)
        {
            std::uint8_t reg[5];

            std::uint8_t instruction[6];
            this->set_length(6);

            if (!KUNAI::read_data_file<std::uint8_t[6]>(instruction, this->get_length(), input_file))
                throw exceptions::DisassemblerException("Error disassembling Instruction35c");

            this->set_OP(instruction[0]);
            this->array_size = (instruction[1] & 0xf0) >> 4;
            this->type_index = *(reinterpret_cast<std::uint16_t *>(&instruction[2]));
            reg[4] = (instruction[1] & 0x0f);
            reg[0] = (instruction[4] & 0x0f);
            reg[1] = (instruction[4] & 0xf0) >> 4;
            reg[2] = (instruction[5] & 0x0f);
            reg[3] = (instruction[5] & 0xf0) >> 4;

            if (this->array_size > 5)
                throw exceptions::InvalidInstruction("Error in array size of Instruction35c, cannot be greater than 5");

            for (size_t i = 0; i < this->array_size; i++)
                this->registers.push_back(reg[i]);
        }

        Instruction35c::~Instruction35c() {}

        std::string Instruction35c::get_output()
        {
            std::string output = "";
            std::string ref = "";

            switch (get_kind())
            {
            case DVMTypes::Kind::TYPE:
                ref = this->get_dalvik_opcodes()->get_dalvik_type_by_id_str(type_index);
                break;
            case DVMTypes::Kind::METH:
                ref = this->get_dalvik_opcodes()->get_dalvik_method_by_id_str(type_index);
                break;
            // don't know how to manage CALL_SITE Kind...
            default:
                ref = std::to_string(type_index);
            }

            for (size_t i = 0; i < array_size; i++)
                output += "v" + std::to_string(registers[i]) + ", ";

            output += ref;

            return output;
        }

        std::uint64_t Instruction35c::get_raw()
        {
            std::uint8_t raw[8] = {0, 0, 0, 0, 0, 0, 0, 0};
            std::uint8_t reg[5] = {0, 0, 0, 0, 0};

            for (size_t i = 0; i < array_size; i++)
                reg[i] = registers[i];

            raw[0] = get_OP();
            raw[1] = array_size << 4 | reg[4];
            *reinterpret_cast<std::uint16_t *>(&raw[2]) = type_index;
            raw[4] = reg[1] << 4 | reg[0];
            raw[5] = reg[3] << 4 | reg[2];

            return *(reinterpret_cast<std::uint64_t *>(raw));
        }

        std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> Instruction35c::get_operands()
        {
            std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> operands;

            for (size_t i = 0; i < array_size; i++)
                operands.push_back({DVMTypes::Operand::REGISTER, registers[i]});

            operands.push_back({DVMTypes::Operand::KIND, type_index});

            return operands;
        }

        std::uint8_t Instruction35c::get_array_size()
        {
            return array_size;
        }

        std::uint16_t Instruction35c::get_type_index()
        {
            return type_index;
        }

        DVMTypes::Operand Instruction35c::get_operands_types()
        {
            return DVMTypes::Operand::REGISTER;
        }

        Type *Instruction35c::get_operands_kind_type()
        {
            return this->get_dalvik_opcodes()->get_dalvik_Type_by_id(type_index);
        }

        std::string Instruction35c::get_operands_kind_type_str()
        {
            return this->get_dalvik_opcodes()->get_dalvik_type_by_id_str(type_index);
        }

        MethodID *Instruction35c::get_operands_kind_method()
        {
            return this->get_dalvik_opcodes()->get_dalvik_method_by_id(type_index);
        }

        std::string Instruction35c::get_operands_kind_method_str()
        {
            return this->get_dalvik_opcodes()->get_dalvik_method_by_id_str(type_index);
        }

        std::uint8_t Instruction35c::get_operand_register(std::uint8_t index)
        {
            if (index >= array_size)
                return 0;
            return registers[index];
        }
        /**
         * Instruction3rc
         */
        Instruction3rc::Instruction3rc(std::shared_ptr<DalvikOpcodes> dalvik_opcodes, std::istream &input_file) : Instruction(dalvik_opcodes, input_file)
        {
            std::uint8_t instruction[6];
            std::uint16_t vCCCC;
            this->set_length(6);

            if (!KUNAI::read_data_file<std::uint8_t[6]>(instruction, this->get_length(), input_file))
                throw exceptions::DisassemblerException("Error disassembling Instruction3rc");

            this->set_OP(instruction[0]);
            this->array_size = instruction[1];
            this->index = *(reinterpret_cast<std::uint16_t *>(&instruction[2]));
            vCCCC = *(reinterpret_cast<std::uint16_t *>(&instruction[4]));

            for (std::uint16_t i = vCCCC; i < vCCCC + this->array_size; i++)
                registers.push_back(i);
        }

        Instruction3rc::~Instruction3rc() {}

        std::string Instruction3rc::get_output()
        {
            std::string output = "";

            for (std::uint16_t i = 0; i < array_size; i++)
                output += "v" + std::to_string(registers[i]) + ", ";

            /**
             * This instruction is a little bit pain in the ass
             * there are different kind so we must decide here
             * which one we will use.
             *  	op {vCCCC .. vNNNN}, meth@BBBB
             *      op {vCCCC .. vNNNN}, site@BBBB
             *      op {vCCCC .. vNNNN}, type@BBBB
             * 
             * Check for meth, and type, any other case, return
             * index as string.
             */
            if (this->get_kind() == DVMTypes::Kind::TYPE)
                output += this->get_dalvik_opcodes()->get_dalvik_type_by_id_str(index);
            else if (this->get_kind() == DVMTypes::Kind::METH)
                output += this->get_dalvik_opcodes()->get_dalvik_method_by_id_str(index);
            else
                output += std::to_string(index);

            return output;
        }

        std::uint64_t Instruction3rc::get_raw()
        {
            return get_OP() | array_size << 8 | static_cast<std::uint64_t>(index) << 16 | static_cast<std::uint64_t>(registers[0]) << 32;
        }

        std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> Instruction3rc::get_operands()
        {
            std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> operands;

            for (std::uint16_t i = 0; i < array_size; i++)
                operands.push_back({DVMTypes::Operand::REGISTER, i});

            operands.push_back({DVMTypes::Operand::KIND, index});

            return operands;
        }

        std::uint8_t Instruction3rc::get_array_size()
        {
            return array_size;
        }

        std::uint16_t Instruction3rc::get_index()
        {
            return index;
        }

        DVMTypes::Operand Instruction3rc::get_operands_types()
        {
            return DVMTypes::Operand::REGISTER;
        }

        DVMTypes::Kind Instruction3rc::get_index_kind()
        {
            return get_kind();
        }

        std::any Instruction3rc::get_last_operand()
        {
            if (get_kind() == DVMTypes::Kind::TYPE)
                return this->get_dalvik_opcodes()->get_dalvik_Type_by_id(index);
            else if (get_kind() == DVMTypes::Kind::METH)
                return this->get_dalvik_opcodes()->get_dalvik_method_by_id(index);
            else
                return index;
        }

        std::string Instruction3rc::get_last_operand_str()
        {
            if (get_kind() == DVMTypes::Kind::TYPE)
                return this->get_dalvik_opcodes()->get_dalvik_type_by_id_str(index);
            else if (get_kind() == DVMTypes::Kind::METH)
                return this->get_dalvik_opcodes()->get_dalvik_method_by_id_str(index);
            else
                return std::to_string(index);
        }

        Type *Instruction3rc::get_operands_type()
        {
            if (get_kind() == DVMTypes::Kind::TYPE)
                return this->get_dalvik_opcodes()->get_dalvik_Type_by_id(index);
            else
                return nullptr;
        }

        std::string Instruction3rc::get_operands_type_str()
        {
            if (get_kind() == DVMTypes::Kind::TYPE)
                return this->get_dalvik_opcodes()->get_dalvik_type_by_id_str(index);
            else
                return "";
        }

        MethodID* Instruction3rc::get_operands_method()
        {
            if (get_kind() == DVMTypes::Kind::METH)
                return this->get_dalvik_opcodes()->get_dalvik_method_by_id(index);
            else
                return nullptr;
        }

        std::string Instruction3rc::get_operands_method_str()
        {
            if (get_kind() == DVMTypes::Kind::METH)
                return this->get_dalvik_opcodes()->get_dalvik_method_by_id_str(index);
            else
                return "";
        }

        std::uint8_t Instruction3rc::get_operand_register(std::uint8_t index)
        {
            if (index >= array_size)
                return 0;
            return registers[index];
        }
        /**
         * Instruction45cc
         */
        Instruction45cc::Instruction45cc(std::shared_ptr<DalvikOpcodes> dalvik_opcodes, std::istream &input_file) : Instruction(dalvik_opcodes, input_file)
        {
            std::uint8_t instruction[8];
            std::uint8_t regC, regD, regE, regF, regG;
            this->set_length(8);

            if (!KUNAI::read_data_file<std::uint8_t[8]>(instruction, this->get_length(), input_file))
                throw exceptions::DisassemblerException("Error disassembling Instruction45cc");

            this->set_OP(instruction[0]);
            this->reg_count = (instruction[1] & 0xF0) >> 4;
            regG = instruction[1] & 0x0F;
            this->method_reference = *(reinterpret_cast<std::uint16_t *>(&instruction[2]));
            regD = (instruction[4] & 0xF0) >> 4;
            regC = instruction[4] & 0x0F;
            regF = (instruction[5] & 0xF0) >> 4;
            regE = instruction[5] & 0x0F;
            this->proto_reference = *(reinterpret_cast<std::uint16_t *>(&instruction[6]));

            if (this->reg_count > 5)
                throw exceptions::InvalidInstruction("Error in reg_count from Instruction45cc cannot be greater than 5");

            if (this->reg_count > 0)
                this->registers.push_back(regC);
            if (this->reg_count > 1)
                this->registers.push_back(regD);
            if (this->reg_count > 2)
                this->registers.push_back(regE);
            if (this->reg_count > 3)
                this->registers.push_back(regF);
            if (this->reg_count > 4)
                this->registers.push_back(regG);
        }

        Instruction45cc::~Instruction45cc() {}

        std::string Instruction45cc::get_output()
        {
            std::string output = "";
            std::string method = get_dalvik_opcodes()->get_dalvik_method_by_id_str(method_reference);
            std::string prototype = get_dalvik_opcodes()->get_dalvik_proto_by_id_str(proto_reference);

            for (size_t i = 0; i < reg_count; i++)
                output += "v" + std::to_string(registers[i]) + ", ";
            output += method + ", " + prototype;
            return output;
        }

        std::uint64_t Instruction45cc::get_raw()
        {
            std::uint8_t instruction[8];
            std::uint8_t regC = 0, regD = 0, regE = 0, regF = 0, regG = 0;

            if (reg_count > 0)
                regC = registers[0];
            if (reg_count > 1)
                regD = registers[1];
            if (reg_count > 2)
                regE = registers[2];
            if (reg_count > 3)
                regF = registers[3];
            if (reg_count > 4)
                regG = registers[4];

            instruction[0] = get_OP();
            instruction[1] = reg_count << 4 | regG;
            *(reinterpret_cast<std::uint16_t *>(&instruction[2])) = method_reference;
            instruction[4] = regD << 4 | regC;
            instruction[5] = regF << 4 | regE;
            *(reinterpret_cast<std::uint16_t *>(&instruction[6])) = proto_reference;
            return *(reinterpret_cast<std::uint64_t *>(&instruction[0]));
        }

        std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> Instruction45cc::get_operands()
        {
            std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> operands;

            for (size_t i = 0; i < reg_count; i++)
            {
                operands.push_back({DVMTypes::Operand::REGISTER, registers[i]});
            }
            operands.push_back({DVMTypes::Operand::KIND, method_reference});
            operands.push_back({DVMTypes::Operand::KIND, proto_reference});

            return operands;
        }

        std::uint8_t Instruction45cc::get_reg_count()
        {
            return reg_count;
        }

        DVMTypes::Operand Instruction45cc::get_register_types()
        {
            return DVMTypes::Operand::REGISTER;
        }

        std::uint8_t Instruction45cc::get_register(std::uint16_t index)
        {
            if (index >= reg_count)
                return 0;
            return registers[index];
        }

        DVMTypes::Kind Instruction45cc::get_method_ref_kind()
        {
            return DVMTypes::Kind::METH;
        }

        MethodID *Instruction45cc::get_method_ref()
        {
            return get_dalvik_opcodes()->get_dalvik_method_by_id(method_reference);
        }

        std::string Instruction45cc::get_method_ref_str()
        {
            return get_dalvik_opcodes()->get_dalvik_method_by_id_str(method_reference);
        }

        DVMTypes::Kind Instruction45cc::get_proto_ref_kind()
        {
            return DVMTypes::Kind::PROTO;
        }

        ProtoID *Instruction45cc::get_proto_ref()
        {
            return get_dalvik_opcodes()->get_dalvik_proto_by_id(proto_reference);
        }

        std::string Instruction45cc::get_proto_ref_str()
        {
            return get_dalvik_opcodes()->get_dalvik_proto_by_id_str(proto_reference);
        }
        /**
         * Instruction4rcc
         */
        Instruction4rcc::Instruction4rcc(std::shared_ptr<DalvikOpcodes> dalvik_opcodes, std::istream &input_file) : Instruction(dalvik_opcodes, input_file)
        {
            std::uint8_t instruction[8];
            std::uint16_t vCCCC;
            this->set_length(8);

            if (!KUNAI::read_data_file<std::uint8_t[8]>(instruction, this->get_length(), input_file))
                throw exceptions::DisassemblerException("Error disassembling Instruction4rcc");

            this->set_OP(instruction[0]);
            this->reg_count = instruction[1];
            this->method_reference = *(reinterpret_cast<std::uint16_t *>(&instruction[2]));
            vCCCC = *(reinterpret_cast<std::uint16_t *>(&instruction[4]));
            this->proto_reference = *(reinterpret_cast<std::uint16_t *>(&instruction[6]));

            for (size_t i = vCCCC; i < vCCCC + this->reg_count; i++)
                this->registers.push_back(i);
        }

        Instruction4rcc::~Instruction4rcc() {}

        std::string Instruction4rcc::get_output()
        {
            std::string output = "";
            std::string method = get_dalvik_opcodes()->get_dalvik_method_by_id_str(method_reference);
            std::string prototype = get_dalvik_opcodes()->get_dalvik_proto_by_id_str(proto_reference);

            for (size_t i = 0; i < reg_count; i++)
                output = "v" + std::to_string(registers[i]) + ", ";
            output += method + ", " + prototype;
            return output;
        }

        std::uint64_t Instruction4rcc::get_raw()
        {
            std::uint8_t instruction[8];

            instruction[0] = get_OP();
            instruction[1] = reg_count;
            *(reinterpret_cast<std::uint16_t *>(&instruction[2])) = method_reference;
            *(reinterpret_cast<std::uint16_t *>(&instruction[4])) = registers[0];
            *(reinterpret_cast<std::uint16_t *>(&instruction[6])) = proto_reference;

            return *(reinterpret_cast<std::uint64_t *>(&instruction[0]));
        }

        std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> Instruction4rcc::get_operands()
        {
            std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> operands;

            for (size_t i = 0; i < reg_count; i++)
                operands.push_back({DVMTypes::Operand::REGISTER, registers[i]});
            operands.push_back({DVMTypes::Operand::KIND, method_reference});
            operands.push_back({DVMTypes::Operand::KIND, proto_reference});

            return operands;
        }

        std::uint8_t Instruction4rcc::get_reg_count()
        {
            return reg_count;
        }

        DVMTypes::Operand Instruction4rcc::get_register_types()
        {
            return DVMTypes::Operand::REGISTER;
        }

        std::uint16_t Instruction4rcc::get_register(std::uint16_t index)
        {
            if (index >= reg_count)
                return 0;
            return registers[index];
        }

        DVMTypes::Kind Instruction4rcc::get_method_ref_kind()
        {
            return DVMTypes::Kind::METH;
        }

        MethodID *Instruction4rcc::get_method_ref()
        {
            return get_dalvik_opcodes()->get_dalvik_method_by_id(method_reference);
        }

        std::string Instruction4rcc::get_method_ref_str()
        {
            return get_dalvik_opcodes()->get_dalvik_method_by_id_str(method_reference);
        }

        DVMTypes::Kind Instruction4rcc::get_proto_ref_kind()
        {
            return DVMTypes::Kind::PROTO;
        }

        ProtoID *Instruction4rcc::get_proto_ref()
        {
            return get_dalvik_opcodes()->get_dalvik_proto_by_id(proto_reference);
        }

        std::string Instruction4rcc::get_proto_ref_str()
        {
            return get_dalvik_opcodes()->get_dalvik_proto_by_id_str(proto_reference);
        }
        /**
         * Instruction51l
         */
        Instruction51l::Instruction51l(std::shared_ptr<DalvikOpcodes> dalvik_opcodes, std::istream &input_file) : Instruction(dalvik_opcodes, input_file)
        {
            std::uint8_t instruction[10];
            this->set_length(10);

            if (!KUNAI::read_data_file<std::uint8_t[10]>(instruction, this->get_length(), input_file))
                throw exceptions::DisassemblerException("Error disassembling Instruction51l");

            this->set_OP(instruction[0]);
            this->vAA = instruction[1];
            this->nBBBBBBBBBBBBBBBB = *(reinterpret_cast<std::uint64_t *>(&instruction[2]));
        }

        Instruction51l::~Instruction51l() {}

        std::string Instruction51l::get_output()
        {
            return "v" + std::to_string(vAA) + ", " + std::to_string(nBBBBBBBBBBBBBBBB);
        }

        std::uint64_t Instruction51l::get_raw()
        {
            return (get_OP() | vAA << 8 | nBBBBBBBBBBBBBBBB << 16);
        }

        std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> Instruction51l::get_operands()
        {
            std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> operands = {
                {DVMTypes::Operand::REGISTER, vAA},
                {DVMTypes::Operand::LITERAL, nBBBBBBBBBBBBBBBB}};

            return operands;
        }

        DVMTypes::Operand Instruction51l::get_source_type()
        {
            return DVMTypes::Operand::LITERAL;
        }

        std::uint64_t Instruction51l::get_source()
        {
            return nBBBBBBBBBBBBBBBB;
        }

        DVMTypes::Operand Instruction51l::get_destination_type()
        {
            return DVMTypes::Operand::REGISTER;
        }

        std::uint8_t Instruction51l::get_destination()
        {
            return vAA;
        }

        /**
         * PackedSwitch
         */
        PackedSwitch::PackedSwitch(std::shared_ptr<DalvikOpcodes> dalvik_opcodes, std::istream &input_file) :
            Instruction(dalvik_opcodes, input_file)
        {
            std::uint8_t instruction_part1[8];
            std::int32_t aux;
            size_t buff_size;

            if (!KUNAI::read_data_file<std::uint8_t[8]>(instruction_part1, 8, input_file))
                throw exceptions::DisassemblerException("Error disassembling PackedSwitch");
            
            this->ident = *(reinterpret_cast<std::uint16_t*>(&instruction_part1[0]));
            this->size = *(reinterpret_cast<std::uint16_t*>(&instruction_part1[2]));
            this->first_key = *(reinterpret_cast<std::int32_t*>(&instruction_part1[4]));

            this->set_OP(this->ident);

            buff_size = size;
            
            for (size_t i = 0; i < buff_size; i++)
            {
                if (!KUNAI::read_data_file<std::int32_t>(aux, 1, input_file))
                    throw exceptions::DisassemblerException("Error disassembling PackedSwitch");
                this->targets.push_back(aux);
            }

            this->set_length(8 + this->targets.size()*4);
        }

        PackedSwitch::~PackedSwitch()
        {
            if (!targets.empty())
                targets.clear();
        }

        std::string PackedSwitch::get_output()
        {
            std::stringstream str;

            str << "(size)" << size << " (first/last key)" << std::hex << first_key << " [";

            for (size_t i = 0; i < targets.size(); i++)
            {
                str << "0x" << std::hex << targets[i];
                if (i != (targets.size() - 1))
                    str << ",";
            }

            str << "]";

            return str.str();
        }

        std::uint64_t PackedSwitch::get_raw()
        {
            return ident | size << 16 | static_cast<std::uint64_t>(first_key) << 32;
        }

        std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> PackedSwitch::get_operands()
        {
            std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> operands;

            for (size_t i = 0; i < targets.size(); i++)
                operands.push_back({DVMTypes::Operand::RAW, targets[i]});
            
            return operands;
        }

        std::int32_t PackedSwitch::get_first_key()
        {
            return first_key;
        }

        std::vector<std::int32_t> PackedSwitch::get_targets()
        {
            return targets;
        }

        /**
         * SparseSwitch
         */
        SparseSwitch::SparseSwitch(std::shared_ptr<DalvikOpcodes> dalvik_opcodes, std::istream &input_file) :
            Instruction(dalvik_opcodes, input_file)
        {
            std::uint8_t instruction_part1[4];
            std::int32_t aux;
            std::vector<std::int32_t> keys;
            std::vector<std::int32_t> targets;

            if (!KUNAI::read_data_file<std::uint8_t[4]>(instruction_part1, 4, input_file))
                throw exceptions::DisassemblerException("Error disassembling SparseSwitch");
            
            this->ident = *(reinterpret_cast<std::uint16_t*>(&instruction_part1[0]));
            this->size = *(reinterpret_cast<std::uint16_t*>(&instruction_part1[2]));
            
            this->set_OP(this->ident);

            for(size_t i = 0; i < this->size; i++)
            {
                if (!KUNAI::read_data_file<std::int32_t>(aux, sizeof(std::int32_t), input_file))
                    throw exceptions::DisassemblerException("Error disassembling SparseSwitch");
                
                keys.push_back(aux);
            }

            for(size_t i = 0; i < this->size; i++)
            {
                if (!KUNAI::read_data_file<std::int32_t>(aux, sizeof(std::int32_t), input_file))
                    throw exceptions::DisassemblerException("Error disassembling SparseSwitch");
                
                targets.push_back(aux);
            }

            for (size_t i = 0; i < this->size; i++)
                this->keys_targets.push_back({keys[i], targets[i]});

            this->set_length(4 + (sizeof(std::int32_t)*this->size)*2);
        }

        SparseSwitch::~SparseSwitch()
        {
            if (!keys_targets.empty())
                keys_targets.clear();
        }

        std::string SparseSwitch::get_output()
        {
            std::stringstream str;

            str << "(size)" << size << " [";

            for(size_t i = 0; i < keys_targets.size(); i++)
            {
                if (std::get<0>(keys_targets[i])  < 0)
                    str << "-0x" << std::hex << std::get<0>(keys_targets[i]) << ":";
                else
                    str << "0x" << std::hex << std::get<0>(keys_targets[i]) << ":";
                if (std::get<1>(keys_targets[i])  < 0)
                    str << "-0x" << std::hex << std::get<1>(keys_targets[i]);
                else
                    str << "0x" << std::hex << std::get<1>(keys_targets[i]);

                if (i != (keys_targets.size() - 1))
                    str << ",";       
            }

            str << "]";
            
            return str.str();
        }

        // probably need to change get_raw for returning
        // an array of bytes
        std::uint64_t SparseSwitch::get_raw()
        {
            return ident | size << 16;
        }

        std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> SparseSwitch::get_operands()
        {
            std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> operands;

            for (size_t i = 0; i < keys_targets.size(); i++)
            {
                operands.push_back({DVMTypes::Operand::LITERAL, std::get<0>(keys_targets[i])});
                operands.push_back({DVMTypes::Operand::OFFSET, std::get<1>(keys_targets[i])});
            }

            return operands;
        }

        std::int32_t SparseSwitch::get_target_by_key(std::int32_t key)
        {
            for (size_t i = 0; i < keys_targets.size(); i++)
            {
                if (std::get<0>(keys_targets[i]) == key)
                    return std::get<1>(keys_targets[i]);
            }

            return 0;
        }
        
        std::int32_t SparseSwitch::get_key_by_pos(size_t pos)
        {
            if (pos >= keys_targets.size())
                return 0;
            return std::get<0>(keys_targets[pos]);
        }

        std::int32_t SparseSwitch::get_target_by_pos(size_t pos)
        {
            if (pos >= keys_targets.size())
                return 0;
            return std::get<1>(keys_targets[pos]);
        }

        std::vector<std::tuple<std::int32_t, std::int32_t>> SparseSwitch::get_keys_targets()
        {
            return keys_targets;
        }

        /**
         * FillArrayData
         */
        FillArrayData::FillArrayData(std::shared_ptr<DalvikOpcodes> dalvik_opcodes, std::istream &input_file) :
            Instruction(dalvik_opcodes, input_file)
        {
            std::uint8_t instruction_part1[8];
            std::uint8_t aux;
            size_t buff_size;

            if (!KUNAI::read_data_file<std::uint8_t[8]>(instruction_part1, 8, input_file))
                throw exceptions::DisassemblerException("Error disassembling FillArrayData");
            
            this->ident = *(reinterpret_cast<std::uint16_t*>(&instruction_part1[0]));
            this->element_width = *(reinterpret_cast<std::uint16_t*>(&instruction_part1[2]));
            this->size = *(reinterpret_cast<std::uint32_t*>(&instruction_part1[4]));

            this->set_OP(this->ident);

            buff_size = this->element_width*this->size;
            if (buff_size % 2 != 0)
                buff_size += 1;

            for (size_t i = 0; i < buff_size; i++)
            {
                if (!KUNAI::read_data_file<std::uint8_t>(aux, 1, input_file))
                    throw exceptions::DisassemblerException("Error disassembling FillArrayData");
                data.push_back(aux);
            }

            this->set_length(8 + buff_size);
        }

        FillArrayData::~FillArrayData()
        {
            if (!data.empty())
                data.clear();
        }

        std::string FillArrayData::get_output()
        {
            std::stringstream str;

            str << "(width)" << element_width << " (size)" << size << " [";
            
            for (size_t i = 0; i < data.size(); i++)
            {
                str << "0x" << std::hex << static_cast<std::uint32_t>(data[i]);
                if (i != (data.size() - 1))
                    str << ",";       
            }

            str << "]";
            
            return str.str();
        }

        std::uint64_t FillArrayData::get_raw()
        {
            return ident | element_width << 16 | static_cast<std::uint64_t>(size) << 32;
        }

        std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> FillArrayData::get_operands()
        {
            std::vector<std::tuple<DVMTypes::Operand, std::uint64_t>> operands;

            for (size_t i = 0; i < data.size(); i++)
                operands.push_back({DVMTypes::Operand::RAW, data[i]});
            
            return operands;
        }

        std::vector<std::uint8_t> FillArrayData::get_data()
        {
            return data;
        }


        std::shared_ptr<Instruction> get_instruction_object(std::uint32_t opcode, std::shared_ptr<DalvikOpcodes> dalvik_opcodes, std::istream &input_file)
        {
            std::shared_ptr<Instruction> instruction;

            switch (opcode)
            {
            case DVMTypes::Opcode::OP_NOP:
            {
                auto current_offset = input_file.tellg();
                char byte[2];
                
                input_file.read(byte, 2);
                
                input_file.seekg(current_offset);

                if (byte[1] == 0x03)
                    instruction = std::make_shared<FillArrayData>(dalvik_opcodes, input_file); // filled-array-data 
                else if (byte[1] == 0x01)
                    instruction = std::make_shared<PackedSwitch>(dalvik_opcodes, input_file); // packed-switch-data
                else if (byte[1] == 0x02)
                    instruction = std::make_shared<SparseSwitch>(dalvik_opcodes, input_file); // sparse-switch-data
                else
                    instruction = std::make_shared<Instruction10x>(dalvik_opcodes, input_file); // "nop"
                break;
            }
            case DVMTypes::Opcode::OP_MOVE:
                instruction = std::make_shared<Instruction12x>(dalvik_opcodes, input_file); // "move"
                break;
            case DVMTypes::Opcode::OP_MOVE_FROM16:
                instruction = std::make_shared<Instruction22x>(dalvik_opcodes, input_file); // "move/from16"
                break;
            case DVMTypes::Opcode::OP_MOVE_16:
                instruction = std::make_shared<Instruction32x>(dalvik_opcodes, input_file); // move/16
                break;
            case DVMTypes::Opcode::OP_MOVE_WIDE:
                instruction = std::make_shared<Instruction12x>(dalvik_opcodes, input_file); // move-wide
                break;
            case DVMTypes::Opcode::OP_MOVE_WIDE_FROM16:
                instruction = std::make_shared<Instruction22x>(dalvik_opcodes, input_file); // move-wide/from16
                break;
            case DVMTypes::Opcode::OP_MOVE_WIDE_16:
                instruction = std::make_shared<Instruction32x>(dalvik_opcodes, input_file); // move-wide/16
                break;
            case DVMTypes::Opcode::OP_MOVE_OBJECT:
                instruction = std::make_shared<Instruction12x>(dalvik_opcodes, input_file); // move-object
                break;
            case DVMTypes::Opcode::OP_MOVE_OBJECT_FROM16:
                instruction = std::make_shared<Instruction22x>(dalvik_opcodes, input_file); // "move-object/from16"
                break;
            case DVMTypes::Opcode::OP_MOVE_OBJECT_16:
                instruction = std::make_shared<Instruction32x>(dalvik_opcodes, input_file); // "move-object/16"
                break;
            case DVMTypes::Opcode::OP_MOVE_RESULT:
                instruction = std::make_shared<Instruction11x>(dalvik_opcodes, input_file); // "move-result"
                break;
            case DVMTypes::Opcode::OP_MOVE_RESULT_WIDE:
                instruction = std::make_shared<Instruction11x>(dalvik_opcodes, input_file); // "move-result-wide"
                break;
            case DVMTypes::Opcode::OP_MOVE_RESULT_OBJECT:
                instruction = std::make_shared<Instruction11x>(dalvik_opcodes, input_file); // "move-result-object"
                break;
            case DVMTypes::Opcode::OP_MOVE_EXCEPTION:
                instruction = std::make_shared<Instruction11x>(dalvik_opcodes, input_file); // "move-exception"
                break;
            case DVMTypes::Opcode::OP_RETURN_VOID:
                instruction = std::make_shared<Instruction10x>(dalvik_opcodes, input_file); // "return-void"
                break;
            case DVMTypes::Opcode::OP_RETURN:
                instruction = std::make_shared<Instruction11x>(dalvik_opcodes, input_file); // "return"
                break;
            case DVMTypes::Opcode::OP_RETURN_WIDE:
                instruction = std::make_shared<Instruction11x>(dalvik_opcodes, input_file); // "return-wide"
                break;
            case DVMTypes::Opcode::OP_RETURN_OBJECT:
                instruction = std::make_shared<Instruction11x>(dalvik_opcodes, input_file); // "return-object"
                break;
            case DVMTypes::Opcode::OP_CONST_4:
                instruction = std::make_shared<Instruction11n>(dalvik_opcodes, input_file); // "const/4"
                break;
            case DVMTypes::Opcode::OP_CONST_16:
                instruction = std::make_shared<Instruction21s>(dalvik_opcodes, input_file); // "const/16"
                break;
            case DVMTypes::Opcode::OP_CONST:
                instruction = std::make_shared<Instruction31i>(dalvik_opcodes, input_file); // "const"
                break;
            case DVMTypes::Opcode::OP_CONST_HIGH16:
                instruction = std::make_shared<Instruction21h>(dalvik_opcodes, input_file); // "const/high16"
                break;
            case DVMTypes::Opcode::OP_CONST_WIDE_16:
                instruction = std::make_shared<Instruction21s>(dalvik_opcodes, input_file); // "const-wide/16"
                break;
            case DVMTypes::Opcode::OP_CONST_WIDE_32:
                instruction = std::make_shared<Instruction31i>(dalvik_opcodes, input_file); // "const-wide/32"
                break;
            case DVMTypes::Opcode::OP_CONST_WIDE:
                instruction = std::make_shared<Instruction51l>(dalvik_opcodes, input_file); // "const-wide"
                break;
            case DVMTypes::Opcode::OP_CONST_WIDE_HIGH16:
                instruction = std::make_shared<Instruction21h>(dalvik_opcodes, input_file); // "const-wide/high16"
                break;
            case DVMTypes::Opcode::OP_CONST_STRING:
                instruction = std::make_shared<Instruction21c>(dalvik_opcodes, input_file); // "const-string", Kind.STRING
                break;
            case DVMTypes::Opcode::OP_CONST_STRING_JUMBO:
                instruction = std::make_shared<Instruction31c>(dalvik_opcodes, input_file); // "const-string/jumbo", Kind.STRING
                break;
            case DVMTypes::Opcode::OP_CONST_CLASS:
                instruction = std::make_shared<Instruction21c>(dalvik_opcodes, input_file); // "const-class", Kind.TYPE
                break;
            case DVMTypes::Opcode::OP_MONITOR_ENTER:
                instruction = std::make_shared<Instruction11x>(dalvik_opcodes, input_file); // "monitor-enter"
                break;
            case DVMTypes::Opcode::OP_MONITOR_EXIT:
                instruction = std::make_shared<Instruction11x>(dalvik_opcodes, input_file); // "monitor-exit"
                break;
            case DVMTypes::Opcode::OP_CHECK_CAST:
                instruction = std::make_shared<Instruction21c>(dalvik_opcodes, input_file); // "check-cast", Kind.TYPE
                break;
            case DVMTypes::Opcode::OP_INSTANCE_OF:
                instruction = std::make_shared<Instruction22c>(dalvik_opcodes, input_file); // "instance-of", Kind.TYPE
                break;
            case DVMTypes::Opcode::OP_ARRAY_LENGTH:
                instruction = std::make_shared<Instruction12x>(dalvik_opcodes, input_file); // "array-length"
                break;
            case DVMTypes::Opcode::OP_NEW_INSTANCE:
                instruction = std::make_shared<Instruction21c>(dalvik_opcodes, input_file); // "new-instance", Kind.TYPE
                break;
            case DVMTypes::Opcode::OP_NEW_ARRAY:
                instruction = std::make_shared<Instruction22c>(dalvik_opcodes, input_file); // "new-array", Kind.TYPE
                break;
            case DVMTypes::Opcode::OP_FILLED_NEW_ARRAY:
                instruction = std::make_shared<Instruction35c>(dalvik_opcodes, input_file); // "filled-new-array", Kind.TYPE
                break;
            case DVMTypes::Opcode::OP_FILLED_NEW_ARRAY_RANGE:
                instruction = std::make_shared<Instruction3rc>(dalvik_opcodes, input_file); // "filled-new-array/range", Kind.TYPE
                break;
            case DVMTypes::Opcode::OP_FILL_ARRAY_DATA:
                instruction = std::make_shared<Instruction31t>(dalvik_opcodes, input_file); // "fill-array-data"
                break;
            case DVMTypes::Opcode::OP_THROW:
                instruction = std::make_shared<Instruction11x>(dalvik_opcodes, input_file); // "throw"
                break;
            case DVMTypes::Opcode::OP_GOTO:
                instruction = std::make_shared<Instruction10t>(dalvik_opcodes, input_file); // "goto"
                break;
            case DVMTypes::Opcode::OP_GOTO_16:
                instruction = std::make_shared<Instruction20t>(dalvik_opcodes, input_file); // "goto/16"
                break;
            case DVMTypes::Opcode::OP_GOTO_32:
                instruction = std::make_shared<Instruction30t>(dalvik_opcodes, input_file); // "goto/32"
                break;
            case DVMTypes::Opcode::OP_PACKED_SWITCH:
                instruction = std::make_shared<Instruction31t>(dalvik_opcodes, input_file); // "packed-switch"
                break;
            case DVMTypes::Opcode::OP_SPARSE_SWITCH:
                instruction = std::make_shared<Instruction31t>(dalvik_opcodes, input_file); // "sparse-switch"
                break;
            case DVMTypes::Opcode::OP_CMPL_FLOAT:
                instruction = std::make_shared<Instruction23x>(dalvik_opcodes, input_file); // "cmpl-float"
                break;
            case DVMTypes::Opcode::OP_CMPG_FLOAT:
                instruction = std::make_shared<Instruction23x>(dalvik_opcodes, input_file); // "cmpg-float"
                break;
            case DVMTypes::Opcode::OP_CMPL_DOUBLE:
                instruction = std::make_shared<Instruction23x>(dalvik_opcodes, input_file); // "cmpl-double"
                break;
            case DVMTypes::Opcode::OP_CMPG_DOUBLE:
                instruction = std::make_shared<Instruction23x>(dalvik_opcodes, input_file); // "cmpg-double"
                break;
            case DVMTypes::Opcode::OP_CMP_LONG:
                instruction = std::make_shared<Instruction23x>(dalvik_opcodes, input_file); // "cmp-long"
                break;
            case DVMTypes::Opcode::OP_IF_EQ:
                instruction = std::make_shared<Instruction22t>(dalvik_opcodes, input_file); // "if-eq"
                break;
            case DVMTypes::Opcode::OP_IF_NE:
                instruction = std::make_shared<Instruction22t>(dalvik_opcodes, input_file); // "if-ne"
                break;
            case DVMTypes::Opcode::OP_IF_LT:
                instruction = std::make_shared<Instruction22t>(dalvik_opcodes, input_file); // "if-lt"
                break;
            case DVMTypes::Opcode::OP_IF_GE:
                instruction = std::make_shared<Instruction22t>(dalvik_opcodes, input_file); // "if-ge"
                break;
            case DVMTypes::Opcode::OP_IF_GT:
                instruction = std::make_shared<Instruction22t>(dalvik_opcodes, input_file); // "if-gt"
                break;
            case DVMTypes::Opcode::OP_IF_LE:
                instruction = std::make_shared<Instruction22t>(dalvik_opcodes, input_file); // "if-le"
                break;
            case DVMTypes::Opcode::OP_IF_EQZ:
                instruction = std::make_shared<Instruction21t>(dalvik_opcodes, input_file); // "if-eqz"
                break;
            case DVMTypes::Opcode::OP_IF_NEZ:
                instruction = std::make_shared<Instruction21t>(dalvik_opcodes, input_file); // "if-nez"
                break;
            case DVMTypes::Opcode::OP_IF_LTZ:
                instruction = std::make_shared<Instruction21t>(dalvik_opcodes, input_file); // "if-ltz"
                break;
            case DVMTypes::Opcode::OP_IF_GEZ:
                instruction = std::make_shared<Instruction21t>(dalvik_opcodes, input_file); // "if-gez"
                break;
            case DVMTypes::Opcode::OP_IF_GTZ:
                instruction = std::make_shared<Instruction21t>(dalvik_opcodes, input_file); // "if-gtz"
                break;
            case DVMTypes::Opcode::OP_IF_LEZ:
                instruction = std::make_shared<Instruction21t>(dalvik_opcodes, input_file); // "if-lez"
                break;
            case DVMTypes::Opcode::OP_UNUSED_3E:
                instruction = std::make_shared<Instruction00x>(dalvik_opcodes, input_file); // "unused"
                break;
            case DVMTypes::Opcode::OP_UNUSED_3F:
                instruction = std::make_shared<Instruction00x>(dalvik_opcodes, input_file); // "unused"
                break;
            case DVMTypes::Opcode::OP_UNUSED_40:
                instruction = std::make_shared<Instruction00x>(dalvik_opcodes, input_file); // "unused"
                break;
            case DVMTypes::Opcode::OP_UNUSED_41:
                instruction = std::make_shared<Instruction00x>(dalvik_opcodes, input_file); // "unused"
                break;
            case DVMTypes::Opcode::OP_UNUSED_42:
                instruction = std::make_shared<Instruction00x>(dalvik_opcodes, input_file); // "unused"
                break;
            case DVMTypes::Opcode::OP_UNUSED_43:
                instruction = std::make_shared<Instruction00x>(dalvik_opcodes, input_file); // "unused"
                break;
            case DVMTypes::Opcode::OP_AGET:
                instruction = std::make_shared<Instruction23x>(dalvik_opcodes, input_file); // "aget"
                break;
            case DVMTypes::Opcode::OP_AGET_WIDE:
                instruction = std::make_shared<Instruction23x>(dalvik_opcodes, input_file); // "aget-wide"
                break;
            case DVMTypes::Opcode::OP_AGET_OBJECT:
                instruction = std::make_shared<Instruction23x>(dalvik_opcodes, input_file); // "aget-object"
                break;
            case DVMTypes::Opcode::OP_AGET_BOOLEAN:
                instruction = std::make_shared<Instruction23x>(dalvik_opcodes, input_file); // "aget-boolean"
                break;
            case DVMTypes::Opcode::OP_AGET_BYTE:
                instruction = std::make_shared<Instruction23x>(dalvik_opcodes, input_file); // "aget-byte"
                break;
            case DVMTypes::Opcode::OP_AGET_CHAR:
                instruction = std::make_shared<Instruction23x>(dalvik_opcodes, input_file); // "aget-char"
                break;
            case DVMTypes::Opcode::OP_AGET_SHORT:
                instruction = std::make_shared<Instruction23x>(dalvik_opcodes, input_file); // "aget-short"
                break;
            case DVMTypes::Opcode::OP_APUT:
                instruction = std::make_shared<Instruction23x>(dalvik_opcodes, input_file); // "aput"
                break;
            case DVMTypes::Opcode::OP_APUT_WIDE:
                instruction = std::make_shared<Instruction23x>(dalvik_opcodes, input_file); // "aput-wide"
                break;
            case DVMTypes::Opcode::OP_APUT_OBJECT:
                instruction = std::make_shared<Instruction23x>(dalvik_opcodes, input_file); // "aput-object"
                break;
            case DVMTypes::Opcode::OP_APUT_BOOLEAN:
                instruction = std::make_shared<Instruction23x>(dalvik_opcodes, input_file); // "aput-boolean"
                break;
            case DVMTypes::Opcode::OP_APUT_BYTE:
                instruction = std::make_shared<Instruction23x>(dalvik_opcodes, input_file); // "aput-byte"
                break;
            case DVMTypes::Opcode::OP_APUT_CHAR:
                instruction = std::make_shared<Instruction23x>(dalvik_opcodes, input_file); // "aput-char"
                break;
            case DVMTypes::Opcode::OP_APUT_SHORT:
                instruction = std::make_shared<Instruction23x>(dalvik_opcodes, input_file); // "aput-short"
                break;
            case DVMTypes::Opcode::OP_IGET:
                instruction = std::make_shared<Instruction22c>(dalvik_opcodes, input_file); // "iget"
                break;
            case DVMTypes::Opcode::OP_IGET_WIDE:
                instruction = std::make_shared<Instruction22c>(dalvik_opcodes, input_file); // "iget-wide"
                break;
            case DVMTypes::Opcode::OP_IGET_OBJECT:
                instruction = std::make_shared<Instruction22c>(dalvik_opcodes, input_file); // "iget-object"
                break;
            case DVMTypes::Opcode::OP_IGET_BOOLEAN:
                instruction = std::make_shared<Instruction22c>(dalvik_opcodes, input_file); // "iget-boolean"
                break;
            case DVMTypes::Opcode::OP_IGET_BYTE:
                instruction = std::make_shared<Instruction22c>(dalvik_opcodes, input_file); // "iget-byte"
                break;
            case DVMTypes::Opcode::OP_IGET_CHAR:
                instruction = std::make_shared<Instruction22c>(dalvik_opcodes, input_file); // "iget-char"
                break;
            case DVMTypes::Opcode::OP_IGET_SHORT:
                instruction = std::make_shared<Instruction22c>(dalvik_opcodes, input_file); // "iget-short"
                break;
            case DVMTypes::Opcode::OP_IPUT:
                instruction = std::make_shared<Instruction22c>(dalvik_opcodes, input_file); // "iput"
                break;
            case DVMTypes::Opcode::OP_IPUT_WIDE:
                instruction = std::make_shared<Instruction22c>(dalvik_opcodes, input_file); // "iput-wide"
                break;
            case DVMTypes::Opcode::OP_IPUT_OBJECT:
                instruction = std::make_shared<Instruction22c>(dalvik_opcodes, input_file); // "iput-object"
                break;
            case DVMTypes::Opcode::OP_IPUT_BOOLEAN:
                instruction = std::make_shared<Instruction22c>(dalvik_opcodes, input_file); // "iput-boolean"
                break;
            case DVMTypes::Opcode::OP_IPUT_BYTE:
                instruction = std::make_shared<Instruction22c>(dalvik_opcodes, input_file); // "iput-byte"
                break;
            case DVMTypes::Opcode::OP_IPUT_CHAR:
                instruction = std::make_shared<Instruction22c>(dalvik_opcodes, input_file); // "iput-char"
                break;
            case DVMTypes::Opcode::OP_IPUT_SHORT:
                instruction = std::make_shared<Instruction22c>(dalvik_opcodes, input_file); // "iput-short"
                break;
            case DVMTypes::Opcode::OP_SGET:
                instruction = std::make_shared<Instruction21c>(dalvik_opcodes, input_file); // "sget"
                break;
            case DVMTypes::Opcode::OP_SGET_WIDE:
                instruction = std::make_shared<Instruction21c>(dalvik_opcodes, input_file); // "sget-wide"
                break;
            case DVMTypes::Opcode::OP_SGET_OBJECT:
                instruction = std::make_shared<Instruction21c>(dalvik_opcodes, input_file); // "sget-object"
                break;
            case DVMTypes::Opcode::OP_SGET_BOOLEAN:
                instruction = std::make_shared<Instruction21c>(dalvik_opcodes, input_file); // "sget-boolean"
                break;
            case DVMTypes::Opcode::OP_SGET_BYTE:
                instruction = std::make_shared<Instruction21c>(dalvik_opcodes, input_file); // "sget-byte"
                break;
            case DVMTypes::Opcode::OP_SGET_CHAR:
                instruction = std::make_shared<Instruction21c>(dalvik_opcodes, input_file); // "sget-char"
                break;
            case DVMTypes::Opcode::OP_SGET_SHORT:
                instruction = std::make_shared<Instruction21c>(dalvik_opcodes, input_file); // "sget-short"
                break;
            case DVMTypes::Opcode::OP_SPUT:
                instruction = std::make_shared<Instruction21c>(dalvik_opcodes, input_file); // "sput"
                break;
            case DVMTypes::Opcode::OP_SPUT_WIDE:
                instruction = std::make_shared<Instruction21c>(dalvik_opcodes, input_file); // "sput-wide"
                break;
            case DVMTypes::Opcode::OP_SPUT_OBJECT:
                instruction = std::make_shared<Instruction21c>(dalvik_opcodes, input_file); // "sput-object"
                break;
            case DVMTypes::Opcode::OP_SPUT_BOOLEAN:
                instruction = std::make_shared<Instruction21c>(dalvik_opcodes, input_file); // "sput-boolean"
                break;
            case DVMTypes::Opcode::OP_SPUT_BYTE:
                instruction = std::make_shared<Instruction21c>(dalvik_opcodes, input_file); // "sput-byte"
                break;
            case DVMTypes::Opcode::OP_SPUT_CHAR:
                instruction = std::make_shared<Instruction21c>(dalvik_opcodes, input_file); // "sput-char"
                break;
            case DVMTypes::Opcode::OP_SPUT_SHORT:
                instruction = std::make_shared<Instruction21c>(dalvik_opcodes, input_file); // "sput-short"
                break;
            case DVMTypes::Opcode::OP_INVOKE_VIRTUAL:
                instruction = std::make_shared<Instruction35c>(dalvik_opcodes, input_file); // "invoke-virtual"
                break;
            case DVMTypes::Opcode::OP_INVOKE_SUPER:
                instruction = std::make_shared<Instruction35c>(dalvik_opcodes, input_file); // "invoke-super"
                break;
            case DVMTypes::Opcode::OP_INVOKE_DIRECT:
                instruction = std::make_shared<Instruction35c>(dalvik_opcodes, input_file); // "invoke-direct"
                break;
            case DVMTypes::Opcode::OP_INVOKE_STATIC:
                instruction = std::make_shared<Instruction35c>(dalvik_opcodes, input_file); // "invoke-static"
                break;
            case DVMTypes::Opcode::OP_INVOKE_INTERFACE:
                instruction = std::make_shared<Instruction35c>(dalvik_opcodes, input_file); // "invoke-interface"
                break;
            case DVMTypes::Opcode::OP_UNUSED_73:
                instruction = std::make_shared<Instruction00x>(dalvik_opcodes, input_file); // "unused"
                break;
            case DVMTypes::Opcode::OP_INVOKE_VIRTUAL_RANGE:
                instruction = std::make_shared<Instruction3rc>(dalvik_opcodes, input_file); // "invoke-virtual/range"
                break;
            case DVMTypes::Opcode::OP_INVOKE_SUPER_RANGE:
                instruction = std::make_shared<Instruction3rc>(dalvik_opcodes, input_file); // "invoke-super/range"
                break;
            case DVMTypes::Opcode::OP_INVOKE_DIRECT_RANGE:
                instruction = std::make_shared<Instruction3rc>(dalvik_opcodes, input_file); // "invoke-direct/range"
                break;
            case DVMTypes::Opcode::OP_INVOKE_STATIC_RANGE:
                instruction = std::make_shared<Instruction3rc>(dalvik_opcodes, input_file); // "invoke-static/range"
                break;
            case DVMTypes::Opcode::OP_INVOKE_INTERFACE_RANGE:
                instruction = std::make_shared<Instruction3rc>(dalvik_opcodes, input_file); // "invoke-interface/range"
                break;
            case DVMTypes::Opcode::OP_UNUSED_79:
                instruction = std::make_shared<Instruction00x>(dalvik_opcodes, input_file); // "unused"
                break;
            case DVMTypes::Opcode::OP_UNUSED_7A:
                instruction = std::make_shared<Instruction00x>(dalvik_opcodes, input_file); // "unused"
                break;
            case DVMTypes::Opcode::OP_NEG_INT:
                instruction = std::make_shared<Instruction12x>(dalvik_opcodes, input_file); // "neg-int"
                break;
            case DVMTypes::Opcode::OP_NOT_INT:
                instruction = std::make_shared<Instruction12x>(dalvik_opcodes, input_file); // "not-int"
                break;
            case DVMTypes::Opcode::OP_NEG_LONG:
                instruction = std::make_shared<Instruction12x>(dalvik_opcodes, input_file); // "neg-long"
                break;
            case DVMTypes::Opcode::OP_NOT_LONG:
                instruction = std::make_shared<Instruction12x>(dalvik_opcodes, input_file); // "not-long"
                break;
            case DVMTypes::Opcode::OP_NEG_FLOAT:
                instruction = std::make_shared<Instruction12x>(dalvik_opcodes, input_file); // "neg-float"
                break;
            case DVMTypes::Opcode::OP_NEG_DOUBLE:
                instruction = std::make_shared<Instruction12x>(dalvik_opcodes, input_file); // "neg-double"
                break;
            case DVMTypes::Opcode::OP_INT_TO_LONG:
                instruction = std::make_shared<Instruction12x>(dalvik_opcodes, input_file); // "int-to-long"
                break;
            case DVMTypes::Opcode::OP_INT_TO_FLOAT:
                instruction = std::make_shared<Instruction12x>(dalvik_opcodes, input_file); // "int-to-float"
                break;
            case DVMTypes::Opcode::OP_INT_TO_DOUBLE:
                instruction = std::make_shared<Instruction12x>(dalvik_opcodes, input_file); // "int-to-double"
                break;
            case DVMTypes::Opcode::OP_LONG_TO_INT:
                instruction = std::make_shared<Instruction12x>(dalvik_opcodes, input_file); // "long-to-int"
                break;
            case DVMTypes::Opcode::OP_LONG_TO_FLOAT:
                instruction = std::make_shared<Instruction12x>(dalvik_opcodes, input_file); // "long-to-float"
                break;
            case DVMTypes::Opcode::OP_LONG_TO_DOUBLE:
                instruction = std::make_shared<Instruction12x>(dalvik_opcodes, input_file); // "long-to-double"
                break;
            case DVMTypes::Opcode::OP_FLOAT_TO_INT:
                instruction = std::make_shared<Instruction12x>(dalvik_opcodes, input_file); // "float-to-int"
                break;
            case DVMTypes::Opcode::OP_FLOAT_TO_LONG:
                instruction = std::make_shared<Instruction12x>(dalvik_opcodes, input_file); // "float-to-long"
                break;
            case DVMTypes::Opcode::OP_FLOAT_TO_DOUBLE:
                instruction = std::make_shared<Instruction12x>(dalvik_opcodes, input_file); // "float-to-double"
                break;
            case DVMTypes::Opcode::OP_DOUBLE_TO_INT:
                instruction = std::make_shared<Instruction12x>(dalvik_opcodes, input_file); // "double-to-int"
                break;
            case DVMTypes::Opcode::OP_DOUBLE_TO_LONG:
                instruction = std::make_shared<Instruction12x>(dalvik_opcodes, input_file); // "double-to-long"
                break;
            case DVMTypes::Opcode::OP_DOUBLE_TO_FLOAT:
                instruction = std::make_shared<Instruction12x>(dalvik_opcodes, input_file); // "double-to-float"
                break;
            case DVMTypes::Opcode::OP_INT_TO_BYTE:
                instruction = std::make_shared<Instruction12x>(dalvik_opcodes, input_file); // "int-to-byte"
                break;
            case DVMTypes::Opcode::OP_INT_TO_CHAR:
                instruction = std::make_shared<Instruction12x>(dalvik_opcodes, input_file); // "int-to-char"
                break;
            case DVMTypes::Opcode::OP_INT_TO_SHORT:
                instruction = std::make_shared<Instruction12x>(dalvik_opcodes, input_file); // "int-to-short"
                break;
            case DVMTypes::Opcode::OP_ADD_INT:
                instruction = std::make_shared<Instruction23x>(dalvik_opcodes, input_file); // "add-int"
                break;
            case DVMTypes::Opcode::OP_SUB_INT:
                instruction = std::make_shared<Instruction23x>(dalvik_opcodes, input_file); // "sub-int"
                break;
            case DVMTypes::Opcode::OP_MUL_INT:
                instruction = std::make_shared<Instruction23x>(dalvik_opcodes, input_file); // "mul-int"
                break;
            case DVMTypes::Opcode::OP_DIV_INT:
                instruction = std::make_shared<Instruction23x>(dalvik_opcodes, input_file); // "div-int"
                break;
            case DVMTypes::Opcode::OP_REM_INT:
                instruction = std::make_shared<Instruction23x>(dalvik_opcodes, input_file); // "rem-int"
                break;
            case DVMTypes::Opcode::OP_AND_INT:
                instruction = std::make_shared<Instruction23x>(dalvik_opcodes, input_file); // "and-int"
                break;
            case DVMTypes::Opcode::OP_OR_INT:
                instruction = std::make_shared<Instruction23x>(dalvik_opcodes, input_file); // "or-int"
                break;
            case DVMTypes::Opcode::OP_XOR_INT:
                instruction = std::make_shared<Instruction23x>(dalvik_opcodes, input_file); // "xor-int"
                break;
            case DVMTypes::Opcode::OP_SHL_INT:
                instruction = std::make_shared<Instruction23x>(dalvik_opcodes, input_file); // "shl-int"
                break;
            case DVMTypes::Opcode::OP_SHR_INT:
                instruction = std::make_shared<Instruction23x>(dalvik_opcodes, input_file); // "shr-int"
                break;
            case DVMTypes::Opcode::OP_USHR_INT:
                instruction = std::make_shared<Instruction23x>(dalvik_opcodes, input_file); // "ushr-int"
                break;
            case DVMTypes::Opcode::OP_ADD_LONG:
                instruction = std::make_shared<Instruction23x>(dalvik_opcodes, input_file); // "add-long"
                break;
            case DVMTypes::Opcode::OP_SUB_LONG:
                instruction = std::make_shared<Instruction23x>(dalvik_opcodes, input_file); // "sub-long"
                break;
            case DVMTypes::Opcode::OP_MUL_LONG:
                instruction = std::make_shared<Instruction23x>(dalvik_opcodes, input_file); // "mul-long"
                break;
            case DVMTypes::Opcode::OP_DIV_LONG:
                instruction = std::make_shared<Instruction23x>(dalvik_opcodes, input_file); // "div-long"
                break;
            case DVMTypes::Opcode::OP_REM_LONG:
                instruction = std::make_shared<Instruction23x>(dalvik_opcodes, input_file); // "rem-long"
                break;
            case DVMTypes::Opcode::OP_AND_LONG:
                instruction = std::make_shared<Instruction23x>(dalvik_opcodes, input_file); // "and-long"
                break;
            case DVMTypes::Opcode::OP_OR_LONG:
                instruction = std::make_shared<Instruction23x>(dalvik_opcodes, input_file); // "or-long"
                break;
            case DVMTypes::Opcode::OP_XOR_LONG:
                instruction = std::make_shared<Instruction23x>(dalvik_opcodes, input_file); // "xor-long"
                break;
            case DVMTypes::Opcode::OP_SHL_LONG:
                instruction = std::make_shared<Instruction23x>(dalvik_opcodes, input_file); // "shl-long"
                break;
            case DVMTypes::Opcode::OP_SHR_LONG:
                instruction = std::make_shared<Instruction23x>(dalvik_opcodes, input_file); // "shr-long"
                break;
            case DVMTypes::Opcode::OP_USHR_LONG:
                instruction = std::make_shared<Instruction23x>(dalvik_opcodes, input_file); // "ushr-long"
                break;
            case DVMTypes::Opcode::OP_ADD_FLOAT:
                instruction = std::make_shared<Instruction23x>(dalvik_opcodes, input_file); // "add-float"
                break;
            case DVMTypes::Opcode::OP_SUB_FLOAT:
                instruction = std::make_shared<Instruction23x>(dalvik_opcodes, input_file); // "sub-float"
                break;
            case DVMTypes::Opcode::OP_MUL_FLOAT:
                instruction = std::make_shared<Instruction23x>(dalvik_opcodes, input_file); // "mul-float"
                break;
            case DVMTypes::Opcode::OP_DIV_FLOAT:
                instruction = std::make_shared<Instruction23x>(dalvik_opcodes, input_file); // "div-float"
                break;
            case DVMTypes::Opcode::OP_REM_FLOAT:
                instruction = std::make_shared<Instruction23x>(dalvik_opcodes, input_file); // "rem-float"
                break;
            case DVMTypes::Opcode::OP_ADD_DOUBLE:
                instruction = std::make_shared<Instruction23x>(dalvik_opcodes, input_file); // "add-double"
                break;
            case DVMTypes::Opcode::OP_SUB_DOUBLE:
                instruction = std::make_shared<Instruction23x>(dalvik_opcodes, input_file); // "sub-double"
                break;
            case DVMTypes::Opcode::OP_MUL_DOUBLE:
                instruction = std::make_shared<Instruction23x>(dalvik_opcodes, input_file); // "mul-double"
                break;
            case DVMTypes::Opcode::OP_DIV_DOUBLE:
                instruction = std::make_shared<Instruction23x>(dalvik_opcodes, input_file); // "div-double"
                break;
            case DVMTypes::Opcode::OP_REM_DOUBLE:
                instruction = std::make_shared<Instruction23x>(dalvik_opcodes, input_file); // "rem-double"
                break;
            case DVMTypes::Opcode::OP_ADD_INT_2ADDR:
                instruction = std::make_shared<Instruction12x>(dalvik_opcodes, input_file); // "add-int/2addr"
                break;
            case DVMTypes::Opcode::OP_SUB_INT_2ADDR:
                instruction = std::make_shared<Instruction12x>(dalvik_opcodes, input_file); // "sub-int/2addr"
                break;
            case DVMTypes::Opcode::OP_MUL_INT_2ADDR:
                instruction = std::make_shared<Instruction12x>(dalvik_opcodes, input_file); // "mul-int/2addr"
                break;
            case DVMTypes::Opcode::OP_DIV_INT_2ADDR:
                instruction = std::make_shared<Instruction12x>(dalvik_opcodes, input_file); // "div-int/2addr"
                break;
            case DVMTypes::Opcode::OP_REM_INT_2ADDR:
                instruction = std::make_shared<Instruction12x>(dalvik_opcodes, input_file); // "rem-int/2addr"
                break;
            case DVMTypes::Opcode::OP_AND_INT_2ADDR:
                instruction = std::make_shared<Instruction12x>(dalvik_opcodes, input_file); // "and-int/2addr"
                break;
            case DVMTypes::Opcode::OP_OR_INT_2ADDR:
                instruction = std::make_shared<Instruction12x>(dalvik_opcodes, input_file); // "or-int/2addr"
                break;
            case DVMTypes::Opcode::OP_XOR_INT_2ADDR:
                instruction = std::make_shared<Instruction12x>(dalvik_opcodes, input_file); // "xor-int/2addr"
                break;
            case DVMTypes::Opcode::OP_SHL_INT_2ADDR:
                instruction = std::make_shared<Instruction12x>(dalvik_opcodes, input_file); // "shl-int/2addr"
                break;
            case DVMTypes::Opcode::OP_SHR_INT_2ADDR:
                instruction = std::make_shared<Instruction12x>(dalvik_opcodes, input_file); // "shr-int/2addr"
                break;
            case DVMTypes::Opcode::OP_USHR_INT_2ADDR:
                instruction = std::make_shared<Instruction12x>(dalvik_opcodes, input_file); // "ushr-int/2addr"
                break;
            case DVMTypes::Opcode::OP_ADD_LONG_2ADDR:
                instruction = std::make_shared<Instruction12x>(dalvik_opcodes, input_file); // "add-long/2addr"
                break;
            case DVMTypes::Opcode::OP_SUB_LONG_2ADDR:
                instruction = std::make_shared<Instruction12x>(dalvik_opcodes, input_file); // "sub-long/2addr"
                break;
            case DVMTypes::Opcode::OP_MUL_LONG_2ADDR:
                instruction = std::make_shared<Instruction12x>(dalvik_opcodes, input_file); // "mul-long/2addr"
                break;
            case DVMTypes::Opcode::OP_DIV_LONG_2ADDR:
                instruction = std::make_shared<Instruction12x>(dalvik_opcodes, input_file); // "div-long/2addr"
                break;
            case DVMTypes::Opcode::OP_REM_LONG_2ADDR:
                instruction = std::make_shared<Instruction12x>(dalvik_opcodes, input_file); // "rem-long/2addr"
                break;
            case DVMTypes::Opcode::OP_AND_LONG_2ADDR:
                instruction = std::make_shared<Instruction12x>(dalvik_opcodes, input_file); // "and-long/2addr"
                break;
            case DVMTypes::Opcode::OP_OR_LONG_2ADDR:
                instruction = std::make_shared<Instruction12x>(dalvik_opcodes, input_file); // "or-long/2addr"
                break;
            case DVMTypes::Opcode::OP_XOR_LONG_2ADDR:
                instruction = std::make_shared<Instruction12x>(dalvik_opcodes, input_file); // "xor-long/2addr"
                break;
            case DVMTypes::Opcode::OP_SHL_LONG_2ADDR:
                instruction = std::make_shared<Instruction12x>(dalvik_opcodes, input_file); // "shl-long/2addr"
                break;
            case DVMTypes::Opcode::OP_SHR_LONG_2ADDR:
                instruction = std::make_shared<Instruction12x>(dalvik_opcodes, input_file); // "shr-long/2addr"
                break;
            case DVMTypes::Opcode::OP_USHR_LONG_2ADDR:
                instruction = std::make_shared<Instruction12x>(dalvik_opcodes, input_file); // "ushr-long/2addr"
                break;
            case DVMTypes::Opcode::OP_ADD_FLOAT_2ADDR:
                instruction = std::make_shared<Instruction12x>(dalvik_opcodes, input_file); // "add-float/2addr"
                break;
            case DVMTypes::Opcode::OP_SUB_FLOAT_2ADDR:
                instruction = std::make_shared<Instruction12x>(dalvik_opcodes, input_file); // "sub-float/2addr"
                break;
            case DVMTypes::Opcode::OP_MUL_FLOAT_2ADDR:
                instruction = std::make_shared<Instruction12x>(dalvik_opcodes, input_file); // "mul-float/2addr"
                break;
            case DVMTypes::Opcode::OP_DIV_FLOAT_2ADDR:
                instruction = std::make_shared<Instruction12x>(dalvik_opcodes, input_file); // "div-float/2addr"
                break;
            case DVMTypes::Opcode::OP_REM_FLOAT_2ADDR:
                instruction = std::make_shared<Instruction12x>(dalvik_opcodes, input_file); // "rem-float/2addr"
                break;
            case DVMTypes::Opcode::OP_ADD_DOUBLE_2ADDR:
                instruction = std::make_shared<Instruction12x>(dalvik_opcodes, input_file); // "add-double/2addr"
                break;
            case DVMTypes::Opcode::OP_SUB_DOUBLE_2ADDR:
                instruction = std::make_shared<Instruction12x>(dalvik_opcodes, input_file); // "sub-double/2addr"
                break;
            case DVMTypes::Opcode::OP_MUL_DOUBLE_2ADDR:
                instruction = std::make_shared<Instruction12x>(dalvik_opcodes, input_file); // "mul-double/2addr"
                break;
            case DVMTypes::Opcode::OP_DIV_DOUBLE_2ADDR:
                instruction = std::make_shared<Instruction12x>(dalvik_opcodes, input_file); // "div-double/2addr"
                break;
            case DVMTypes::Opcode::OP_REM_DOUBLE_2ADDR:
                instruction = std::make_shared<Instruction12x>(dalvik_opcodes, input_file); // "rem-double/2addr"
                break;
            case DVMTypes::Opcode::OP_ADD_INT_LIT16:
                instruction = std::make_shared<Instruction22s>(dalvik_opcodes, input_file); // "add-int/lit16"
                break;
            case DVMTypes::Opcode::OP_RSUB_INT:
                instruction = std::make_shared<Instruction22s>(dalvik_opcodes, input_file); // "rsub-int"
                break;
            case DVMTypes::Opcode::OP_MUL_INT_LIT16:
                instruction = std::make_shared<Instruction22s>(dalvik_opcodes, input_file); // "mul-int/lit16"
                break;
            case DVMTypes::Opcode::OP_DIV_INT_LIT16:
                instruction = std::make_shared<Instruction22s>(dalvik_opcodes, input_file); // "div-int/lit16"
                break;
            case DVMTypes::Opcode::OP_REM_INT_LIT16:
                instruction = std::make_shared<Instruction22s>(dalvik_opcodes, input_file); // "rem-int/lit16"
                break;
            case DVMTypes::Opcode::OP_AND_INT_LIT16:
                instruction = std::make_shared<Instruction22s>(dalvik_opcodes, input_file); // "and-int/lit16"
                break;
            case DVMTypes::Opcode::OP_OR_INT_LIT16:
                instruction = std::make_shared<Instruction22s>(dalvik_opcodes, input_file); // "or-int/lit16"
                break;
            case DVMTypes::Opcode::OP_XOR_INT_LIT16:
                instruction = std::make_shared<Instruction22s>(dalvik_opcodes, input_file); // "xor-int/lit16"
                break;
            case DVMTypes::Opcode::OP_ADD_INT_LIT8:
                instruction = std::make_shared<Instruction22b>(dalvik_opcodes, input_file); // "add-int/lit8"
                break;
            case DVMTypes::Opcode::OP_RSUB_INT_LIT8:
                instruction = std::make_shared<Instruction22b>(dalvik_opcodes, input_file); // "rsub-int/lit8"
                break;
            case DVMTypes::Opcode::OP_MUL_INT_LIT8:
                instruction = std::make_shared<Instruction22b>(dalvik_opcodes, input_file); // "mul-int/lit8"
                break;
            case DVMTypes::Opcode::OP_DIV_INT_LIT8:
                instruction = std::make_shared<Instruction22b>(dalvik_opcodes, input_file); // "div-int/lit8"
                break;
            case DVMTypes::Opcode::OP_REM_INT_LIT8:
                instruction = std::make_shared<Instruction22b>(dalvik_opcodes, input_file); // "rem-int/lit8"
                break;
            case DVMTypes::Opcode::OP_AND_INT_LIT8:
                instruction = std::make_shared<Instruction22b>(dalvik_opcodes, input_file); // "and-int/lit8"
                break;
            case DVMTypes::Opcode::OP_OR_INT_LIT8:
                instruction = std::make_shared<Instruction22b>(dalvik_opcodes, input_file); // "or-int/lit8"
                break;
            case DVMTypes::Opcode::OP_XOR_INT_LIT8:
                instruction = std::make_shared<Instruction22b>(dalvik_opcodes, input_file); // "xor-int/lit8"
                break;
            case DVMTypes::Opcode::OP_SHL_INT_LIT8:
                instruction = std::make_shared<Instruction22b>(dalvik_opcodes, input_file); // "shl-int/lit8"
                break;
            case DVMTypes::Opcode::OP_SHR_INT_LIT8:
                instruction = std::make_shared<Instruction22b>(dalvik_opcodes, input_file); // "shr-int/lit8"
                break;
            case DVMTypes::Opcode::OP_USHR_INT_LIT8:
                instruction = std::make_shared<Instruction22b>(dalvik_opcodes, input_file); // "ushr-int/lit8"
                break;
            case DVMTypes::Opcode::OP_IGET_VOLATILE:
                instruction = std::make_shared<Instruction00x>(dalvik_opcodes, input_file); // "unused"
                break;
            case DVMTypes::Opcode::OP_IPUT_VOLATILE:
                instruction = std::make_shared<Instruction00x>(dalvik_opcodes, input_file); // "unused"
                break;
            case DVMTypes::Opcode::OP_SGET_VOLATILE:
                instruction = std::make_shared<Instruction00x>(dalvik_opcodes, input_file); // "unused"
                break;
            case DVMTypes::Opcode::OP_SPUT_VOLATILE:
                instruction = std::make_shared<Instruction00x>(dalvik_opcodes, input_file); // "unused"
                break;
            case DVMTypes::Opcode::OP_IGET_OBJECT_VOLATILE:
                instruction = std::make_shared<Instruction00x>(dalvik_opcodes, input_file); // "unused"
                break;
            case DVMTypes::Opcode::OP_IGET_WIDE_VOLATILE:
                instruction = std::make_shared<Instruction00x>(dalvik_opcodes, input_file); // "unused"
                break;
            case DVMTypes::Opcode::OP_IPUT_WIDE_VOLATILE:
                instruction = std::make_shared<Instruction00x>(dalvik_opcodes, input_file); // "unused"
                break;
            case DVMTypes::Opcode::OP_SGET_WIDE_VOLATILE:
                instruction = std::make_shared<Instruction00x>(dalvik_opcodes, input_file); // "unused"
                break;
            case DVMTypes::Opcode::OP_SPUT_WIDE_VOLATILE:
                instruction = std::make_shared<Instruction00x>(dalvik_opcodes, input_file); // "unused"
                break;
            case DVMTypes::Opcode::OP_BREAKPOINT:
                instruction = std::make_shared<Instruction00x>(dalvik_opcodes, input_file); // "unused"
                break;
            case DVMTypes::Opcode::OP_THROW_VERIFICATION_ERROR:
                instruction = std::make_shared<Instruction00x>(dalvik_opcodes, input_file); // "unused"
                break;
            case DVMTypes::Opcode::OP_EXECUTE_INLINE:
                instruction = std::make_shared<Instruction00x>(dalvik_opcodes, input_file); // "unused"
                break;
            case DVMTypes::Opcode::OP_EXECUTE_INLINE_RANGE:
                instruction = std::make_shared<Instruction00x>(dalvik_opcodes, input_file); // "unused"
                break;
            case DVMTypes::Opcode::OP_INVOKE_OBJECT_INIT_RANGE:
                instruction = std::make_shared<Instruction00x>(dalvik_opcodes, input_file); // "unused"
                break;
            case DVMTypes::Opcode::OP_RETURN_VOID_BARRIER:
                instruction = std::make_shared<Instruction00x>(dalvik_opcodes, input_file); // "unused"
                break;
            case DVMTypes::Opcode::OP_IGET_QUICK:
                instruction = std::make_shared<Instruction00x>(dalvik_opcodes, input_file); // "unused"
                break;
            case DVMTypes::Opcode::OP_IGET_WIDE_QUICK:
                instruction = std::make_shared<Instruction00x>(dalvik_opcodes, input_file); // "unused"
                break;
            case DVMTypes::Opcode::OP_IGET_OBJECT_QUICK:
                instruction = std::make_shared<Instruction00x>(dalvik_opcodes, input_file); // "unused"
                break;
            case DVMTypes::Opcode::OP_IPUT_QUICK:
                instruction = std::make_shared<Instruction00x>(dalvik_opcodes, input_file); // "unused"
                break;
            case DVMTypes::Opcode::OP_IPUT_WIDE_QUICK:
                instruction = std::make_shared<Instruction00x>(dalvik_opcodes, input_file); // "unused"
                break;
            case DVMTypes::Opcode::OP_IPUT_OBJECT_QUICK:
                instruction = std::make_shared<Instruction00x>(dalvik_opcodes, input_file); // "unused"
                break;
            case DVMTypes::Opcode::OP_INVOKE_VIRTUAL_QUICK:
                instruction = std::make_shared<Instruction00x>(dalvik_opcodes, input_file); // "unused"
                break;
            case DVMTypes::Opcode::OP_INVOKE_VIRTUAL_QUICK_RANGE:
                instruction = std::make_shared<Instruction00x>(dalvik_opcodes, input_file); // "unused"
                break;
            case DVMTypes::Opcode::OP_INVOKE_SUPER_QUICK:
                instruction = std::make_shared<Instruction45cc>(dalvik_opcodes, input_file); // "invoke-polymorphic" # Dalvik 038
                break;
            case DVMTypes::Opcode::OP_INVOKE_SUPER_QUICK_RANGE:
                instruction = std::make_shared<Instruction4rcc>(dalvik_opcodes, input_file); // "invoke-polymorphic/range" # Dalvik 038
                break;
            case DVMTypes::Opcode::OP_IPUT_OBJECT_VOLATILE:
                instruction = std::make_shared<Instruction35c>(dalvik_opcodes, input_file); // "invoke-custom" # Dalvik 038
                break;
            case DVMTypes::Opcode::OP_SGET_OBJECT_VOLATILE:
                instruction = std::make_shared<Instruction3rc>(dalvik_opcodes, input_file); // "invoke-custom/range" # Dalvik 038
                break;
            case DVMTypes::Opcode::OP_SPUT_OBJECT_VOLATILE:
                instruction = std::make_shared<Instruction21c>(dalvik_opcodes, input_file); // "const-method-handle" # Dalvik 039
                break;
            case DVMTypes::Opcode::OP_UNUSED_FF:
                instruction = std::make_shared<Instruction21c>(dalvik_opcodes, input_file); // 'const-method-type' # Dalvik 039
                break;
            default:
                std::string msg = "Invalud Instruction '"+std::to_string(opcode)+"'";
                throw exceptions::InvalidInstruction(msg);
            }

            return instruction;
        }
    }
}