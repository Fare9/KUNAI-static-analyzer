#include "dex_instructions.hpp"

namespace KUNAI {
    namespace DEX {

        /**
         * Instruction
         */
        Instruction::Instruction(std::shared_ptr<DalvikOpcodes> dalvik_opcodes, std::ifstream& input_file)
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
            std::cout << get_name() << " " << get_output();
        }

        std::string Instruction::get_output()
        {
            return "";
        }

        std::uint64_t Instruction::get_raw()
        {
            return this->OP;
        }

        std::map<DVMTypes::Operand, std::uint64_t> Instruction::get_operands()
        {
            std::map<DVMTypes::Operand, std::uint64_t> operands;

            return operands;
        }

        std::shared_ptr<DalvikOpcodes> Instruction::get_dalvik_opcodes()
        {
            return dalvik_opcodes;
        }
        /**
         * Instruction00x
         */

        Instruction00x::Instruction00x(std::shared_ptr<DalvikOpcodes> dalvik_opcodes, std::ifstream& input_file) :
            Instruction(dalvik_opcodes, input_file) 
        {
            this->set_length(0);
        }
        
        Instruction00x::~Instruction00x() {}

        /**
         * Instruction10x
         */
        Instruction10x::Instruction10x(std::shared_ptr<DalvikOpcodes> dalvik_opcodes, std::ifstream& input_file) :
            Instruction(dalvik_opcodes, input_file)
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
        Instruction12x::Instruction12x(std::shared_ptr<DalvikOpcodes> dalvik_opcodes, std::ifstream& input_file) :
            Instruction(dalvik_opcodes, input_file)
        {
            std::uint8_t instruction[2];
            this->set_length(2);

            if (!KUNAI::read_data_file<std::uint8_t[2]>(instruction, this->get_length(), input_file))
                throw exceptions::DisassemblerException("Error disassembling Instruction12x");
            
            this->set_OP(instruction[0]);
            this->vA = (instruction[1] & 0x0F);
            this->vB = (instruction[1] & 0xF0)  >> 4;
        }

        Instruction12x::~Instruction12x() {}

        std::string Instruction12x::get_output()
        {
            return "v" + std::to_string(vA) + ", v" + std::to_string(vB);
        }

        std::uint64_t Instruction12x::get_raw()
        {
            return (this->get_OP() | vA << 8 | vB << 12);
        }

        std::map<DVMTypes::Operand, std::uint64_t> Instruction12x::get_operands()
        {
            std::map<DVMTypes::Operand, std::uint64_t> operands = {
                {DVMTypes::Operand::REGISTER, vA},
                {DVMTypes::Operand::REGISTER, vB}
            };

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
         * Instruction22x
         */
        Instruction22x::Instruction22x(std::shared_ptr<DalvikOpcodes> dalvik_opcodes, std::ifstream& input_file) :
            Instruction(dalvik_opcodes, input_file)
        {
            std::uint8_t instruction[4];
            this->set_length(4);

            if (!KUNAI::read_data_file<std::uint8_t[4]>(instruction, this->get_length(), input_file))
                throw exceptions::DisassemblerException("Error disassembling Instruction22x");

            this->set_OP(instruction[0]);
            this->vAA = instruction[1];
            this->vBBBB = *(reinterpret_cast<std::uint16_t*>(&instruction[2]));
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

        std::map<DVMTypes::Operand, std::uint64_t> Instruction22x::get_operands()
        {
            std::map<DVMTypes::Operand, std::uint64_t> operands = {
                {DVMTypes::Operand::REGISTER, vAA},
                {DVMTypes::Operand::REGISTER, vBBBB}
            };

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
         * Instruction32x
         */
        Instruction32x::Instruction32x(std::shared_ptr<DalvikOpcodes> dalvik_opcodes, std::ifstream& input_file) :
            Instruction(dalvik_opcodes, input_file)
        {
            std::uint8_t instruction[6];

            this->set_length(6);

            if (!KUNAI::read_data_file<std::uint8_t[6]>(instruction, this->get_length(), input_file))
                throw exceptions::DisassemblerException("Error disassembling Instruction32x");

            if (instruction[1] != 0)
                throw exceptions::InvalidInstruction("Instruction32x OP code high byte should be 0");
            
            this->set_OP(instruction[0]);
            this->vAAAA = *(reinterpret_cast<std::uint16_t*>(&instruction[2]));
            this->vBBBB = *(reinterpret_cast<std::uint16_t*>(&instruction[4]));
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

        std::map<DVMTypes::Operand, std::uint64_t> Instruction32x::get_operands()
        {
            std::map<DVMTypes::Operand, std::uint64_t> operands = {
                {DVMTypes::Operand::REGISTER, vAAAA},
                {DVMTypes::Operand::REGISTER, vBBBB}
            };

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
         * Instruction11x
         */
        Instruction11x::Instruction11x(std::shared_ptr<DalvikOpcodes> dalvik_opcodes, std::ifstream& input_file) :
            Instruction(dalvik_opcodes, input_file)
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

        std::map<DVMTypes::Operand, std::uint64_t> Instruction11x::get_operands()
        {
            std::map<DVMTypes::Operand, std::uint64_t> operands = {
                {DVMTypes::Operand::REGISTER, vAA}
            };

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

        /**
         * Instruction11n
         */
        Instruction11n::Instruction11n(std::shared_ptr<DalvikOpcodes> dalvik_opcodes, std::ifstream& input_file) :
            Instruction(dalvik_opcodes, input_file)
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
            return (this->get_OP() | this->vA << 8 | this->nB << 12);
        }

        std::map<DVMTypes::Operand, std::uint64_t> Instruction11n::get_operands()
        {
            std::map<DVMTypes::Operand, std::uint64_t> operands = {
                {DVMTypes::Operand::REGISTER, vA},
                {DVMTypes::Operand::LITERAL, nB}
            };

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
         * Instruction21s
         */
        Instruction21s::Instruction21s(std::shared_ptr<DalvikOpcodes> dalvik_opcodes, std::ifstream& input_file) :
            Instruction(dalvik_opcodes, input_file)
        {
            std::uint8_t instruction[4];
            this->set_length(4);

            if (!KUNAI::read_data_file<std::uint8_t[4]>(instruction, this->get_length(), input_file))
                throw exceptions::DisassemblerException("Error disassembling Instruction21s");

            this->set_OP(instruction[0]);
            this->vA = instruction[1];
            this->nBBBB = *(reinterpret_cast<std::int16_t*>(&instruction[2]));
        }

        std::string Instruction21s::get_output()
        {
            return "v" + std::to_string(vA) + ", " + std::to_string(nBBBB);
        }

        std::uint64_t Instruction21s::get_raw()
        {
            return (get_OP() | vA << 8 | nBBBB << 16);
        }

        std::map<DVMTypes::Operand, std::uint64_t> Instruction21s::get_operands()
        {
            std::map<DVMTypes::Operand, std::uint64_t> operands = {
                {DVMTypes::Operand::REGISTER, vA},
                {DVMTypes::Operand::LITERAL, nBBBB}
            };

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
         * Instruction31i
         */
        Instruction31i::Instruction31i(std::shared_ptr<DalvikOpcodes> dalvik_opcodes, std::ifstream& input_file) :
            Instruction(dalvik_opcodes, input_file)
        {
            this->set_length(6);
            std::uint8_t instruction[6];

            if (!KUNAI::read_data_file<std::uint8_t[6]>(instruction, this->get_length(), input_file))
                throw exceptions::DisassemblerException("Error disassembling Instruction31i");

            this->set_OP(instruction[0]);
            this->vAA = instruction[1];
            this->nBBBBBBBB = *(reinterpret_cast<std::uint32_t*>(&instruction[2]));
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

        std::map<DVMTypes::Operand, std::uint64_t> Instruction31i::get_operands()
        {
            std::map<DVMTypes::Operand, std::uint64_t> operands = {
                {DVMTypes::Operand::REGISTER, vAA},
                {DVMTypes::Operand::LITERAL, nBBBBBBBB}
            };

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
         * Instruction21h
         */
        Instruction21h::Instruction21h(std::shared_ptr<DalvikOpcodes> dalvik_opcodes, std::ifstream& input_file) :
            Instruction(dalvik_opcodes, input_file)
        {
            std::uint8_t instruction[4];
    
            this->set_length(4);
            

            if (!KUNAI::read_data_file<std::uint8_t[4]>(instruction, this->get_length(), input_file))
                throw exceptions::DisassemblerException("Error disassembling Instruction21h");

            this->set_OP(instruction[0]);
            this->vAA = instruction[1];
            this->nBBBB_aux = *(reinterpret_cast<std::int16_t*>(&instruction[2]));

            switch(this->get_OP())
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

        std::string Instruction21h::get_output()
        {
            return "v" + std::to_string(vAA) + ", " + std::to_string(nBBBB);
        }

        std::uint64_t Instruction21h::get_raw()
        {
            return (get_OP() | vAA << 8 | nBBBB << 16);
        }

        std::map<DVMTypes::Operand, std::uint64_t> Instruction21h::get_operands()
        {
            std::map<DVMTypes::Operand, std::uint64_t> operands = {
                {DVMTypes::Operand::REGISTER, vAA},
                {DVMTypes::Operand::LITERAL, nBBBB}
            };

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
         * Instruction51l
         */
        Instruction51l::Instruction51l(std::shared_ptr<DalvikOpcodes> dalvik_opcodes, std::ifstream& input_file) : 
            Instruction(dalvik_opcodes, input_file)
        {
            std::uint8_t instruction[10];
            this->set_length(10);

            if (!KUNAI::read_data_file<std::uint8_t[10]>(instruction, this->get_length(), input_file))
                throw exceptions::DisassemblerException("Error disassembling Instruction51l");
            
            this->set_OP(instruction[0]);
            this->vAA = instruction[1];
            this->nBBBBBBBBBBBBBBBB = *(reinterpret_cast<std::uint64_t*>(&instruction[2]));
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

        std::map<DVMTypes::Operand, std::uint64_t> Instruction51l::get_operands()        
        {
            std::map<DVMTypes::Operand, std::uint64_t> operands = {
                {DVMTypes::Operand::REGISTER, vAA},
                {DVMTypes::Operand::LITERAL, nBBBBBBBBBBBBBBBB}
            };

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
         * Instruction21c
         */
        Instruction21c::Instruction21c(std::shared_ptr<DalvikOpcodes> dalvik_opcodes, std::ifstream& input_file) :
            Instruction(dalvik_opcodes, input_file)
        {
            std::uint8_t instruction[4];
            this->set_length(4);

            if (!KUNAI::read_data_file<std::uint8_t[4]>(instruction, this->get_length(), input_file))
                throw exceptions::DisassemblerException("Error disassembling Instruction21c");
            
            this->set_OP(instruction[0]);
            this->vAA = instruction[1];
            this->iBBBB = *(reinterpret_cast<std::uint16_t*>(&instruction[2]));
        }

        Instruction21c::~Instruction21c() {}

        std::string Instruction21c::get_output()
        {
            std::string str = "";

            switch(this->get_kind())
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

        std::map<DVMTypes::Operand, std::uint64_t> Instruction21c::get_operands()
        {
            std::map<DVMTypes::Operand, std::uint64_t> operands = {
                {DVMTypes::Operand::REGISTER, vAA},
                {DVMTypes::Operand::KIND, iBBBB}
            };

            return operands;
        }

        DVMTypes::Kind Instruction21c::get_source_kind()
        {
            return this->get_kind();
        }

        std::string* Instruction21c::get_source_str()
        {
            if (this->get_kind() == DVMTypes::Kind::STRING)
                return this->get_dalvik_opcodes()->get_dalvik_string_by_id(iBBBB);
            return nullptr;
        }
        Type* Instruction21c::get_source_typeid()
        {
            if (this->get_kind() == DVMTypes::Kind::TYPE)
                return this->get_dalvik_opcodes()->get_dalvik_Type_by_id(iBBBB);
            return nullptr;
        }
        FieldID* Instruction21c::get_source_static_field()
        {
            if (this->get_kind() == DVMTypes::Kind::FIELD)
                return this->get_dalvik_opcodes()->get_dalvik_field_by_id(iBBBB);
            return nullptr;
        }
        MethodID* Instruction21c::get_source_method()
        {
            if (this->get_kind() == DVMTypes::Kind::METH)
                return this->get_dalvik_opcodes()->get_dalvik_method_by_id(iBBBB);
            return nullptr;
        }
        ProtoID* Instruction21c::get_source_proto()
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
         * Instruction31c
         */
        Instruction31c::Instruction31c(std::shared_ptr<DalvikOpcodes> dalvik_opcodes, std::ifstream& input_file) :
            Instruction(dalvik_opcodes, input_file)
        {
            std::uint8_t instruction[6];
            this->set_length(6);

            if (!KUNAI::read_data_file<std::uint8_t[6]>(instruction, this->get_length(), input_file))
                throw exceptions::DisassemblerException("Error disassembling Instruction31c");
            
            this->set_OP(instruction[0]);
            this->vAA = instruction[1];
            this->iBBBBBBBB = *(reinterpret_cast<std::uint32_t*>(&instruction[2]));
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

        std::map<DVMTypes::Operand, std::uint64_t> Instruction31c::get_operands()
        {
            std::map<DVMTypes::Operand, std::uint64_t> operands = {
                {DVMTypes::Operand::REGISTER, vAA},
                {DVMTypes::Operand::KIND, iBBBBBBBB}
            };

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

        std::string* Instruction31c::get_source_str()
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
         * Instruction22c
         */
        Instruction22c::Instruction22c(std::shared_ptr<DalvikOpcodes> dalvik_opcodes, std::ifstream& input_file) :
            Instruction(dalvik_opcodes, input_file)
        {
            std::uint8_t instruction[4];
            this->set_length(4);

            if (!KUNAI::read_data_file<std::uint8_t[4]>(instruction, this->get_length(), input_file))
                throw exceptions::DisassemblerException("Error disassembling Instruction22c");

            this->set_OP(instruction[0]);
            this->vA = instruction[1] & 0xF;
            this->vB = (instruction[1] & 0xF0) >> 4;
            this->iCCCC = *(reinterpret_cast<std::uint16_t*>(&instruction[2]));
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

        std::map<DVMTypes::Operand, std::uint64_t> Instruction22c::get_operands()
        {
            std::map<DVMTypes::Operand, std::uint64_t> operands = {
                {DVMTypes::Operand::REGISTER, vA},
                {DVMTypes::Operand::REGISTER, vB},
                {DVMTypes::Operand::KIND, iCCCC}
            };

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

        Type* Instruction22c::get_third_operand_typeId()
        {
            if (get_kind() == DVMTypes::Kind::TYPE)
                return this->get_dalvik_opcodes()->get_dalvik_Type_by_id(iCCCC);
            return nullptr;
        }

        FieldID* Instruction22c::get_third_operand_FieldId()
        {
            if (get_kind() == DVMTypes::Kind::FIELD)
                return this->get_dalvik_opcodes()->get_dalvik_field_by_id(iCCCC);
            return nullptr;
        }
    }
}