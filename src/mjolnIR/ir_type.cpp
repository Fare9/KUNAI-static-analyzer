#include "KUNAI/mjolnIR/ir_grammar.hpp"

namespace KUNAI
{
    namespace MJOLNIR
    {
        /**
         * IRType class.
         */

        IRType::IRType(type_t type, op_type_t op_type, std::string type_name, size_t type_size) : IRExpr(IRExpr::TYPE_EXPR_T, op_type),
                                                                                                  type(type),
                                                                                                  type_name(type_name),
                                                                                                  type_size(type_size),
                                                                                                  annotations("")
        {
        }

        std::string IRType::to_string()
        {
            if (this->type == IRType::REGISTER_TYPE)
            {
                auto reg = reinterpret_cast<IRReg *>(this);
                return reg->to_string();
            }
            else if (this->type == IRType::TEMP_REGISTER_TYPE)
            {
                auto temp_reg = reinterpret_cast<IRTempReg *>(this);
                return temp_reg->to_string();
            }
            else if (this->type == IRType::CONST_INT_TYPE)
            {
                auto const_int = reinterpret_cast<IRConstInt *>(this);
                return const_int->to_string();
            }
            else if (this->type == IRType::MEM_TYPE)
            {
                auto mem = reinterpret_cast<IRMemory *>(this);
                return mem->to_string();
            }
            else if (this->type == IRType::STRING_TYPE)
            {
                auto str = reinterpret_cast<IRString *>(this);
                return str->to_string();
            }
            else if (this->type == IRType::CLASS_TYPE)
            {
                auto cls = reinterpret_cast<IRClass *>(this);
                return cls->to_string();
            }
            else if (this->type == IRType::CALLEE_TYPE)
            {
                auto callee = reinterpret_cast<IRCallee *>(this);
                return callee->to_string();
            }
            else if (this->type == IRType::FIELD_TYPE)
            {
                auto field = reinterpret_cast<IRField *>(this);
                return field->to_string();
            }
            else if (this->type == IRType::NONE_TYPE)
            {
                return "IRType [NONE]";
            }

            return "";
        }

        bool IRType::equal(irtype_t type)
        {
            return *this == *(type.get());
        }

        bool operator==(IRType &type1, IRType &type2)
        {
            if (type1.type != type2.type)
                return false;

            if (type1.type == IRType::REGISTER_TYPE)
            {
                IRReg &reg1 = reinterpret_cast<IRReg &>(type1);
                IRReg &reg2 = reinterpret_cast<IRReg &>(type2);

                return reg1 == reg2;
            }
            else if (type1.type == IRType::TEMP_REGISTER_TYPE)
            {
                IRTempReg &treg1 = reinterpret_cast<IRTempReg &>(type1);
                IRTempReg &treg2 = reinterpret_cast<IRTempReg &>(type2);

                return treg1 == treg2;
            }
            else if (type1.type == IRType::CONST_INT_TYPE)
            {
                IRConstInt &int1 = reinterpret_cast<IRConstInt &>(type1);
                IRConstInt &int2 = reinterpret_cast<IRConstInt &>(type2);

                return int1 == int2;
            }
            else if (type1.type == IRType::MEM_TYPE)
            {
                IRMemory &mem1 = reinterpret_cast<IRMemory &>(type1);
                IRMemory &mem2 = reinterpret_cast<IRMemory &>(type2);

                return mem1 == mem2;
            }
            else if (type1.type == IRType::STRING_TYPE)
            {
                IRString &str1 = reinterpret_cast<IRString &>(type1);
                IRString &str2 = reinterpret_cast<IRString &>(type2);

                return str1 == str2;
            }
            else if (type1.type == IRType::CLASS_TYPE)
            {
                IRClass &class1 = reinterpret_cast<IRClass &>(type1);
                IRClass &class2 = reinterpret_cast<IRClass &>(type2);

                return class1 == class2;
            }
            else if (type1.type == IRType::CALLEE_TYPE)
            {
                IRCallee &callee1 = reinterpret_cast<IRCallee &>(type1);
                IRCallee &callee2 = reinterpret_cast<IRCallee &>(type2);

                return callee1 == callee2;
            }
            else if (type1.type == IRType::FIELD_TYPE)
            {
                IRField &field1 = reinterpret_cast<IRField &>(type1);
                IRField &field2 = reinterpret_cast<IRField &>(type2);

                return field1 == field2;
            }

            return false;
        }

        /**
         * IRReg class
         */

        IRReg::IRReg(std::uint32_t reg_id, int current_arch, std::string type_name, size_t type_size)
            : IRType(REGISTER_TYPE, REGISTER_OP_T, type_name, type_size),
              id(reg_id),
              sub_id(-1),
              current_arch(current_arch)
        {
        }

        IRReg::IRReg(std::uint32_t reg_id, std::int32_t reg_sub_id, int current_arch, std::string type_name, size_t type_size)
            : IRType(REGISTER_TYPE, REGISTER_OP_T, type_name, type_size),
              id(reg_id),
              sub_id(reg_sub_id),
              current_arch(current_arch)
        {
        }

        std::string IRReg::to_string()
        {
            std::stringstream stream;

            stream << "IRReg: ";

            switch (current_arch)
            {
            case MJOLNIR::x86_arch:
                stream << "[x86: " << x86_regs_name.at(static_cast<x86_regs_t>(id));
                if (sub_id != -1)
                    stream << "." << sub_id;
                stream << "]";
                break;
            case MJOLNIR::dalvik_arch:
                stream << "[dalvik: v" << id;
                if (sub_id != -1)
                    stream << "." << sub_id;
                stream << "]";
                break;
            default:
                stream << "[None: reg" << id;
                if (sub_id != -1)
                    stream << "." << sub_id;
                stream << "]";
                break;
            }

            return stream.str();
        }

        bool IRReg::same(irreg_t reg)
        {
            if (id == reg->get_id() && current_arch == reg->get_current_arch())
                return true;
            return false;
        }

        bool IRReg::equal(irreg_t reg)
        {
            return *(this) == *(reg.get());
        }

        bool operator==(IRReg &type1, IRReg &type2)
        {
            if (type1.id == type2.id && type1.sub_id == type2.sub_id && type1.current_arch == type2.current_arch)
                return true;
            return false;
        }

        /**
         * IRTempReg class
         */

        IRTempReg::IRTempReg(std::uint32_t reg_id, std::string type_name, size_t type_size)
            : IRType(TEMP_REGISTER_TYPE, TEMP_REGISTER_OP_T, type_name, type_size),
              id(reg_id)
        {
        }

        std::string IRTempReg::to_string()
        {
            std::stringstream stream;

            stream << "IRTempReg [" << id << "]";

            return stream.str();
        }

        bool IRTempReg::equal(irtempreg_t temp_reg)
        {
            return *(this) == *(temp_reg.get());
        }

        bool operator==(IRTempReg &type1, IRTempReg &type2)
        {
            if (type1.id == type2.id)
                return true;
            return false;
        }

        /**
         * IRConstInt class.
         */

        IRConstInt::IRConstInt(std::uint64_t value, bool is_signed, mem_access_t byte_order, std::string type_name, size_t type_size)
            : IRType(CONST_INT_TYPE, CONST_INT_OP_T, type_name, type_size),
              value(value),
              is_signed(is_signed),
              byte_order(byte_order)
        {
        }

        std::string IRConstInt::to_string()
        {
            std::stringstream stream;

            if (is_signed)
                stream << "IRConstInt [value: " << static_cast<std::int64_t>(value) << "]";
            else
                stream << "IRConstInt [value: " << value << "]";

            switch (byte_order)
            {
            case IRType::LE_ACCESS:
                stream << "[Little-Endian]";
                break;
            case IRType::BE_ACCESS:
                stream << "[Big-Endian]";
                break;
            case IRType::ME_ACCESS:
                stream << "[Middle-Endian]";
                break;
            default:
                break;
            }

            return stream.str();
        }

        bool IRConstInt::equal(irconstint_t const_int)
        {
            return *(this) == *(const_int.get());
        }

        bool operator==(IRConstInt &type1, IRConstInt &type2)
        {
            if (type1.value == type2.value && type1.is_signed == type2.is_signed)
                return true;
            return false;
        }

        IRConstInt operator+(IRConstInt &a, IRConstInt &b)
        {
            if (a.is_signed)
            {
                int64_t result = static_cast<int64_t>(a.value) + static_cast<int64_t>(b.value);
                IRConstInt res(result, a.is_signed, a.byte_order, a.get_type_name(), a.get_type_size());

                return res;
            }
            uint64_t result = a.value + a.value;

            IRConstInt res(result, a.is_signed, a.byte_order, a.get_type_name(), a.get_type_size());
            return res;
        }

        IRConstInt operator-(IRConstInt &a, IRConstInt &b)
        {
            if (a.is_signed)
            {
                int64_t result = static_cast<int64_t>(a.value) - static_cast<int64_t>(b.value);
                IRConstInt res(result, a.is_signed, a.byte_order, a.get_type_name(), a.get_type_size());

                return res;
            }
            uint64_t result = a.value - b.value;

            IRConstInt res(result, a.is_signed, a.byte_order, a.get_type_name(), a.get_type_size());
            return res;
        }

        IRConstInt operator/(IRConstInt &a, IRConstInt &b)
        {
            if (a.is_signed)
            {
                int64_t result = static_cast<int64_t>(a.value) / static_cast<int64_t>(b.value);
                IRConstInt res(result, a.is_signed, a.byte_order, a.get_type_name(), a.get_type_size());

                return res;
            }
            uint64_t result = a.value / b.value;

            IRConstInt res(result, a.is_signed, a.byte_order, a.get_type_name(), a.get_type_size());
            return res;
        }

        IRConstInt operator*(IRConstInt &a, IRConstInt &b)
        {
            if (a.is_signed)
            {
                int64_t result = static_cast<int64_t>(a.value) * static_cast<int64_t>(b.value);
                IRConstInt res(result, a.is_signed, a.byte_order, a.get_type_name(), a.get_type_size());

                return res;
            }
            uint64_t result = a.value * b.value;

            IRConstInt res(result, a.is_signed, a.byte_order, a.get_type_name(), a.get_type_size());
            return res;
        }

        IRConstInt operator%(IRConstInt &a, IRConstInt &b)
        {
            if (a.is_signed)
            {
                int64_t result = static_cast<int64_t>(a.value) % static_cast<int64_t>(b.value);
                IRConstInt res(result, a.is_signed, a.byte_order, a.get_type_name(), a.get_type_size());

                return res;
            }
            uint64_t result = a.value % b.value;

            IRConstInt res(result, a.is_signed, a.byte_order, a.get_type_name(), a.get_type_size());
            return res;
        }

        IRConstInt operator&(IRConstInt &a, IRConstInt &b)
        {
            uint64_t result = a.value & b.value;

            IRConstInt res(result, a.is_signed, a.byte_order, a.get_type_name(), a.get_type_size());
            return res;
        }

        IRConstInt operator|(IRConstInt &a, IRConstInt &b)
        {
            uint64_t result = a.value | b.value;

            IRConstInt res(result, a.is_signed, a.byte_order, a.get_type_name(), a.get_type_size());
            return res;
        }

        IRConstInt operator^(IRConstInt &a, IRConstInt &b)
        {
            uint64_t result = a.value ^ b.value;

            IRConstInt res(result, a.is_signed, a.byte_order, a.get_type_name(), a.get_type_size());
            return res;
        }

        IRConstInt operator<<(IRConstInt &a, IRConstInt &b)
        {
            uint64_t result = a.value << b.value;

            IRConstInt res(result, a.is_signed, a.byte_order, a.get_type_name(), a.get_type_size());
            return res;
        }

        IRConstInt operator>>(IRConstInt &a, IRConstInt &b)
        {
            uint64_t result = a.value >> b.value;

            IRConstInt res(result, a.is_signed, a.byte_order, a.get_type_name(), a.get_type_size());
            return res;
        }

        IRConstInt operator++(IRConstInt &a, int)
        {
            uint64_t result = a.value++;

            IRConstInt res(result, a.is_signed, a.byte_order, a.get_type_name(), a.get_type_size());
            return res;
        }

        IRConstInt operator--(IRConstInt &a, int)
        {
            uint64_t result = a.value--;

            IRConstInt res(result, a.is_signed, a.byte_order, a.get_type_name(), a.get_type_size());
            return res;
        }

        IRConstInt operator!(IRConstInt &a)
        {
            uint64_t result = !a.value;

            IRConstInt res(result, a.is_signed, a.byte_order, a.get_type_name(), a.get_type_size());
            return res;
        }

        IRConstInt operator~(IRConstInt &a)
        {
            uint64_t result = ~a.value;

            IRConstInt res(result, a.is_signed, a.byte_order, a.get_type_name(), a.get_type_size());
            return res;
        }

        /**
         * IRMemory class
         */

        IRMemory::IRMemory(std::uint64_t mem_address, std::int32_t offset, mem_access_t byte_order, std::string type_name, size_t type_size)
            : IRType(MEM_TYPE, MEM_OP_T, type_name, type_size),
              mem_address(mem_address),
              offset(offset),
              byte_order(byte_order)
        {
        }

        std::string IRMemory::to_string()
        {
            std::stringstream stream;

            stream << "IRMemory: ";

            if (mem_address != 0)
                stream << "[address: 0x" << std::hex << mem_address << "]";

            if (offset != 0)
                stream << "[offset: 0x" << std::hex << offset << "]";

            switch (byte_order)
            {
            case IRType::LE_ACCESS:
                stream << "[Little-Endian]";
                break;
            case IRType::BE_ACCESS:
                stream << "[Big-Endian]";
                break;
            case IRType::ME_ACCESS:
                stream << "[Middle-Endian]";
                break;
            default:
                break;
            }

            return stream.str();
        }

        bool IRMemory::equal(irmemory_t memory)
        {
            return *this == *(memory.get());
        }

        bool operator==(IRMemory &type1, IRMemory &type2)
        {
            if ((type1.mem_address == type2.mem_address) && (type1.offset == type2.offset))
                return true;
            return false;
        }

        /**
         * IRString class
         */

        IRString::IRString(std::string str_value, std::string type_name, size_t type_size)
            : IRType(STRING_TYPE, STRING_OP_T, type_name, type_size),
              str_value(str_value)
        {
        }

        std::string IRString::to_string()
        {
            std::stringstream stream;

            stream << "IRString [" << str_value << "]";

            return stream.str();
        }

        bool IRString::equal(irstring_t str)
        {
            return *this == *(str.get());
        }

        bool operator==(IRString &type1, IRString &type2)
        {
            if (!type1.str_value.compare(type2.str_value))
                return true;
            return false;
        }

        /**
         * IRClass
         */

        IRClass::IRClass(std::string class_name, std::string type_name, size_t type_size)
            : IRType(CLASS_TYPE, CLASS_OP_T, type_name, type_size),
              class_name(class_name)
        {
        }

        std::string IRClass::to_string()
        {
            std::stringstream stream;

            stream << "IRClass [" << class_name << "]";

            return stream.str();
        }

        bool IRClass::equal(irclass_t class_)
        {
            return *(this) == *(class_.get());
        }

        bool operator==(IRClass &type1, IRClass &type2)
        {
            if (!type1.class_name.compare(type2.class_name))
                return true;
            return false;
        }

        /**
         * IRCallee class
         */

        IRCallee::IRCallee(std::uint64_t addr,
                           std::string name,
                           std::string class_name,
                           int n_of_params,
                           std::string description,
                           std::string type_name,
                           size_t type_size)
            : IRType(CALLEE_TYPE, CALLEE_OP_T, type_name, type_size),
              addr(addr),
              name(name),
              class_name(class_name),
              n_of_params(n_of_params),
              description(description)
        {
        }

        std::string IRCallee::to_string()
        {
            std::stringstream str_stream;

            str_stream << "IRCallee ";

            if (addr != 0)
                str_stream << "[addr: 0x" << std::hex << addr << "]";

            str_stream << "[method: ";

            if (!class_name.empty())
                str_stream << class_name << "->";

            if (!name.empty())
                str_stream << name;

            if (!description.empty())
                str_stream << description;

            str_stream << "]";

            return str_stream.str();
        }

        bool IRCallee::equal(ircallee_t callee)
        {
            return *this == *(callee.get());
        }

        bool operator==(IRCallee &type1, IRCallee &type2)
        {
            // check first for address
            if (type1.addr != 0 && (type1.addr == type2.addr))
                return true;
            // check for whole method
            std::string callee1 = type1.class_name + type1.name + type1.description;
            std::string callee2 = type2.class_name + type2.name + type2.description;
            if (!(callee1.compare(callee2)))
                return true;
            return false;
        }

        /**
         * IRField class
         */

        IRField::IRField(std::string class_name,
                         field_t type,
                         std::string field_name,
                         std::string type_name,
                         size_t type_size) : IRType(FIELD_TYPE, FIELD_OP_T, type_name, type_size),
                                             class_name(class_name),
                                             type(type),
                                             type_class(type_name),
                                             field_name(field_name)
        {
        }

        IRField::IRField(std::string class_name,
                         std::string type_class_name,
                         std::string field_name,
                         std::string type_name,
                         size_t type_size)
            : IRType(FIELD_TYPE, FIELD_OP_T, type_name, type_size),
              class_name(class_name),
              type(CLASS_F),
              type_class(type_class_name),
              field_name(field_name)
        {
        }

        std::string IRField::to_string()
        {
            std::stringstream str_stream;

            str_stream << "IRField ";

            str_stream << "[Class: " << class_name << "]";

            switch (type)
            {
            case CLASS_F:
                str_stream << "[Type: CLASS<" << type_class << ">]";
                break;
            case BOOLEAN_F:
                str_stream << "[Type: BOOLEAN]";
                break;
            case BYTE_F:
                str_stream << "[Type: BYTE]";
                break;
            case CHAR_F:
                str_stream << "[Type: CHAR]";
                break;
            case DOUBLE_F:
                str_stream << "[Type: DOUBLE]";
                break;
            case FLOAT_F:
                str_stream << "[Type: FLOAT]";
                break;
            case INT_F:
                str_stream << "[Type: INT]";
                break;
            case LONG_F:
                str_stream << "[Type: LONG]";
                break;
            case SHORT_F:
                str_stream << "[Type: SHORT]";
                break;
            case VOID_F:
                str_stream << "[Type: VOID]";
                break;
            default:
                break;
            }

            str_stream << "[Name: " << field_name << "]";

            return str_stream.str();
        }

        bool IRField::equal(irfield_t field)
        {
            return *this == *(field.get());
        }

        bool operator==(IRField &field1, IRField &field2)
        {
            return (!field1.class_name.compare(field2.class_name) &&
                    field1.type == field2.type &&
                    !field1.type_class.compare(field2.type_class) &&
                    !field1.field_name.compare(field2.field_name));
        }
    }
}