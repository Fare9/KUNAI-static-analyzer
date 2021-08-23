#include "ir_type.hpp"

namespace KUNAI
{
    namespace MJOLNIR
    {

        /**
         * IRType class.
         */

        /**
         * @brief Constructor of the IRType, this will be the generic type used for the others.
         * @param type: type of the class.
         * @param type_name: name used for representing the type while printing.
         * @param type_size: size of the type in bytes.
         * @return void
         */
        IRType::IRType(type_t type, std::string type_name, size_t type_size) :
            IRExpr(IRExpr::NONE_EXPR_T, nullptr, nullptr)
        {
            this->type = type;
            this->type_name = type_name;
            this->type_size = type_size;
            this->annotations = "";
        }

        /**
         * @brief Destructor of the IRType.
         * @return void
         */
        IRType::~IRType() {}

        /**
         * @brief retrieve the type name.
         * @return std::string
         */
        std::string IRType::get_type_name()
        {
            return type_name;
        }

        /**
         * @brief retrieve the type size.
         * @return size_t
         */
        size_t IRType::get_type_size()
        {
            return type_size;
        }

        /**
         * @brief method from IRType this return one of the types given by class.
         * @return type_t
         */
        IRType::type_t IRType::get_type()
        {
            return type;
        }

        /**
         * @brief virtual method from IRType this must be implemented by other types too. Returns one of the mem_access_t enum values.
         * @return mem_access_t
         */
        IRType::mem_access_t IRType::get_access()
        {
            return NONE_ACCESS;
        }

        /**
         * @brief Write an annotations to the type object.
         * @return void
         */
        void IRType::write_annotations(std::string annotations)
        {
            this->annotations = annotations;
        }

        /**
         * @brief Read the annotations from the type object.
         * @return std::string
         */
        std::string IRType::read_annotations()
        {
            return annotations;
        }

        /**
         * @brief == operator for IRType, we have specific check to do.
         * @param type1: first type for comparison.
         * @param type2: second type for comparison
         * @return bool
         */
        bool operator==(IRType& type1, IRType& type2)
        {
            if (type1.type != type2.type)
                return false;
            
            if (type1.type == IRType::REGISTER_TYPE)
            {
                IRReg& reg1 = dynamic_cast<IRReg&>(type1);
                IRReg& reg2 = dynamic_cast<IRReg&>(type2);

                return reg1 == reg2;
            }
            else if (type1.type == IRType::TEMP_REGISTER_TYPE)
            {
                IRTempReg& treg1 = dynamic_cast<IRTempReg&>(type1);
                IRTempReg& treg2 = dynamic_cast<IRTempReg&>(type2);

                return treg1 == treg2;
            }
            else if (type1.type == IRType::CONST_INT_TYPE)
            {
                IRConstInt& int1 = dynamic_cast<IRConstInt&>(type1);
                IRConstInt& int2 = dynamic_cast<IRConstInt&>(type2);

                return int1 == int2;
            }
            else if (type1.type == IRType::MEM_TYPE)
            {
                IRMemory& mem1 = dynamic_cast<IRMemory&>(type1);
                IRMemory& mem2 = dynamic_cast<IRMemory&>(type2);

                return mem1 == mem2;
            }
            else if (type1.type == IRType::STRING_TYPE)
            {
                IRString& str1 = dynamic_cast<IRString&>(type1);
                IRString& str2 = dynamic_cast<IRString&>(type2);

                return str1 == str2;
            }
            else if (type1.type == IRType::CALLEE_TYPE)
            {
                IRCallee& callee1 = dynamic_cast<IRCallee&>(type1);
                IRCallee& callee2 = dynamic_cast<IRCallee&>(type2);

                return callee1 == callee2;
            }


            return false;
        }

        /**
         * IRReg class
         */

        /**
         * @brief Constructor of IRReg type.
         * @param reg_id: id of the register this can be an enum if is a well known register, or just an id.
         * @param type_name: string for representing the register.
         * @param type_size: size of the register.
         * @return void
         */
        IRReg::IRReg(std::uint32_t reg_id, std::string type_name, size_t type_size)
            : IRType(REGISTER_TYPE, type_name, type_size)
        {
            this->id = reg_id;
        }

        /**
         * @brief Destructor of IRReg.
         * @return void
         */
        IRReg::~IRReg() {}

        /**
         * @brief Return the ID from the register.
         * @return std::uint32_t
         */
        std::uint32_t IRReg::get_id()
        {
            return id;
        }

        /**
         * @brief Return the register type as str.
         * @return std::string
         */
        std::string IRReg::get_type_str()
        {
            return "Register";
        }

        /**
         * @brief Return the access type to the register. We get as all the register are accessed as Little-Endian.
         * @return mem_access_t
         */
        IRReg::mem_access_t IRReg::get_access()
        {
            return NONE_ACCESS;
        }

        /**
         * @brief == operator for IRRegs, we have specific check to do.
         * @param type1: first type for comparison.
         * @param type2: second type for comparison
         * @return bool
         */
        bool operator==(IRReg& type1, IRReg& type2)
        {
            if (type1.id == type2.id)
                return true;
            return false;
        }

        /**
         * IRTempReg class
         */

        /**
         * @brief Constructor of IRTempReg type.
         * @param reg_id: id of the register this will be an incremental id.
         * @param type_name: string for representing the register.
         * @param type_size: size of the register.
         * @return void
         */
        IRTempReg::IRTempReg(std::uint32_t reg_id, std::string type_name, size_t type_size)
            : IRType(TEMP_REGISTER_TYPE, type_name, type_size)
        {
            this->id = reg_id;
        }

        /**
         * @brief Destructor of IRTempReg.
         * @return void
         */
        IRTempReg::~IRTempReg() {}

        /**
         * @brief Return the ID from the register.
         * @return std::uint32_t
         */
        std::uint32_t IRTempReg::get_id()
        {
            return id;
        }

        /**
         * @brief Return the temporal register type as str.
         * @return std::string
         */
        std::string IRTempReg::get_type_str()
        {
            return "Temporal Register";
        }

        /**
         * @brief Return the access type to the register. We get as all the register are accessed as Little-Endian.
         * @return mem_access_t
         */
        IRTempReg::mem_access_t IRTempReg::get_access()
        {
            return NONE_ACCESS;
        }

        /**
         * @brief == operator for IRTempReg, we have specific check to do.
         * @param type1: first type for comparison.
         * @param type2: second type for comparison
         * @return bool
         */
        bool operator==(IRTempReg& type1, IRTempReg& type2)
        {
            if (type1.id == type2.id)
                return true;
            return false;
        }

        /**
         * IRConstInt class.
         */

        /**
         * @brief Constructor of IRConstInt this represent any integer used in the code.
         * @param value: value of the constant integer
         * @param is_signed: is signed value (true) or unsigned (false).
         * @param byte_order: byte order of the value.
         * @param type_name: name used for representing the value.
         * @param type_size: size of the integer.
         * @return void
         */
        IRConstInt::IRConstInt(std::uint64_t value, bool is_signed, mem_access_t byte_order, std::string type_name, size_t type_size)
            : IRType(CONST_INT_TYPE, type_name, type_size)
        {
            this->value = value;
            this->is_signed = is_signed;
            this->byte_order = byte_order;
        }

        /**
         * @brief Destructor of IRConstInt.
         * @return void
         */
        IRConstInt::~IRConstInt() {}

        /**
         * @brief Get if the int value is signed  or not.
         * @return bool
         */
        bool IRConstInt::get_is_signed()
        {
            return is_signed;
        }

        /**
         * @brief Return the CONST_INT type as string.
         * @return std::string
         */
        std::string IRConstInt::get_type_str()
        {
            return "ConstInt";
        }

        /**
         * @brief Return the type of access of the constant int.
         * @return mem_access_t
         */
        IRConstInt::mem_access_t IRConstInt::get_access()
        {
            return byte_order;
        }

        /**
         * @brief == operator for IRConstInt, we have specific check to do.
         * @param type1: first type for comparison.
         * @param type2: second type for comparison
         * @return bool
         */
        bool operator==(IRConstInt& type1, IRConstInt& type2)
        {
            if (type1.value == type2.value)
                return true;
            return false;
        }

        /**
         * IRMemory class
         */
        
        /**
         * @brief IRMemory constructor this represent a memory address with accessed offset and size.
         * @param mem_address: address of the memory.
         * @param offset: offset accessed (commonly 0).
         * @param byte_order: byte order of the memory (LE, BE, ME?).
         * @param type_name: memory representation with a string.
         * @param type_size: size of the memory.
         * @return void
         */
        IRMemory::IRMemory(std::uint64_t mem_address, std::int32_t offset, mem_access_t byte_order, std::string type_name, size_t type_size)
            : IRType(MEM_TYPE, type_name, type_size)
        {
            this->mem_address = mem_address;
            this->offset = offset;
            this->byte_order = byte_order;
        }
        
        /**
         * @brief IRMemory destructor.
         * @return void
         */
        IRMemory::~IRMemory(){}

        /**
         * @brief Get the memory address of the type.
         * @return std::uint64_t
         */
        std::uint64_t IRMemory::get_mem_address()
        {
            return mem_address;
        }

        /**
         * @brief Get the accessed offset of the type.
         * @return std::int32_t
         */
        std::int32_t IRMemory::get_offset()
        {
            return offset;
        }

        /**
         * @brief Get the MEM_TYPE as string.
         * @return std::string
         */
        std::string IRMemory::get_type_str()
        {
            return "Memory";
        }

        /**
         * @brief Get the mem access type from the memory.
         * @return mem_access_t
         */
        IRMemory::mem_access_t IRMemory::get_access()
        {
            return byte_order;
        }

        /**
         * @brief == operator for IRMemory, we have specific check to do.
         * @param type1: first type for comparison.
         * @param type2: second type for comparison
         * @return bool
         */
        bool operator==(IRMemory& type1, IRMemory& type2)
        {
            if ((type1.mem_address == type2.mem_address) && (type1.offset == type2.offset))
                return true;
            return false;
        }

        /**
         * IRString class
         */
        /**
         * @brief Constructor of IRString class, this represent strings used in code.
         * @param str_value: value of that string.
         * @param type_name: some meaninful string name.
         * @param type_size: size of the type (probably here string length)
         * @return void
         */
        IRString::IRString(std::string str_value, std::string type_name, size_t type_size)
            : IRType(STRING_TYPE, type_name, type_size)
        {
            this->str_value = str_value;
        }

        /**
         * @brief Destructor of IRString, nothing to be done.
         * @return void
         */
        IRString::~IRString() {}

        /**
         * @brief Return the value of the string.
         * @return std::string
         */
        std::string IRString::get_str_value()
        {
            return str_value;
        }

        /**
         * @brief Get the type as a string.
         * @return std::string
         */
        std::string IRString::get_type_str()
        {
            return "String";
        }

        /**
         * @brief Get the type of access of string, in this one NONE.
         * @return mem_access_t
         */
        IRString::mem_access_t IRString::get_access()
        {
            return NONE_ACCESS;
        }

        /**
         * @brief == operator for IRString, we have specific check to do.
         * @param type1: first type for comparison.
         * @param type2: second type for comparison
         * @return bool
         */
        bool operator==(IRString& type1, IRString& type2)
        {
            if (type1.str_value == type2.str_value)
                return true;
            return false;
        }

        /**
         * IRCallee class
         */

        /**
         * @brief Constructor of IRCallee this represent any function/method called by a caller!
         * @param addr: address of the function/method called (if available).
         * @param name: name of the function/method called (if available).
         * @param class_name: name of the class from the method called (if available).
         * @param n_of_params: number of the parameters for the function/method (if available).
         * @param description: description of the parameters from the function/method (if available).
         * @param type_name: some meaninful string name.
         * @param type_size: size of the type (probably here 0)
         * @return void
         */
        IRCallee::IRCallee(std::uint64_t addr,
                     std::string name,
                     std::string class_name,
                     int n_of_params,
                     std::string description,
                     std::string type_name, 
                     size_t type_size)
            : IRType(CALLEE_TYPE, type_name, type_size)
        {
            this->addr = addr;
            this->name = name;
            this->class_name = class_name;
            this->n_of_params = n_of_params;
            this->description = description;
        }

        /**
         * @brief Destructor of IRCallee, nothing to be done.
         * @return void
         */
        IRCallee::~IRCallee() {}

        /**
         * @brief Get address of function/method
         * @return std::uint64_t
         */
        std::uint64_t IRCallee::get_addr()
        {
            return addr;
        }

        /**
         * @brief Get name of the called function/method.
         * @return std::string
         */
        std::string IRCallee::get_name()
        {
            return name;
        }

        /**
         * @brief Get the class name of the called method.
         * @return std::string
         */
        std::string IRCallee::get_class_name()
        {
            return class_name;
        }

        /**
         * @brief Get the number of parameters from the called function/method.
         * @return int
         */
        int IRCallee::get_number_of_params()
        {
            return n_of_params;
        }

        /**
         * @brief Get the description of the method if exists.
         * @return std::string
         */
        std::string IRCallee::get_description()
        {
            return description;
        }

        /**
         * @brief Get the type of IRCallee as string
         * @return std::string
         */
        std::string IRCallee::get_type_str()
        {
            return "Callee";
        }

        /**
         * @brief Get memory access NONE in this case.
         * @return mem_access_t
         */
        IRCallee::mem_access_t IRCallee::get_access()
        {
            return NONE_ACCESS;
        }

        /**
         * @brief == operator for IRCallee, we have specific check to do.
         * @param type1: first type for comparison.
         * @param type2: second type for comparison
         * @return bool
         */
        bool operator==(IRCallee& type1, IRCallee& type2)
        {
            // check first for address
            if (type1.addr != 0 && (type1.addr == type2.addr))
                return true;
            // check for whole method
            std::string callee1 = type1.class_name + type1.name + type1.description;
            std::string callee2 = type2.class_name + type2.name + type2.description;
            if (callee1 != "" && (callee1 == callee2))
                return true;
            return false;
        }
    }
}