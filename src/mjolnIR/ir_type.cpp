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
         * @param type_name: name used for representing the type while printing.
         * @param type_size: size of the type in bytes.
         * @return void
         */
        IRType::IRType(std::string type_name, size_t type_size) :
            IRExpr(IRExpr::NONE_EXPR_T, nullptr, nullptr)
        {
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
         * @brief virtual method from IRType this must be implemented by other types too. Returns one of the type_t enum values.
         * @return type_t
         */
        IRType::type_t IRType::get_type()
        {
            return NONE_TYPE;
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
            : IRType(type_name, type_size)
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
         * @brief Return the register type for this class.
         * @return type_t
         */
        IRReg::type_t IRReg::get_type()
        {
            return REGISTER_TYPE;
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
            : IRType(type_name, type_size)
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
         * @brief Return the temporal register type for this class.
         * @return type_t
         */
        IRTempReg::type_t IRTempReg::get_type()
        {
            return TEMP_REGISTER_TYPE;
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
         * IRConstInt class.
         */

        /**
         * @brief Constructor of IRConstInt this represent any integer used in the code.
         * @param is_signed: is signed value (true) or unsigned (false).
         * @param byte_order: byte order of the value.
         * @param type_name: name used for representing the value.
         * @param type_size: size of the integer.
         * @return void
         */
        IRConstInt::IRConstInt(bool is_signed, mem_access_t byte_order, std::string type_name, size_t type_size)
            : IRType(type_name, type_size)
        {
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
         * @brief return in this case CONST_INT.
         * @return type_t
         */
        IRConstInt::type_t IRConstInt::get_type()
        {
            return CONST_INT_TYPE;
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
            : IRType(type_name, type_size)
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
         * @brief Get the MEM_TYPE.
         * @return type_t
         */
        IRMemory::type_t IRMemory::get_type()
        {
            return MEM_TYPE;
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

    }
}