#include "ir_grammar.hpp"

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
            IRExpr(IRExpr::TYPE_EXPR_T, nullptr, nullptr)
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
         * @brief Return the proper to_string.
         * 
         * @return std::string 
         */
        std::string IRType::to_string()
        {
            if (this->type == IRType::REGISTER_TYPE)
            {
                auto reg = reinterpret_cast<IRReg*>(this);
                return reg->to_string();
            }
            else if (this->type == IRType::TEMP_REGISTER_TYPE)
            {
                auto temp_reg = reinterpret_cast<IRTempReg*>(this);
                return temp_reg->to_string();
            }
            else if (this->type == IRType::CONST_INT_TYPE)
            {
                auto const_int = reinterpret_cast<IRConstInt*>(this);
                return const_int->to_string();
            }
            else if (this->type == IRType::MEM_TYPE)
            {
                auto mem = reinterpret_cast<IRMemory*>(this);
                return mem->to_string();
            }
            else if (this->type == IRType::STRING_TYPE)
            {
                auto str = reinterpret_cast<IRString*>(this);
                return str->to_string();
            }
            else if (this->type == IRType::CLASS_TYPE)
            { 
                auto cls = reinterpret_cast<IRClass*>(this);
                return cls->to_string();
            }
            else if (this->type == IRType::CALLEE_TYPE)
            {
                auto callee = reinterpret_cast<IRCallee*>(this);
                return callee->to_string();
            }
            else if (this->type == IRType::FIELD_TYPE)
            {
                auto field = reinterpret_cast<IRField*>(this);
                return field->to_string();
            }
            else if (this->type == IRType::NONE_TYPE)
            {
                return "IRType [NONE]";
            }

            return "";
        }

        /**
         * @brief Comparison of two IRType with shared_ptr.
         * @return bool
         */
        bool IRType::equal(std::shared_ptr<IRType> type)
        {
            return *this == *(type.get());
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
                IRReg& reg1 = reinterpret_cast<IRReg&>(type1);
                IRReg& reg2 = reinterpret_cast<IRReg&>(type2);

                return reg1 == reg2;
            }
            else if (type1.type == IRType::TEMP_REGISTER_TYPE)
            {
                IRTempReg& treg1 = reinterpret_cast<IRTempReg&>(type1);
                IRTempReg& treg2 = reinterpret_cast<IRTempReg&>(type2);

                return treg1 == treg2;
            }
            else if (type1.type == IRType::CONST_INT_TYPE)
            {
                IRConstInt& int1 = reinterpret_cast<IRConstInt&>(type1);
                IRConstInt& int2 = reinterpret_cast<IRConstInt&>(type2);

                return int1 == int2;
            }
            else if (type1.type == IRType::MEM_TYPE)
            {
                IRMemory& mem1 = reinterpret_cast<IRMemory&>(type1);
                IRMemory& mem2 = reinterpret_cast<IRMemory&>(type2);

                return mem1 == mem2;
            }
            else if (type1.type == IRType::STRING_TYPE)
            {
                IRString& str1 = reinterpret_cast<IRString&>(type1);
                IRString& str2 = reinterpret_cast<IRString&>(type2);

                return str1 == str2;
            }
            else if (type1.type == IRType::CLASS_TYPE)
            {
                IRClass& class1 = reinterpret_cast<IRClass&>(type1);
                IRClass& class2 = reinterpret_cast<IRClass&>(type2);

                return class1 == class2;
            }
            else if (type1.type == IRType::CALLEE_TYPE)
            {
                IRCallee& callee1 = reinterpret_cast<IRCallee&>(type1);
                IRCallee& callee2 = reinterpret_cast<IRCallee&>(type2);

                return callee1 == callee2;
            }
            else if (type1.type == IRType::FIELD_TYPE)
            {
                IRField& field1 = reinterpret_cast<IRField&>(type1);
                IRField& field2 = reinterpret_cast<IRField&>(type2);

                return field1 == field2;
            }


            return false;
        }

        /**
         * IRReg class
         */

        /**
         * @brief Constructor of IRReg type.
         * @param reg_id: id of the register this can be an enum if is a well known register, or just an id.
         * @param current_arch: curreng architecture to create the register.
         * @param type_name: string for representing the register.
         * @param type_size: size of the register.
         * @return void
         */
        IRReg::IRReg(std::uint32_t reg_id, int current_arch, std::string type_name, size_t type_size)
            : IRType(REGISTER_TYPE, type_name, type_size)
        {
            this->id = reg_id;
            this->current_arch = current_arch;
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
         * @brief Return string representation of IRReg.
         * @return std::string
         */
        std::string IRReg::to_string()
        {
            std::stringstream stream;
            
            stream << "IRReg: ";

            switch (current_arch)
            {
            case MJOLNIR::x86_arch:
                stream << "[x86: " << x86_regs_name.at(static_cast<x86_regs_t>(id)) << "]";
                break;
            case MJOLNIR::dalvik_arch:
                stream << "[dalvik: v" << id << "]";
                break;
            default:
                stream << "[None: reg" << id << "]";
                break;
            }

            return stream.str();
        }

        /**
         * @brief Compare two IRRegs given by smart pointer.
         * @return bool
         */
        bool IRReg::equal(std::shared_ptr<IRReg> reg)
        {
            return *(this) == *(reg.get());
        }

        /**
         * @brief == operator for IRRegs, we have specific check to do.
         * @param type1: first type for comparison.
         * @param type2: second type for comparison
         * @return bool
         */
        bool operator==(IRReg& type1, IRReg& type2)
        {
            if (type1.id == type2.id && type1.current_arch == type2.current_arch)
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
         * @brief String representation of IRTempReg.
         * @return std::string
         */
        std::string IRTempReg::to_string()
        {
            std::stringstream stream;

            stream << "IRTempReg [" << id << "]";

            return stream.str();
        }

        /**
         * @brief Compare two IRTempReg temporal registers.
         * @param temp_reg: IRTempReg to compare with.
         * @return bool
         */
        bool IRTempReg::equal(std::shared_ptr<IRTempReg> temp_reg)
        {
            return *(this) == *(temp_reg.get());
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
         * @brief Return a string representation of the IRConstInt instruction.
         * @return bool
         */
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
            }

            return stream.str();
        }

        /**
         * @brief Comparison of two IRConstInt instructions with shared pointers.
         * @param const_int: IRConstInt to compare with.
         * @return bool
         */
        bool IRConstInt::equal(std::shared_ptr<IRConstInt> const_int)
        {
            return *(this) == *(const_int.get());
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
         * @brief Return a string representation of IRMemory instruction.
         * @return std::string
         */
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
            }

            return stream.str();
        }

        /**
         * @brief Comparison of two IRMemory by shared_ptr.
         * @param memory: IRMemory instruction to compare.
         * @return bool.
         */
        bool IRMemory::equal(std::shared_ptr<IRMemory> memory)
        {
            return *this == *(memory.get());
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
         * @brief Get a string representation from IRString instruction.
         * @return std::string
         */
        std::string IRString::to_string()
        {
            std::stringstream stream;

            stream << "IRString [" << str_value << "]";

            return stream.str();
        }


        /**
         * @brief Compare two IRString values.
         * @return bool
         */
        bool IRString::equal(std::shared_ptr<IRString> str)
        {
            return *this == *(str.get());
        }

        /**
         * @brief == operator for IRString, we have specific check to do.
         * @param type1: first type for comparison.
         * @param type2: second type for comparison
         * @return bool
         */
        bool operator==(IRString& type1, IRString& type2)
        {
            if (!type1.str_value.compare(type2.str_value))
                return true;
            return false;
        }

        /**
         * IRClass
         */
        /**
         * @brief Constructor of IRClass, this represent the name of a class
         *        that is assigned as a type.
         * @param class_name: name of the class.
         * @param type_name: should be the same value than previous one.
         * @param type_size: should be 0.
         * @return void
         */
        IRClass::IRClass(std::string class_name, std::string type_name, size_t type_size)
            : IRType(CLASS_TYPE, type_name, type_size)
        {
            this->class_name = class_name;
        }

        /**
         * @brief Destructor of IRClass, nothing to be done here.
         * @return void
         */
        IRClass::~IRClass() {}

        /**
         * @brief get the name of the class.
         * @return std::string
         */
        std::string IRClass::get_class()
        {
            return class_name;
        }

        /**
         * @brief Get the name of the type as a string
         * @return std::string
         */
        std::string IRClass::get_type_str()
        {
            return "Class";
        }

        /**
         * @brief Get type of access in this case NONE.
         * @return mem_access_t
         */
        IRClass::mem_access_t IRClass::get_access()
        {
            return NONE_ACCESS;
        }

        /**
         * @brief Return a string representing the IRClass.
         * @return std::string
         */
        
        std::string IRClass::to_string()
        {
            std::stringstream stream;

            stream << "IRClass [" << class_name << "]";

            return stream.str();
        }

        /**
         * @brief Compare two IRClass with smart pointers
         * @return bool
         */
        bool IRClass::equal(std::shared_ptr<IRClass> class_)
        {
            return *(this) == *(class_.get());
        }

        /**
         * @brief Operator == of IRClass we will compare name of classes.
         * @param type1: first class to compare.
         * @param type2: second class to compare.
         * @return bool
         */
        bool operator==(IRClass& type1, IRClass& type2)
        {
            if (!type1.class_name.compare(type2.class_name))
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
         * @brief Return a string representation of the IRCallee type.
         * @return std::string
         */
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
            
            if  (!description.empty())
                str_stream << description;
            
            str_stream << "]";
            
            return str_stream.str();
        }

        /**
         * @brief Check if two shared objects of Callee are the same.
         * @param callee: IRType instruction to compare.
         * @return bool
         */
        bool IRCallee::equal(std::shared_ptr<IRCallee> callee)
        {
            return *this == *(callee.get());
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
            if (!(callee1.compare(callee2)))
                return true;
            return false;
        }

        /**
         * IRField class
         */

        /**
         * @brief Construct a new IRField::IRField object
         * 
         * @param class_name: class name of the field
         * @param type: type from field_t
         * @param field_name: name of the field.
         * @param type_name: some meaninful string name.
         * @param type_size: size of the type (probably here 0)
         */
        IRField::IRField(std::string class_name, 
                    field_t type, 
                    std::string field_name,
                    std::string type_name, 
                    size_t type_size) :
                    IRType(FIELD_TYPE, type_name, type_size)
        {
            this->class_name = class_name;
            this->type = type;
            this->type_class = type_class;
            this->field_name = field_name;
        }

        IRField::IRField(std::string class_name, 
                    std::string type_class_name, 
                    std::string field_name,
                    std::string type_name, 
                    size_t type_size)
                    : IRType(FIELD_TYPE, type_name, type_size)
        {
            this->class_name = class_name;
            this->type = CLASS_F;
            this->type_class = type_class_name;
            this->field_name = field_name;
        }

        /**
         * @brief Destroy the IRField::IRField object
         */
        IRField::~IRField() {}

        /**
         * @brief Getter for class name
         * 
         * @return std::string 
         */
        std::string IRField::get_class_name()
        {
            return this->class_name;
        }

        /**
         * @brief Getter for type name.
         * 
         * @return field_t 
         */
        IRField::field_t IRField::get_type()
        {
            return this->type;
        }

        /**
         * @brief Return if type == CLASS the class name.
         * 
         * @return std::string 
         */
        std::string IRField::get_type_class()
        {
            return this->type_class;
        }

        /**
         * @brief Getter for field name.
         * 
         * @return std::string 
         */
        std::string IRField::get_name()
        {
            return this->field_name;
        }

        /**
         * @brief Get a string representation of IRField
         * 
         * @return std::string
         */
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
            }

            str_stream << "[Name: " << field_name << "]";

            return str_stream.str();
        }

        /**
         * @brief Function to check if two IRField are the same with shared_ptr.
         * 
         * @param field 
         * @return true 
         * @return false 
         */
        bool IRField::equal(std::shared_ptr<IRField> field)
        {
            return *this == *(field.get());
        }

        /**
         * @brief == operator for IRField.
         * 
         * @param field1 
         * @param field2 
         * @return true 
         * @return false 
         */
        bool operator==(IRField& field1, IRField& field2)
        {
            return (!field1.class_name.compare(field2.class_name) &&
                    field1.type == field2.type &&
                    !field1.type_class.compare(field2.type_class) &&
                    !field1.field_name.compare(field2.field_name));
        }
    }
}