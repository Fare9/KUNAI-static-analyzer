/**
 * @file ir_type.hpp
 * @author Farenain
 * 
 * @brief This are the different types that a user will be able to use
 *        for the analysis of the IR, this will be constant values (integers,
 *        floats, etc), registers, memory addresses or even fields specific
 *        from Java/Android.
 */

#include <iostream>
#include "ir_expr.hpp"
#include "arch/ir_x86.hpp"

namespace KUNAI
{
    namespace MJOLNIR
    {
        class IRType : public IRExpr
        {
        public:
            enum type_t
            {
                REGISTER_TYPE = 0,
                TEMP_REGISTER_TYPE,
                CONST_INT_TYPE,
                CONST_FLOAT_TYPE,
                FIELD_TYPE,
                MEM_TYPE,
                STRING_TYPE,
                CALLEE_TYPE,
                NONE_TYPE = 99
            };

            enum mem_access_t
            {
                LE_ACCESS = 0, //! little-endian access
                BE_ACCESS,     //! big-endian access
                ME_ACCESS,     //! This shouldn't commonly happen?
                NONE_ACCESS = 99
            };

            IRType(type_t type, std::string type_name, size_t type_size);
            ~IRType();

            std::string get_type_name();
            size_t get_type_size();

            type_t get_type();
            virtual std::string get_type_str() = 0;
            virtual mem_access_t get_access() = 0;

            void write_annotations(std::string annotations);
            std::string read_annotations();

            friend bool operator==(IRType&, IRType&);
        private:
            //! type value as a type_t
            type_t type;

            //! name used to represent the type in IR representation.
            std::string type_name;

            //! size of the type, this can vary depending on architecture
            //! and so on.
            size_t type_size;

            //! annotations are there for you to write whatever you want
            std::string annotations;
        };

        class IRReg : public IRType
        {
        public:
            IRReg(std::uint32_t reg_id, std::string type_name, size_t type_size);
            ~IRReg();

            std::uint32_t get_id();

            std::string get_type_str();
            mem_access_t get_access();

            friend bool operator==(IRReg&, IRReg&);
        private:
            //! id of the register, this will be an enum
            //! in case the arquitecture contains a known set
            //! of registers, for example x86-64 will have a
            //! well known set of registers, e.g. EAX, AX, RSP
            //! RIP, etc.
            //! Other arquitectures like DEX VM will not have
            //! an specific set.
            std::uint32_t id;
        };

        class IRTempReg : public IRType
        {
        public:
            IRTempReg(std::uint32_t reg_id, std::string type_name, size_t type_size);
            ~IRTempReg();

            std::uint32_t get_id();

            std::string get_type_str();
            mem_access_t get_access();

            friend bool operator==(IRTempReg&, IRTempReg&);
        private:
            //! This id will be just an incremental number
            //! as these are temporal registers.
            std::uint32_t id;
        };

        class IRConstInt : public IRType
        {
        public:
            IRConstInt(std::uint64_t value, bool is_signed, mem_access_t byte_order, std::string type_name, size_t type_size);
            ~IRConstInt();

            bool get_is_signed();
            
            std::string get_type_str();
            mem_access_t get_access();

            friend bool operator==(IRConstInt&, IRConstInt&);
        private:
            //! Value of the integer
            std::uint64_t value;
            //! Check to know if the constant is a unsigned
            //! or signed value.
            bool is_signed;
            //! byte order of the value.
            mem_access_t byte_order;
        };

        class IRMemory : public IRType
        {
        public:
            IRMemory(std::uint64_t mem_address, std::int32_t offset, mem_access_t byte_order, std::string type_name, size_t type_size);
            ~IRMemory();

            std::uint64_t get_mem_address();
            std::int32_t get_offset();

            std::string get_type_str();
            mem_access_t get_access();

            friend bool operator==(IRMemory&, IRMemory&);
        private:
            //! accessed address
            std::uint64_t mem_address;
            //! offset of the memory accessed
            std::int32_t offset;
            //! byte order of the memory.
            mem_access_t byte_order;
        };

        class IRString : IRType
        {
        public:
            IRString(std::string str_value, std::string type_name, size_t type_size);
            ~IRString();

            std::string get_str_value();

            std::string get_type_str();
            mem_access_t get_access();

            friend bool operator==(IRString&, IRString&);
        private:
            //! string value, probably nothing more will be here
            std::string str_value;
        };

        class IRCallee : IRType
        {
        public:
            IRCallee(std::uint64_t addr,
                     std::string name,
                     std::string class_name,
                     int n_of_params,
                     std::string description,
                     std::string type_name, 
                     size_t type_size);
            ~IRCallee();

            std::uint64_t get_addr();
            std::string get_name();
            std::string get_class_name();
            int get_number_of_params();
            std::string get_description();

            std::string get_type_str();
            mem_access_t get_access();

            friend bool operator==(IRCallee&, IRCallee&);
        private:
            //! for those functions of binary formats we will mostly have the address
            //! only, these can be from a library, from the same binary, etc.
            std::uint64_t addr;
            //! name of the callee function or method, this can be resolved from the
            //! binary symbols if those exist or is given in case of other formats.
            std::string name;
            //! in case it is a method, probably we will need to know class name
            //! for possible analysis which requires to know about a calls.
            std::string class_name;
            //! there are cases where functions/methods can have the same name but
            //! different parameters, you can give the number of parameters (if recognized)
            //! or the string with the description of the method
            int n_of_params;
            std::string description;
        };
    }
}