//--------------------------------------------------------------------*- C++ -*-
// Kunai-static-analyzer: library for doing analysis of dalvik files
// @author Farenain <kunai.static.analysis@gmail.com>
// @author Ernesto Java <javaernesto@gmail.com>
//
// @file encoded.hpp
// @brief This file contains all the information from encoded data
// these classes are for annotations, arrays, fields, try-catch
// information, etc.

#ifndef KUNAI_DEX_PARSER_ENCODED_HPP
#define KUNAI_DEX_PARSER_ENCODED_HPP

#include "Kunai/DEX/parser/strings.hpp"
#include "Kunai/DEX/parser/fields.hpp"
#include "Kunai/DEX/parser/fields.hpp"
#include "Kunai/DEX/parser/methods.hpp"
#include "Kunai/DEX/DVM/dvm_types.hpp"
#include "Kunai/Utils/kunaistream.hpp"

#include <iostream>
#include <vector>

namespace KUNAI
{
namespace DEX
{
    /// @brief Forward declaration of EncodedValue for allowing its usage
    /// in EncodedArray and AnnotationElement
    class EncodedValue;
    using encodedvalue_t = std::unique_ptr<EncodedValue>;

    /// @brief Information of an array with encoded values
    class EncodedArray
    {
        /// @brief size of the array
        std::uint64_t array_size;
        /// @brief encoded values of the array
        std::vector<encodedvalue_t> values;
    public:
        /// @brief Constructor of the encoded array
        EncodedArray() = default;
        /// @brief Destructor of encoded array
        ~EncodedArray() = default;

        /// @brief Parse the encoded Array
        /// @param stream stream where to read data
        /// @param types object with types for parsing encoded array
        /// @param strings object with strings for parsing encoded array
        void parse_encoded_array(stream::KunaiStream* stream,
                                Types* types,
                                Strings* strings);

        /// @brief get the size of the array
        /// @return size of the array
        std::uint64_t get_array_size() const
        {
            return array_size;
        }

        /// @brief Get constant reference to encoded values
        /// @return constant reference to encoded values
        const std::vector<encodedvalue_t>& get_values() const
        {
            return values;
        }

        /// @brief Get a reference to encoded values
        /// @return reference to encoded values
        std::vector<encodedvalue_t>& get_values()
        {
            return values;
        }
    };

    using encodedarray_t = std::unique_ptr<EncodedArray>;

    /// @brief Annotation element with value and a name
    /// this is contained in the EncodedAnnotation class
    class AnnotationElement
    {
        /// @brief name of the annotation element
        std::string& name;
        /// @brief element value
        encodedvalue_t value;
    public:
        /// @brief Constructor of the annotation element
        /// @param name name of the annotation
        /// @param value value of the annotation
        AnnotationElement(std::string& name, encodedvalue_t value)
            : name(name), value(std::move(value))
        {}

        /// @brief Destructor of the annotation element
        ~AnnotationElement() = default;

        /// @brief Get the name of the annotation
        /// @return name of annotation
        std::string& get_name()
        {
            return name;
        }

        /// @brief Get the value of the annotation
        /// @return value of the annotation
        EncodedValue* get_value()
        {
            return value.get();
        }
    };

    using annotationelement_t = std::unique_ptr<AnnotationElement>;

    /// @brief Class to parse and create a vector of
    /// Annotations
    class EncodedAnnotation
    {
        /// @brief Type of the annotation, this has
        /// to be a class, nor an Array or Fundamental
        DVMType* type;
        /// @brief Size of the annotation array
        std::uint64_t size;
        /// @brief vector of annotation elements
        std::vector<annotationelement_t> elements;
    public:
        /// @brief Constructor from EncodedAnnotation
        EncodedAnnotation() = default;
        /// @brief Destructor from EncodedAnnotation
        ~EncodedAnnotation() = default;
        
        /// @brief Function to parse an encoded annotation
        /// @param stream stream with the DEX file
        /// @param types types for parsing the encoded annotation
        /// @param strings strings for parsing the encoded annotation
        void parse_encoded_annotation(
            stream::KunaiStream* stream,
            Types* types,
            Strings* strings
        );

        /// @brief Get the type of the annotations
        /// @return annotations type
        DVMType* get_annotation_type()
        {
            return type;
        }

        /// @brief Get size of annotations
        /// @return number of annotation elements
        std::uint64_t get_size() const
        {
            return size;
        }

        /// @brief Get a constant reference to all the annotation elements
        /// @return constant reference to all annotation elements
        const std::vector<annotationelement_t>& get_annotations() const
        {
            return elements;
        }

        /// @brief Get a reference to all the annotation elements
        /// @return reference to all annotation elements
        std::vector<annotationelement_t>& get_annotations()
        {
            return elements;
        }

        /// @brief Get an annotation element by position
        /// @param pos position to retrieve
        /// @return pointer to AnnotationElement
        AnnotationElement* get_annotation_by_pos(std::uint32_t pos);
    };

    /// @brief encoded piece of (nearly) arbitrary hierarchically structured data.
    class EncodedValue
    {
        /// @brief type of value
        TYPES::value_format value_type;
        /// @brief number of values
        std::uint8_t value_args;
        /// @brief save the value in an array of bytes
        /// the value can have different format
        std::vector<std::uint8_t> values;
        /// @brief In case the value is an array
        /// store an array
        EncodedArray array_data;
        /// @brief In case it is an annotation, store
        /// an encoded annotation
        EncodedAnnotation annotation;
    
    public:
        /// @brief Constructor of EncodedValue
        EncodedValue() = default;
        /// @brief Destructor of EncodedValue
        ~EncodedValue() = default;

        /// @brief Parse the encoded value
        /// @param stream stream to read data
        /// @param types object with types for parsing encoded values
        /// @param strings object with strings for parsing encoded values
        void parse_encoded_value(stream::KunaiStream* stream,
                                Types* types,
                                Strings* strings);

        /// @brief Get the enum with the value type
        /// @return value_format of the encoded value
        TYPES::value_format get_value_type() const
        {
            return value_type;
        }

        /// @brief Return the size of the value
        /// @return size of the value
        std::uint8_t size_of_value() const
        {
            return value_args;
        }

        /// @brief Get a constant reference to the values
        /// @return constant reference to values
        const std::vector<std::uint8_t>& get_values() const
        {
            return values;
        }

        /// @brief Get a reference to the values
        /// @return reference to the values
        std::vector<std::uint8_t>& get_values()
        {
            return values;
        }

        /// @brief in case the value is an array return it
        /// @return reference to encoded array
        const EncodedArray& get_array() const
        {
            return array_data;
        }

        /// @brief get a reference to the array in case value is an array
        /// @return reference to encoded array
        EncodedArray& get_array()
        {
            return array_data;
        }

        /// @brief get a reference to annotation in case it is annotation
        /// @return reference to encoded annotation
        const EncodedAnnotation& get_annotation() const
        {
            return annotation;
        }

        /// @brief get a reference to annotation in case it is annotation
        /// @return reference to encoded annotation
        EncodedAnnotation& get_annotation()
        {
            return annotation;
        }
    };

    /// @brief Class that represent field information
    /// it contains a FieldID and also the access flags.
    class EncodedField
    {
        /// @brief FieldID of the EncodedField
        FieldID * field_idx;
        /// @brief access flags for the field
        TYPES::access_flags flags;
        /// @brief Initial Value
        EncodedArray* initial_value;
    public:
        /// @brief Constructor of an encoded field
        /// @param field_idx FieldID for the encoded field
        /// @param flags 
        EncodedField(FieldID * field_idx, TYPES::access_flags flags)
            : field_idx(field_idx), flags(flags)
        {
        }

        /// @brief Destructor of Encoded Field
        ~EncodedField() = default;

        /// @brief Get a constant pointer to the FieldID
        /// @return constant pointer to the FieldID
        const FieldID* get_field() const
        {
            return field_idx;
        }

        /// @brief Get a pointer to the FieldID
        /// @return pointer to the FieldID
        FieldID* get_field()
        {
            return field_idx;
        }

        /// @brief Get the access flags from the Field
        /// @return access flags
        TYPES::access_flags  get_access_flags() const
        {
            return flags;
        }

        /// @brief Those fields that are static contains an initial value
        /// @param initial_value 
        void set_initial_value(EncodedArray* initial_value)
        {
            this->initial_value = initial_value;
        }

        EncodedArray* get_initial_value()
        {
            return initial_value;
        }
    };

    using encodedfield_t = std::unique_ptr<EncodedField>;

    /// @brief Type of exception to catch with its address
    class EncodedTypePair
    {
        /// @brief Type of the exception to catch
        DVMType * type;
        /// @brief bytecode address of the associated exception handler
        std::uint64_t addr;
    public:
        /// @brief Constructor for EncodedTypePair
        /// @param type type of the exception
        /// @param addr address associated with the exception
        EncodedTypePair(DVMType * type, std::uint64_t addr)
            : type(type), addr(addr)
        {}
        /// @brief Destructor of EncodedTypePair
        ~EncodedTypePair() = default;

        /// @brief Get the bytecode address of the associated exception handler
        /// @return address of the exception handler
        std::uint64_t get_addr() const
        {
            return addr;
        }

        /// @brief Get a constant pointer to the exception type
        /// @return pointer to exception type
        const DVMType* get_exception_type() const
        {
            return type;
        }

        /// @brief Get pointer to exception type
        /// @return pointer to exception type
        DVMType* get_exception_type()
        {
            return type;
        }
    };

    using encodedtypepair_t = std::unique_ptr<EncodedTypePair>;

    /// @brief Information of catch handlers
    class EncodedCatchHandler
    {
        /// @brief Size of the vector of EncodedTypePair
        /// if > 0 indicates the size of the handlers
        /// if == 0 there are no handlers nor catch_all_addr
        /// if < 0 no handlers and catch_all_addr is set
        std::int64_t size;
        /// @brief Vector of EncodedTypePair
        std::vector<encodedtypepair_t> handlers;
        /// @brief bytecode of the catch all-handler.
        /// This element is only present if size is non-positive. 
        std::uint64_t catch_all_addr = 0;
        /// @brief Offset where the encoded catch handler is
        /// in the file
        std::uint64_t offset;
    public:
        /// @brief Constructor of EncodedCatchHandler
        EncodedCatchHandler() = default;
        /// @brief Destructor of EncodedCatchHandler
        ~EncodedCatchHandler() = default;

        /// @brief Parse all the encoded type pairs
        /// @param stream stream with DEX data
        /// @param types types for the EncodedTypePair
        void parse_encoded_catch_handler(stream::KunaiStream* stream,
                                         Types* types);

        /// @brief Check value of size to test if there are encodedtypepairs 
        /// @return if there are explicit typed catches
        bool has_explicit_typed_catches() const
        {
            if (size >= 0)
                return true; // user should check size of handlers
            return false;
        }

        /// @brief Get the size of the EncodedCatchHandler
        /// @return value of size, refer to `size` documentation
        /// to check the possible values
        std::int64_t get_size() const
        {
            return size;
        }

        /// @brief Return the value from catch_all_addr
        /// @return catch_all_addr value
        std::uint64_t get_catch_all_addr() const
        {
            return catch_all_addr;
        }

        /// @brief Get a constant reference to the vector of EncodedTypePair
        /// @return constant reference to vector of EncodedTypePair
        const std::vector<encodedtypepair_t>& get_handlers() const
        {
            return handlers;
        }

        /// @brief Get a reference to the vector of EncodedTypePair
        /// @return reference to vector of EncodedTypePair
        std::vector<encodedtypepair_t>& get_handlers()
        {
            return handlers;
        }

        /// @brief Get the offset where encoded catch handler is
        /// @return offset of encoded catch handler
        std::uint64_t get_offset() const
        {
            return offset;
        }

        EncodedTypePair *get_handler_by_pos(std::uint64_t pos);
    };

    using encodedcatchhandler_t = std::unique_ptr<EncodedCatchHandler>;

    /// @brief Class that specify the information from a try
    /// specifies address, number of instructions, and offset
    /// of handlers
    class TryItem
    {
    public:
        /// @brief Structure with the information from a
        /// Try code
#pragma pack(0)
        struct try_item_struct_t
        {
            std::uint32_t start_addr;   //! start address of block of code covered by this entry.
                                        //! Count of 16-bit code units to start of first.
            std::uint16_t insn_count;   //! number of 16-bit code units covered by this entry.
            std::uint16_t handler_off;  //! offset in bytes from starts of associated encoded_catch_handler_list
                                        //! to encoded_catch_handler for this entry.
        };
#pragma pack()

    private:
        /// @brief structure of try_item
        try_item_struct_t try_item_struct;

    public:
        /// @brief Constructor of TryItem
        TryItem() = default;
        /// @brief Destructor of TryItem
        ~TryItem() = default;
        /// @brief Parse a `try_item_struct_t` from stream
        /// @param stream DEX to read `try_item_struct_t`
        void parse_try_item(stream::KunaiStream* stream);

        /// @brief Get the start address of the try block
        /// @return start address of try block
        std::uint32_t get_start_addr() const
        {
            return try_item_struct.start_addr;
        }

        /// @brief Get the number of instruction counts from the try
        /// @return instruction count from try block
        std::uint16_t get_insn_count() const
        {
            return try_item_struct.insn_count;
        }

        /// @brief Get offset of handler from try
        /// @return handler offset
        std::uint16_t get_handler_off() const
        {
            return try_item_struct.handler_off;
        }
    };

    using tryitem_t = std::unique_ptr<TryItem>;

    /// @brief Save the information of the code from a Method
    class CodeItemStruct
    {
    public:
        /// @brief Structure with information about a method code
        struct code_item_struct_t
        {
            std::uint16_t registers_size;       //! number of registers used in the code
            std::uint16_t ins_size;             //! number of words of incoming arguments to the method
            std::uint16_t outs_size;            //! number of words of outgoung arguments space required
                                                //! for method invocation.
            std::uint16_t tries_size;           //! number of TryItem, can be 0
            std::uint32_t debug_info_off;       //! offset to debug_info_item
            std::uint32_t insns_size;           //! size of instruction list
        };
    private:
        /// @brief Information of code item
        code_item_struct_t code_item;
        /// @brief Vector with the bytecode of the instructions
        std::vector<std::uint8_t> instructions_raw;
        /// @brief Vector of try_item
        std::vector<tryitem_t> try_items;
        /// @brief encoded catch handler offset for exception
        /// calculation
        std::uint64_t encoded_catch_handler_list_offset;
        /// @brief encoded catch handler size
        std::uint64_t encoded_catch_handler_size;
        /// @brief encoded_catch_handler list
        std::vector<encodedcatchhandler_t> encoded_catch_handlers;
    public:
        /// @brief Constructor of CodeItemStruct
        CodeItemStruct() = default;
        /// @brief Destructor of CodeItemStruct
        ~CodeItemStruct() = default;

        /// @brief Parser for the CodeItemStruct
        /// @param stream DEX file where to read data
        /// @param types types of the DEX
        void parse_code_item_struct(
            stream::KunaiStream* stream,
            Types* types
        );
        /// @brief Get the number of registers used in a method
        /// @return number of registers
        std::uint16_t get_registers_size() const
        {
            return code_item.registers_size;
        }

        /// @brief Get the number of words incoming arguments to the method
        /// @return number of words incoming arguments
        std::uint16_t get_incomings_args() const
        {
            return code_item.ins_size;
        }

        /// @brief Get the number of words outgoing argument space required by the code
        /// @return number of words outgoing argument space
        std::uint16_t get_outgoing_args() const
        {
            return code_item.outs_size;
        }

        /// @brief Get the number of try items in the method
        /// @return number of try items
        std::uint16_t get_number_try_items() const
        {
            return code_item.tries_size;
        }

        /// @brief Get the offset to the debug information
        /// @return offset to debug information
        std::uint16_t get_offset_to_debug_info() const
        {
            return code_item.debug_info_off;
        }

        /// @brief Get size of the dalvik instructions (number of opcodes)
        /// @return size of dalvik instructions
        std::uint16_t get_instructions_size() const
        {
            return code_item.insns_size;
        }

        /// @brief Get a constant reference to the vector with the bytecode
        /// @return constant reference to the bytecode of the method
        const std::vector<std::uint8_t>& get_bytecode() const
        {
            return instructions_raw;
        }

        /// @brief Get a reference to the vector with the bytecode
        /// @return reference to bytecode of the method
        std::vector<std::uint8_t>& get_bytecode()
        {
            return instructions_raw;
        }
    
        /// @brief Get a constant reference to the vector of Try Items
        /// @return constant reference to all try items
        const std::vector<tryitem_t>& get_try_items() const
        {
            return try_items;
        }

        /// @brief Get a reference to the vector of Try Items
        /// @return reference to all try items
        std::vector<tryitem_t>& get_try_items()
        {
            return try_items;
        }

        /// @brief Return the offset where encoded catch handler is read
        /// @return offset to encoded catch handler list
        std::uint64_t get_encoded_catch_handler_offset()
        {
            return encoded_catch_handler_list_offset;
        }

        /// @brief Get a constant reference to the catch handlers vector
        /// @return constant reference catch handlers of the method
        const std::vector<encodedcatchhandler_t>& get_encoded_catch_handlers() const
        {
            return encoded_catch_handlers;
        }

        /// @brief Get a reference to the catch handlers vector
        /// @return reference to catch handlers of the method
        std::vector<encodedcatchhandler_t>& get_encoded_catch_handlers()
        {
            return encoded_catch_handlers;
        }
    };

    using codeitemstruct_t = std::unique_ptr<CodeItemStruct>;
    
    /// @brief Class that represent the information from a Method
    class EncodedMethod
    {
        /// @brief MethodID that represents this encoded method
        MethodID* method_id;
        /// @brief Access flags of the method
        TYPES::access_flags access_flags;
        /// @brief Code Item of the method
        CodeItemStruct code_item;
    public:
        /// @brief Constructor of Encoded method
        /// @param method_id method of the current encoded method
        /// @param access_flags access flags of access of the method
        EncodedMethod(MethodID* method_id, TYPES::access_flags access_flags)
            : method_id(method_id), access_flags(access_flags)
        {}

        /// @brief Destructor of Encoded method
        ~EncodedMethod() = default;

        /// @brief Parse the encoded method, this will parse the code item
        /// @param stream stream with DEX file
        /// @param code_off offset where code item struct
        /// @param types types from the DEX
        void parse_encoded_method(stream::KunaiStream* stream,
                                  std::uint64_t code_off,
                                  Types* types);
        
        /// @brief Get a constant pointer to the MethodID of the method
        /// @return constant pointer to the MethodID
        const MethodID* getMethodID() const
        {
            return method_id;
        }

        /// @brief Get a pointer to the MethodID of the encoded method
        /// @return pointer to the MethodID
        MethodID* getMethodID()
        {
            return method_id;
        }

        /// @brief Get access flags from the encoded method
        /// @return access flags of method
        TYPES::access_flags get_access_flags() const
        {
            return access_flags;
        }

        /// @brief Get the code item from the encoded method
        /// @return constant reference to code item
        const CodeItemStruct& get_code_item() const
        {
            return code_item;
        }

        /// @brief Get the code item from the encoded method
        /// @return reference to code item
        CodeItemStruct& get_code_item()
        {
            return code_item;
        }
    };

    using encodedmethod_t = std::unique_ptr<EncodedMethod>;
} // namespace DEX
} // namespace KUNAI


#endif