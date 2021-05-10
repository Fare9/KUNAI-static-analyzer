#include "dex_classes.hpp"

namespace KUNAI
{
    namespace DEX
    {
        /***
         * EncodedField
         */
        EncodedField::EncodedField(FieldID *field_idx, std::uint64_t access_flags)
        {
            this->field_idx = field_idx;
            this->access_flags = static_cast<DVMTypes::ACCESS_FLAGS>(access_flags);
        }

        EncodedField::~EncodedField() {}

        FieldID *EncodedField::get_field()
        {
            return field_idx;
        }

        DVMTypes::ACCESS_FLAGS EncodedField::get_access_flags()
        {
            return access_flags;
        }
        /***
         * EncodedMethod
         */
        EncodedMethod::EncodedMethod(MethodID *method_id,
                                     std::uint64_t access_flags,
                                     std::uint64_t code_off)
        {
            this->method_id = method_id;
            this->access_flags = static_cast<DVMTypes::ACCESS_FLAGS>(access_flags);
            this->code_off = code_off;
        }

        EncodedMethod::~EncodedMethod() {}

        MethodID *EncodedMethod::get_method()
        {
            return method_id;
        }

        DVMTypes::ACCESS_FLAGS EncodedMethod::get_access_flags()
        {
            return access_flags;
        }

        std::uint64_t EncodedMethod::get_code_offset()
        {
            return code_off;
        }
        /***
         * ClassDataItem
         */
        ClassDataItem::ClassDataItem(std::ifstream &input_file,
                                     std::shared_ptr<DexFields> dex_fields,
                                     std::shared_ptr<DexMethods> dex_methods)
        {
            auto current_offset = input_file.tellg();

            std::uint64_t static_fields_size,
                instance_fields_size,
                direct_methods_size,
                virtual_methods_size;
            std::uint64_t static_field,
                instance_field,
                direct_method,
                virtual_method,
                access_flags,
                code_off;

            static_fields_size = KUNAI::read_uleb128(input_file);
            instance_fields_size = KUNAI::read_uleb128(input_file);
            direct_methods_size = KUNAI::read_uleb128(input_file);
            virtual_methods_size = KUNAI::read_uleb128(input_file);

            for (size_t i = 0; i < static_fields_size; i++)
            {
                static_field = KUNAI::read_uleb128(input_file);

                if (static_field >= dex_fields->get_number_of_fields())
                    throw exceptions::IncorrectFieldId("Error reading ClassDataItem static_field out of field bound");

                access_flags = KUNAI::read_uleb128(input_file);

                static_fields[static_field] = std::make_shared<EncodedField>(dex_fields->get_field_id_by_order(static_field), access_flags);
            }

            for (size_t i = 0; i < instance_fields_size; i++)
            {
                instance_field = KUNAI::read_uleb128(input_file);

                if (instance_field >= dex_fields->get_number_of_fields())
                    throw exceptions::IncorrectFieldId("Error reading ClassDataItem instance_field out of field bound");

                access_flags = KUNAI::read_uleb128(input_file);
                instance_fields[instance_field] = std::make_shared<EncodedField>(dex_fields->get_field_id_by_order(instance_field), access_flags);
            }

            for (size_t i = 0; i < direct_methods_size; i++)
            {
                direct_method = KUNAI::read_uleb128(input_file);

                if (direct_method >= dex_methods->get_number_of_methods())
                    throw exceptions::IncorrectMethodId("Error reading ClassDataItem direct_method out of method bound");

                access_flags = KUNAI::read_uleb128(input_file);

                code_off = KUNAI::read_uleb128(input_file);

                direct_methods[direct_method] = std::make_shared<EncodedMethod>(dex_methods->get_method_by_order(direct_method), access_flags, code_off);
            }

            for (size_t i = 0; i < virtual_methods_size; i++)
            {
                virtual_method = KUNAI::read_uleb128(input_file);

                if (virtual_method >= dex_methods->get_number_of_methods())
                    throw exceptions::IncorrectMethodId("Error reading ClassDataItem virtual_method out of method bound");

                access_flags = KUNAI::read_uleb128(input_file);

                code_off = KUNAI::read_uleb128(input_file);

                virtual_methods[virtual_method] = std::make_shared<EncodedMethod>(dex_methods->get_method_by_order(virtual_method), access_flags, code_off);
            }

            input_file.seekg(current_offset);
        }

        ClassDataItem::~ClassDataItem() {}

        std::uint64_t ClassDataItem::get_number_of_static_fields()
        {
            return static_fields.size();
        }

        std::shared_ptr<EncodedField> ClassDataItem::get_static_field_by_id(std::uint64_t id)
        {
            auto it = static_fields.find(id);

            if (it != static_fields.end())
                return it->second;

            return nullptr;
        }

        std::shared_ptr<EncodedField> ClassDataItem::get_static_field_by_pos(std::uint64_t pos)
        {
            if (pos >= static_fields.size())
                return nullptr;
            return static_fields[pos];
        }

        std::uint64_t ClassDataItem::get_number_of_instance_fields()
        {
            return instance_fields.size();
        }

        std::shared_ptr<EncodedField> ClassDataItem::get_instance_field_by_id(std::uint64_t id)
        {
            auto it = instance_fields.find(id);

            if (it != instance_fields.end())
                return it->second;

            return nullptr;
        }

        std::shared_ptr<EncodedField> ClassDataItem::get_instance_field_by_pos(std::uint64_t pos)
        {
            if (pos >= instance_fields.size())
                return nullptr;
            return instance_fields[pos];
        }

        std::uint64_t ClassDataItem::get_number_of_direct_methods()
        {
            return direct_methods.size();
        }

        std::shared_ptr<EncodedMethod> ClassDataItem::get_direct_method_by_id(std::uint64_t id)
        {
            auto it = direct_methods.find(id);

            if (it != direct_methods.end())
                return it->second;

            return nullptr;
        }

        std::shared_ptr<EncodedMethod> ClassDataItem::get_direct_method_by_pos(std::uint64_t pos)
        {
            if (pos >= direct_methods.size())
                return nullptr;
            return direct_methods[pos];
        }

        std::uint64_t ClassDataItem::get_number_of_virtual_methods()
        {
            return virtual_methods.size();
        }

        std::shared_ptr<EncodedMethod> ClassDataItem::get_virtual_method_by_id(std::uint64_t id)
        {
            auto it = virtual_methods.find(id);

            if (it != virtual_methods.end())
                return it->second;

            return nullptr;
        }

        std::shared_ptr<EncodedMethod> ClassDataItem::get_virtual_method_by_pos(std::uint64_t pos)
        {
            if (pos >= virtual_methods.size())
                return nullptr;
            return virtual_methods[pos];
        }

        /***
         * ClassDef
         */
        ClassDef::ClassDef(classdef_t class_def,
                           std::shared_ptr<DexStrings> dex_str,
                           std::shared_ptr<DexTypes> dex_types,
                           std::shared_ptr<DexFields> dex_fields,
                           std::shared_ptr<DexMethods> dex_methods,
                           std::ifstream &input_file,
                           std::uint64_t file_size)
        {
            this->class_idx[class_def.class_idx] = dynamic_cast<Class *>(dex_types->get_type_from_order(class_def.class_idx));
            if (class_def.superclass_idx == DVMTypes::NO_INDEX)
                this->superclass_idx[class_def.superclass_idx] = nullptr;
            else
                this->superclass_idx[class_def.superclass_idx] = dynamic_cast<Class *>(dex_types->get_type_from_order(class_def.superclass_idx));

            if (class_def.source_file_idx == DVMTypes::NO_INDEX)
                this->source_file_idx[class_def.source_file_idx] = nullptr;
            else
                this->source_file_idx[class_def.source_file_idx] = dex_str->get_string_from_order(class_def.source_file_idx);
            this->access_flag = static_cast<DVMTypes::ACCESS_FLAGS>(class_def.access_flags);
            this->interfaces_off        = class_def.interfaces_off;
            this->annotations_off       = class_def.annotations_off;
            this->classess_off          = class_def.class_data_off;
            this->static_values_off     = class_def.static_values_off;
            if (!parse_class_defs(input_file, file_size, dex_str, dex_types, dex_fields, dex_methods))
                throw exceptions::ParserReadingException("Error reading DEX ClassDef");
        }

        ClassDef::~ClassDef() {}

        bool ClassDef::parse_class_defs(std::ifstream &input_file,
                                        std::uint64_t file_size,
                                        std::shared_ptr<DexStrings> dex_str,
                                        std::shared_ptr<DexTypes> dex_types,
                                        std::shared_ptr<DexFields> dex_fields,
                                        std::shared_ptr<DexMethods> dex_methods)
        {
            auto current_offset = input_file.tellg();
            size_t i;
            std::uint32_t size;
            std::uint16_t interface;

            // parse first the interfaces
            if (interfaces_off > 0)
            {
                input_file.seekg(interfaces_off);

                if (!KUNAI::read_data_file<std::uint32_t>(size, sizeof(std::uint32_t), input_file))
                    return false;

                for (i = 0; i < size; i++)
                {
                    if (!KUNAI::read_data_file<std::uint16_t>(interface, sizeof(std::uint16_t), input_file))
                        return false;

                    if (interface >= dex_types->get_number_of_types())
                        throw exceptions::IncorrectTypeId("Error reading DEX ClassDefs interface out of type bound");

                    interfaces[interface] = dynamic_cast<Class *>(dex_types->get_type_from_order(interface));
                }
            }

            // parse annotations
            if (annotations_off > 0)
            {
                input_file.seekg(annotations_off);

                // ToDo
            }

            // parse class data
            if (classess_off > 0)
            {
                input_file.seekg(classess_off);

                class_data_items = std::make_shared<ClassDataItem>(input_file, dex_fields, dex_methods);
            }

            if (static_values_off > 0)
            {
                input_file.seekg(static_values_off);

                //std::uint64_t static_values_size = KUNAI::read_uleb128(input_file);

                // ToDo
            }

            input_file.seekg(current_offset);
            return true;
        }

        Class *ClassDef::get_class_idx()
        {
            return class_idx.begin()->second;
        }

        DVMTypes::ACCESS_FLAGS ClassDef::get_access_flags()
        {
            return access_flag;
        }

        Class *ClassDef::get_superclass_idx()
        {
            return superclass_idx.begin()->second;
        }

        std::string *ClassDef::get_source_file_idx()
        {
            return source_file_idx.begin()->second;
        }

        std::uint64_t ClassDef::get_number_of_interfaces()
        {
            return interfaces.size();
        }

        Class *ClassDef::get_interface_by_class_id(std::uint16_t id)
        {
            auto it = interfaces.find(id);

            if (it != interfaces.end())
                return it->second;

            return nullptr;
        }

        Class *ClassDef::get_interface_by_pos(std::uint64_t pos)
        {
            if (pos >= interfaces.size())
                return nullptr;
            return interfaces[pos];
        }

        std::shared_ptr<ClassDataItem> ClassDef::get_class_data()
        {
            return class_data_items;
        }

        /***
         * DexClasses
         */
        DexClasses::DexClasses(std::ifstream &input_file,
                               std::uint64_t file_size,
                               std::uint32_t number_of_classes,
                               std::uint32_t offset,
                               std::shared_ptr<DexStrings> dex_str,
                               std::shared_ptr<DexTypes> dex_types,
                               std::shared_ptr<DexFields> dex_fields,
                               std::shared_ptr<DexMethods> dex_methods)
        {
            this->number_of_classes = number_of_classes;
            this->offset = offset;
            this->dex_str = dex_str;
            this->dex_types = dex_types;
            this->dex_fields = dex_fields;
            this->dex_methods = dex_methods;

            if (!parse_classes(input_file, file_size))
                throw exceptions::ParserReadingException("Error reading DEX classes");
        }

        DexClasses::~DexClasses()
        {
            if (!class_defs.empty())
            {
                class_defs.clear();
            }
        }

        std::uint32_t DexClasses::get_number_of_classes()
        {
            return number_of_classes;
        }

        std::shared_ptr<ClassDef> DexClasses::get_class_by_pos(std::uint64_t pos)
        {
            if (pos >= class_defs.size())
                return nullptr;
            return class_defs[pos];
        }

        bool DexClasses::parse_classes(std::ifstream &input_file, std::uint64_t file_size)
        {
            auto current_offset = input_file.tellg();
            size_t i;
            ClassDef::classdef_t class_def_struct;
            std::shared_ptr<ClassDef> class_def;

            input_file.seekg(offset);

            for (i = 0; i < number_of_classes; i++)
            {
                if (!KUNAI::read_data_file<ClassDef::classdef_t>(class_def_struct, sizeof(ClassDef::classdef_t), input_file))
                    return false;

                if (class_def_struct.class_idx >= dex_types->get_number_of_types())
                    throw exceptions::IncorrectTypeId("Error reading DEX classes class_idx out of type bound");

                if (class_def_struct.access_flags > DVMTypes::ACC_DECLARED_SYNCHRONIZED)
                    throw exceptions::IncorrectValue("Error reading DEX classes access_flags incorrect value");

                if ((class_def_struct.superclass_idx != DVMTypes::NO_INDEX) && (class_def_struct.superclass_idx >= dex_types->get_number_of_types()))
                    throw exceptions::IncorrectTypeId("Error reading DEX classes superclass_idx out of type bound");

                if (class_def_struct.interfaces_off > file_size)
                    throw exceptions::OutOfBoundException("Error reading DEX classes interfaces_off out of file bound");

                if ((class_def_struct.source_file_idx != DVMTypes::NO_INDEX) && (class_def_struct.source_file_idx >= dex_str->get_number_of_strings()))
                    throw exceptions::IncorrectStringId("Error reading DEX classes source_file_idx out of string bound");

                if (class_def_struct.annotations_off > file_size)
                    throw exceptions::OutOfBoundException("Error reading DEX classes annotations_off out of file bound");

                if (class_def_struct.class_data_off > file_size)
                    throw exceptions::OutOfBoundException("Error reading DEX classes class_data_off out of file bound");

                if (class_def_struct.static_values_off > file_size)
                    throw exceptions::OutOfBoundException("Error reading DEX classes static_values_off out of file bound");

                class_def = std::make_shared<ClassDef>(class_def_struct, dex_str, dex_types, dex_fields, dex_methods, input_file, file_size);
                class_defs.push_back(class_def);
            }

            input_file.seekg(current_offset);
            return true;
        }

    }
}