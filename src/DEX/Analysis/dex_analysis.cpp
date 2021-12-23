#include "dex_analysis.hpp"

namespace KUNAI
{
    namespace DEX
    {

        Analysis::Analysis(std::shared_ptr<DexParser> dex_parser, std::shared_ptr<DalvikOpcodes> dalvik_opcodes, std::map<std::tuple<std::shared_ptr<ClassDef>, std::shared_ptr<EncodedMethod>>, std::map<std::uint64_t, std::shared_ptr<Instruction>>> instructions) : created_xrefs(false),
                                                                                                                                                                                                                                                                         dalvik_opcodes(dalvik_opcodes),
                                                                                                                                                                                                                                                                         instructions(instructions)
        {
            if (dex_parser)
                this->add(dex_parser);
        }

        void Analysis::add(std::shared_ptr<DexParser> dex_parser)
        {
            // first of all add it to vector
            this->dex_parsers.push_back(dex_parser);

            auto class_dex = dex_parser->get_classes();

            for (size_t i = 0; i < class_dex->get_number_of_classes(); i++)
            {
                auto class_def_item = class_dex->get_class_by_pos(i);

                if (class_def_item == nullptr)
                    continue;

                classes[class_def_item->get_class_idx()->get_name()] = std::make_shared<ClassAnalysis>(class_def_item);
                auto new_class = classes[class_def_item->get_class_idx()->get_name()];

                // get class data item
                auto class_data_item = class_def_item->get_class_data();

                if (class_data_item == nullptr)
                    continue;

                // add the direct methods
                for (size_t j = 0; j < class_data_item->get_number_of_direct_methods(); j++)
                {
                    auto encoded_method = class_data_item->get_direct_method_by_pos(j);

                    methods[encoded_method->full_name()] = std::make_shared<MethodAnalysis>(encoded_method, dalvik_opcodes, instructions[{class_def_item, encoded_method}]);
                    auto new_method = methods[encoded_method->full_name()];

                    new_class->add_method(new_method);

                    method_hashes[{class_def_item->get_class_idx()->get_name(), *encoded_method->get_method()->get_method_name(), encoded_method->get_method()->get_method_prototype()->get_proto_str()}] = new_method;
                }

                // add the virtual methods
                for (size_t j = 0; j < class_data_item->get_number_of_virtual_methods(); j++)
                {
                    auto encoded_method = class_data_item->get_virtual_method_by_pos(j);

                    methods[encoded_method->full_name()] = std::make_shared<MethodAnalysis>(encoded_method, dalvik_opcodes, instructions[{class_def_item, encoded_method}]);
                    auto new_method = methods[encoded_method->full_name()];

                    new_class->add_method(new_method);

                    method_hashes[{class_def_item->get_class_idx()->get_name(), *encoded_method->get_method()->get_method_name(), encoded_method->get_method()->get_method_prototype()->get_proto_str()}] = new_method;
                }
            }
        }

        void Analysis::create_xref()
        {
            if (created_xrefs)
            {
                std::cerr << "Requested create_xref() method more than once" << std::endl;
                std::cerr << "This will not work again, function will exit right now" << std::endl;
                std::cerr << "Please if you want to analze various dex parsers, add all of them first, then call this function." << std::endl;

                return;
            }

            created_xrefs = true;

            for (size_t i = 0; i < dex_parsers.size(); i++)
            {
                auto dex_parser = dex_parsers[i];

                auto class_dex = dex_parser->get_classes();

                for (size_t j = 0; j < class_dex->get_number_of_classes(); j++)
                {
                    auto class_def_item = class_dex->get_class_by_pos(i);

                    _create_xref(class_def_item);
                }
            }
        }

        bool Analysis::is_class_present(std::string class_name)
        {
            return (classes.find(class_name) == classes.end());
        }

        std::shared_ptr<ClassAnalysis> Analysis::get_class_analysis(std::string class_name)
        {
            if (classes.find(class_name) == classes.end())
                return nullptr;
            return classes[class_name];
        }

        std::vector<std::shared_ptr<ClassAnalysis>> Analysis::get_classes()
        {
            std::vector<std::shared_ptr<ClassAnalysis>> classes_vector;

            for (auto it = classes.begin(); it != classes.end(); it++)
            {
                classes_vector.push_back(it->second);
            }

            return classes_vector;
        }

        std::vector<std::shared_ptr<ClassAnalysis>> Analysis::get_external_classes()
        {
            std::vector<std::shared_ptr<ClassAnalysis>> external_classes;

            for (auto it = classes.begin(); it != classes.end(); it++)
            {
                if (it->second->is_class_external())
                    external_classes.push_back(it->second);
            }

            return external_classes;
        }

        std::vector<std::shared_ptr<ClassAnalysis>> Analysis::get_internal_classes()
        {
            std::vector<std::shared_ptr<ClassAnalysis>> internal_classes;

            for (auto it = classes.begin(); it != classes.end(); it++)
            {
                if (!it->second->is_class_external())
                    internal_classes.push_back(it->second);
            }

            return internal_classes;
        }

        std::shared_ptr<MethodAnalysis> Analysis::get_method(std::shared_ptr<ParentMethod> method)
        {
            std::string method_key;

            if (method->is_internal())
                method_key = std::dynamic_pointer_cast<EncodedMethod>(method)->full_name();
            else
                method_key = std::dynamic_pointer_cast<ExternalMethod>(method)->full_name();

            if (methods.find(method_key) != methods.end())
                return methods[method_key];
            return nullptr;
        }

        MethodID *Analysis::get_method_by_name(std::string class_name, std::string method_name, std::string method_descriptor)
        {
            auto m_a = get_method_analysis_by_name(class_name, method_name, method_descriptor);

            if (m_a && (!m_a->external()))
                return std::dynamic_pointer_cast<MethodID>(m_a->get_method()).get();

            return nullptr;
        }

        std::shared_ptr<MethodAnalysis> Analysis::get_method_analysis_by_name(std::string class_name, std::string method_name, std::string method_descriptor)
        {
            std::tuple<std::string, std::string, std::string> m_hash = {class_name, method_name, method_descriptor};

            if (method_hashes.find(m_hash) == method_hashes.end())
                return nullptr;
            return method_hashes[m_hash];
        }

        std::vector<std::shared_ptr<MethodAnalysis>> Analysis::get_methods()
        {
            std::vector<std::shared_ptr<MethodAnalysis>> methods;

            for (auto it = method_hashes.begin(); it != method_hashes.end(); it++)
                methods.push_back(it->second);

            return methods;
        }

        std::shared_ptr<FieldAnalysis> Analysis::get_field_analysis(std::shared_ptr<EncodedField> field)
        {
            auto class_analysis = get_class_analysis(reinterpret_cast<Class *>(field->get_field()->get_class_idx())->get_name());

            if (class_analysis)
                return class_analysis->get_field_analysis(field);

            return nullptr;
        }

        std::vector<std::shared_ptr<FieldAnalysis>> Analysis::get_fields()
        {
            std::vector<std::shared_ptr<FieldAnalysis>> fields;

            for (auto it = classes.begin(); it != classes.end(); it++)
            {
                auto aux = it->second->get_fields();

                fields.insert(std::end(fields), std::begin(aux), std::end(aux));
            }

            return fields;
        }

        std::vector<std::shared_ptr<StringAnalysis>> Analysis::get_strings()
        {
            std::vector<std::shared_ptr<StringAnalysis>> str_vector;

            for (auto it = strings.begin(); it != strings.end(); it++)
            {
                str_vector.push_back(it->second);
            }

            return str_vector;
        }

        std::vector<std::shared_ptr<ClassAnalysis>> Analysis::find_classes(std::string name = ".*", bool no_external = false)
        {
            std::vector<std::shared_ptr<ClassAnalysis>> classes_vector;
            std::regex class_name_regex(name);

            for (auto it = classes.begin(); it != classes.end(); it++)
            {
                if (no_external && (it->second->is_class_external()))
                    continue;
                if (std::regex_search(it->second->name(), class_name_regex))
                    classes_vector.push_back(it->second);
            }

            return classes_vector;
        }

        std::vector<std::shared_ptr<MethodAnalysis>> Analysis::find_methods(std::string class_name = ".*",
                                                                            std::string method_name = ".*",
                                                                            std::string descriptor = ".*",
                                                                            std::string accessflags = ".*",
                                                                            bool no_external = false)
        {
            std::vector<std::shared_ptr<MethodAnalysis>> methods_vector;

            std::regex class_name_regex(class_name),
                method_name_regex(method_name),
                descriptor_regex(descriptor),
                accessflags_regex(accessflags);

            for (auto c = classes.begin(); c != classes.end(); c++)
            {
                if (std::regex_search(c->second->name(), class_name_regex))
                {
                    auto methods = c->second->get_methods();

                    for (auto m = methods.begin(); m != methods.end(); m++)
                    {
                        if (no_external && (*m)->external())
                            continue;

                        if (std::regex_search((*m)->name(), method_name_regex) &&
                            std::regex_search((*m)->descriptor(), descriptor_regex) &&
                            std::regex_search((*m)->access(), accessflags_regex))
                            methods_vector.push_back(*m);
                    }
                }
            }

            return methods_vector;
        }

        std::vector<std::shared_ptr<StringAnalysis>> Analysis::find_strings(std::string string = ".*")
        {
            std::vector<std::shared_ptr<StringAnalysis>> strings_list;
            std::regex str_reg(string);

            for (auto it = strings.begin(); it != strings.end(); it++)
            {
                if (std::regex_search(it->first, str_reg))
                    strings_list.push_back(it->second);
            }

            return strings_list;
        }

        std::vector<std::shared_ptr<FieldAnalysis>> Analysis::find_fields(std::string class_name = ".*",
                                                                          std::string field_name = ".*",
                                                                          std::string field_type = ".*",
                                                                          std::string accessflags = ".*")
        {
            std::regex class_name_regex(class_name),
                field_name_regex(field_name),
                field_type_regex(field_type),
                accessflags_regex(accessflags);

            std::vector<std::shared_ptr<FieldAnalysis>> fields_list;

            for (auto c = classes.begin(); c != classes.end(); c++)
            {
                if (std::regex_search(c->second->name(), class_name_regex))
                {
                    auto fields = c->second->get_fields();
                    for (auto f = fields.begin(); f != fields.end(); f++)
                    {
                        if (std::regex_search((*f)->name(), field_name_regex) &&
                            std::regex_search((*f)->get_field()->get_field()->get_type_idx()->get_raw(), field_type_regex) &&
                            std::regex_search(dalvik_opcodes->get_access_flags_string((*f)->get_field()->get_access_flags()), accessflags_regex))
                        {
                            fields_list.push_back(*f);
                        }
                    }
                }
            }

            return fields_list;
        }

        void Analysis::_create_xref(std::shared_ptr<KUNAI::DEX::ClassDef> current_class)
        {
            auto current_class_name = current_class->get_class_idx()->get_name();

            auto class_data_item = current_class->get_class_data();

            // add the methods
            auto current_methods = class_data_item->get_methods();
            for (size_t j = 0; j < current_methods.size(); j++)
            {
                auto current_method = current_methods[j];
                auto current_method_analysis = methods[current_method->full_name()];
                auto current_class = classes[current_class_name];

                // std::cout << "Analyzing the method instructions from: " << current_class_name << "->" << current_method_analysis->name() << current_method_analysis->descriptor() << ";" << std::endl;

                // now we go parsing each instruction
                auto instructions = current_method_analysis->get_instructions();
                for (auto it = instructions.begin(); it != instructions.end(); it++)
                {
                    auto off = it->first;
                    auto instruction = it->second;

                    auto op_value = instruction->get_OP();

                    // first check for
                    // const-class
                    // new-instance
                    // instructions.
                    if ((op_value == DVMTypes::Opcode::OP_CONST_CLASS) || (op_value == DVMTypes::Opcode::OP_NEW_INSTANCE))
                    {
                        auto const_class_new_instance = std::dynamic_pointer_cast<Instruction21c>(instruction);

                        // we need to check that we get a TYPE from CONST_CLASS
                        // or from NEW_INSTANCE, any other Kind (FIELD, PROTO, etc)
                        // it's not good for us
                        if ((const_class_new_instance->get_kind() != DVMTypes::Kind::TYPE) ||
                            (const_class_new_instance->get_source_typeid()->get_type() != Type::CLASS))
                            continue;

                        auto type_info = reinterpret_cast<Class *>(const_class_new_instance->get_source_typeid())->get_name();

                        // we do not going to analyze our current class name
                        if (type_info == current_class_name)
                            continue;

                        // if the name of the class is not already in classes
                        // probably is an external class, add it to classes.
                        if (classes.find(type_info) == classes.end())
                            classes[type_info] = std::make_shared<ClassAnalysis>(std::make_shared<ExternalClass>(type_info));

                        auto oth_cls = classes[type_info];

                        /**
                         * FIXME: xref_to does not work here! current_method is wrong, as it is not the target!
                         * In this case that means, that current_method calls the class oth_class.
                         * Hence, on xref_to the method info is the calling method not the called one,
                         * as there is no called method!
                         * With the _new_instance and _const_class can this be deprecated?
                         * Removing these does not impact tests
                         */
                        current_class->add_xref_to(static_cast<DVMTypes::REF_TYPE>(op_value), oth_cls, current_method_analysis, off);
                        oth_cls->add_xref_from(static_cast<DVMTypes::REF_TYPE>(op_value), current_class, current_method_analysis, off);

                        if (op_value == DVMTypes::Opcode::OP_CONST_CLASS)
                        {
                            current_method_analysis->add_xref_const_class(oth_cls, off);
                            oth_cls->add_xref_const_class(current_method_analysis, off);
                        }
                        else if (op_value == DVMTypes::Opcode::OP_NEW_INSTANCE)
                        {
                            current_method_analysis->add_xref_new_instance(oth_cls, off);
                            oth_cls->add_xref_new_instance(current_method_analysis, off);
                        }
                    }

                    // check for others
                    // invoke-*
                    // invoke-xxx/range
                    else if ((DVMTypes::Opcode::OP_INVOKE_VIRTUAL <= op_value) && (op_value <= DVMTypes::Opcode::OP_INVOKE_INTERFACE))
                    {
                        auto invoke_ = std::dynamic_pointer_cast<Instruction35c>(instruction);

                        // get that kind is method
                        if (invoke_->get_kind() != DVMTypes::Kind::METH)
                            continue;

                        auto class_info = reinterpret_cast<Class *>(invoke_->get_operands_kind_method()->get_method_class())->get_name();
                        auto method_info = *invoke_->get_operands_kind_method()->get_method_name();
                        auto proto_info = invoke_->get_operands_kind_method()->get_method_prototype()->get_proto_str();

                        auto oth_meth = _resolve_method(class_info, method_info, proto_info);
                        auto oth_cls = classes[class_info];

                        current_class->add_method_xref_to(current_method_analysis, oth_cls, oth_meth, off);
                        oth_cls->add_method_xref_from(oth_meth, current_class, current_method_analysis, off);

                        current_class->add_xref_to(static_cast<DVMTypes::REF_TYPE>(op_value), oth_cls, oth_meth, off);
                        oth_cls->add_xref_from(static_cast<DVMTypes::REF_TYPE>(op_value), current_class, current_method_analysis, off);
                    }
                    else if ((DVMTypes::Opcode::OP_INVOKE_VIRTUAL_RANGE <= op_value) && (op_value <= DVMTypes::Opcode::OP_INVOKE_INTERFACE_RANGE))
                    {
                        auto invoke_xxx_range = std::dynamic_pointer_cast<Instruction3rc>(instruction);

                        // get that kind is type, and it's also a class
                        if (invoke_xxx_range->get_kind() != DVMTypes::Kind::METH)
                            continue;

                        auto class_info = reinterpret_cast<Class *>(invoke_xxx_range->get_operands_method()->get_method_class())->get_name();
                        auto method_info = *invoke_xxx_range->get_operands_method()->get_method_name();
                        auto proto_info = invoke_xxx_range->get_operands_method()->get_method_prototype()->get_proto_str();

                        auto oth_meth = _resolve_method(class_info, method_info, proto_info);
                        auto oth_cls = classes[class_info];

                        current_class->add_method_xref_to(current_method_analysis, oth_cls, oth_meth, off);
                        oth_cls->add_method_xref_from(oth_meth, current_class, current_method_analysis, off);

                        current_class->add_xref_to(static_cast<DVMTypes::REF_TYPE>(op_value), oth_cls, oth_meth, off);
                        oth_cls->add_xref_from(static_cast<DVMTypes::REF_TYPE>(op_value), current_class, current_method_analysis, off);
                    }

                    // check for string usage:
                    // const-string and const-string/jumbo
                    else if (op_value == DVMTypes::Opcode::OP_CONST_STRING)
                    {
                        auto const_string = std::dynamic_pointer_cast<Instruction21c>(instruction);

                        // we want strings, only strings and much strings
                        if (const_string->get_kind() != DVMTypes::Kind::STRING)
                            continue;

                        auto string_value = const_string->get_source_str();

                        if (strings.find(*string_value) == strings.end())
                            strings[*string_value] = std::make_shared<StringAnalysis>(string_value);

                        strings[*string_value]->add_xref_from(current_class, current_method_analysis, off);
                    }

                    // check now for field usage, we will first
                    // analyze those from OP_IGET to OP_IPUT_SHORT
                    // then those from OP_SGET to OP_SPUT_SHORT
                    else if ((DVMTypes::Opcode::OP_IGET <= op_value) && (op_value <= DVMTypes::Opcode::OP_IPUT_SHORT))
                    {
                        auto op_i = std::dynamic_pointer_cast<Instruction22c>(instruction);

                        if ((op_i->get_kind() != DVMTypes::Kind::FIELD) || (op_i->get_third_operand_kind() != DVMTypes::Kind::FIELD))
                            continue;

                        auto field_item = dalvik_opcodes->get_dalvik_encoded_field_by_fieldid(op_i->get_third_operand_FieldId());

                        if (dalvik_opcodes->get_instruction_operation(op_value) == DVMTypes::Operation::FIELD_READ_DVM_OPCODE)
                        {
                            classes[current_class_name]->add_field_xref_read(current_method_analysis, current_class, field_item, off);

                            // necessary to give a field analysis to the add_xref_read method
                            // we can get the created by the add_field_xref_read.
                            auto field_analysis = classes[current_class_name]->get_field_analysis(field_item);

                            current_method_analysis->add_xref_read(current_class, field_analysis, off);
                        }
                        else if (dalvik_opcodes->get_instruction_operation(op_value) == DVMTypes::Operation::FIELD_WRITE_DVM_OPCODE)
                        {
                            classes[current_class_name]->add_field_xref_write(current_method_analysis, current_class, field_item, off);

                            // same as before
                            auto field_analysis = classes[current_class_name]->get_field_analysis(field_item);

                            current_method_analysis->add_xref_write(current_class, field_analysis, off);
                        }
                    }
                    else if ((DVMTypes::Opcode::OP_SGET <= op_value) && (op_value <= DVMTypes::Opcode::OP_SPUT_SHORT))
                    {
                        auto op_s = std::dynamic_pointer_cast<Instruction21c>(instruction);

                        if ((op_s->get_kind() != DVMTypes::Kind::FIELD) || (op_s->get_source_kind() != DVMTypes::Kind::FIELD))
                            continue;

                        auto field_item = dalvik_opcodes->get_dalvik_encoded_field_by_fieldid(op_s->get_source_static_field());

                        if (dalvik_opcodes->get_instruction_operation(op_value) == DVMTypes::Operation::FIELD_READ_DVM_OPCODE)
                        {
                            classes[current_class_name]->add_field_xref_read(current_method_analysis, current_class, field_item, off);

                            // necessary to give a field analysis to the add_xref_read method
                            // we can get the created by the add_field_xref_read.
                            auto field_analysis = classes[current_class_name]->get_field_analysis(field_item);

                            current_method_analysis->add_xref_read(current_class, field_analysis, off);
                        }
                        else if (dalvik_opcodes->get_instruction_operation(op_value) == DVMTypes::Operation::FIELD_WRITE_DVM_OPCODE)
                        {
                            classes[current_class_name]->add_field_xref_write(current_method_analysis, current_class, field_item, off);

                            // same as before
                            auto field_analysis = classes[current_class_name]->get_field_analysis(field_item);

                            current_method_analysis->add_xref_write(current_class, field_analysis, off);
                        }
                    }
                }
            }
        }

        std::shared_ptr<MethodAnalysis> Analysis::_resolve_method(std::string class_name, std::string method_name, std::string method_descriptor)
        {
            std::tuple<std::string, std::string, std::string> m_hash = {class_name, method_name, method_descriptor};
            std::map<std::uint64_t, std::shared_ptr<Instruction>> empty_instructions;

            if (method_hashes.find(m_hash) == method_hashes.end())
            {
                // create if necessary create a class
                if (classes.find(class_name) == classes.end())
                    // add as external class
                    classes[class_name] = std::make_shared<ClassAnalysis>(std::make_shared<ExternalClass>(class_name));

                auto meth = std::make_shared<ExternalMethod>(class_name, method_name, method_descriptor);
                auto meth_analysis = std::make_shared<MethodAnalysis>(std::static_pointer_cast<ParentMethod>(meth), dalvik_opcodes, empty_instructions);

                // add to all the collections we have
                method_hashes[m_hash] = meth_analysis;
                classes[class_name]->add_method(meth_analysis);
                methods[meth->full_name()] = meth_analysis;
            }

            return method_hashes[m_hash];
        }

    } // namespace DEX
} // namespace KUNAI
