#include "KUNAI/DEX/Analysis/dex_analysis.hpp"

namespace KUNAI
{
    namespace DEX
    {

        Analysis::Analysis(DexParser* dex_parser, DalvikOpcodes* dalvik_opcodes, instruction_map_t instructions, bool create_xrefs) : created_xrefs(!create_xrefs),
                                                                                                                                                   dalvik_opcodes(dalvik_opcodes),
                                                                                                                                                   instructions(instructions)
        {
            if (dex_parser)
                this->add(dex_parser);
        }

        void Analysis::add(DexParser* dex_parser)
        {
            auto logger = LOGGER::logger();

            // first of all add it to vector
            this->dex_parsers.push_back(dex_parser);

            auto class_dex = dex_parser->get_classes();

#ifdef DEBUG
            logger->debug("Adding to the analysis {} number of classes", class_dex->get_number_of_classes());
#endif

            for (auto & class_def_item : class_dex->get_classes())
            {
                if (class_def_item == nullptr)
                    continue;

                classes[class_def_item->get_class_idx()->get_name()] = std::make_unique<ClassAnalysis>(class_def_item.get());
                auto & new_class = classes[class_def_item->get_class_idx()->get_name()];

                // get class data item
                auto class_data_item = class_def_item->get_class_data();

                if (class_data_item == nullptr)
                    continue;

#ifdef DEBUG
                logger->debug("Adding to the class {} direct and virtual methods", class_data_item->get_number_of_direct_methods());
#endif
                // add the direct & virtual methods
                for (auto encoded_method : class_data_item->get_methods())
                {
                    std::map<std::uint64_t, Instruction*> method_instructions;

                    for (auto & instr : instructions[{class_def_item.get(), encoded_method}])
                        method_instructions[instr.first] = instr.second.get();

                    methods[encoded_method->full_name()] = std::make_unique<MethodAnalysis>(encoded_method, dalvik_opcodes, method_instructions);
                    auto & new_method = methods[encoded_method->full_name()];

                    new_class->add_method(new_method.get());

                    method_hashes[class_def_item->get_class_idx()->get_name() + *encoded_method->get_method()->get_method_name() + encoded_method->get_method()->get_method_prototype()->get_proto_str()] = new_method.get();
                }
            }

            logger->info("Analysis: correctly added parser to analysis object");
        }

        void Analysis::create_xref()
        {
            auto logger = LOGGER::logger();

            if (created_xrefs)
            {
                logger->info("Requested create_xref() method more than once.");
                logger->info("create_xref() will not work again, function will exit right now.");
                logger->info("Please if you want to analze various dex parsers, add all of them first, then call this function.");

                return;
            }

            created_xrefs = true;

#ifdef DEBUG
            logger->debug("create_xref(): creating xrefs for {} dex files", dex_parsers.size());
#endif

            for (auto& dex_parser : dex_parsers)
            {
#ifdef DEBUG
                static size_t i = 0;
                logger->debug("Analyzing {} parser from {}", i++, dex_parsers.size());
#endif

                auto class_dex = dex_parser->get_classes();
                auto class_def_items = class_dex->get_classes();

                for (auto& class_def_item : class_def_items)
                {
#ifdef DEBUG
                    static size_t j = 0;
                    logger->debug("Analyzing {}({}) class from {}", class_def_item->get_class_idx()->get_raw(), j++, class_dex->get_number_of_classes());
#endif

                    _create_xref(class_def_item.get());
                }
            }

            logger->info("cross-references correctly created");
        }

        bool Analysis::is_class_present(std::string& class_name)
        {
            return (classes.find(class_name) == classes.end());
        }

        ClassAnalysis* Analysis::get_class_analysis(std::string& class_name)
        {
            if (classes.find(class_name) == classes.end())
                return nullptr;
            return classes[class_name].get();
        }

        std::vector<ClassAnalysis*> Analysis::get_classes()
        {
            std::vector<ClassAnalysis*> classes_vector;

            for (auto & class_ : classes)
            {
                classes_vector.push_back(class_.second.get());
            }

            return classes_vector;
        }

        std::vector<ClassAnalysis*> Analysis::get_external_classes()
        {
            std::vector<ClassAnalysis*> external_classes;

            for (auto & class_ : classes)
            {
                if (class_.second->is_class_external())
                    external_classes.push_back(class_.second.get());
            }

            return external_classes;
        }

        std::vector<ClassAnalysis*> Analysis::get_internal_classes()
        {
            std::vector<ClassAnalysis*> internal_classes;

            for (auto & class_ : classes)
            {
                if (!class_.second->is_class_external())
                    internal_classes.push_back(class_.second.get());
            }

            return internal_classes;
        }

        MethodAnalysis* Analysis::get_method(std::variant<EncodedMethod*, ExternalMethod*> method)
        {
            std::string method_key;

            if (method.index() == 0)
                method_key = std::get<EncodedMethod*>(method)->full_name();
            else
                method_key = std::get<ExternalMethod*>(method)->full_name();

            if (methods.find(method_key) != methods.end())
                return methods[method_key].get();
            return nullptr;
        }

        MethodID* Analysis::get_method_by_name(std::string& class_name, std::string& method_name, std::string& method_descriptor)
        {
            auto m_a = get_method_analysis_by_name(class_name, method_name, method_descriptor);

            if (m_a && (!m_a->external()))
                return std::get<EncodedMethod*>(m_a->get_method())->get_method();

            return nullptr;
        }

        MethodAnalysis* Analysis::get_method_analysis_by_name(std::string& class_name, std::string& method_name, std::string& method_descriptor)
        {
            std::string m_hash = class_name+method_name+method_descriptor;

            if (method_hashes.find(m_hash) == method_hashes.end())
                return nullptr;
            return method_hashes[m_hash];
        }

        std::vector<MethodAnalysis*> Analysis::get_methods()
        {
            std::vector<MethodAnalysis*> methods;

            for (auto hash : method_hashes)
                methods.push_back(hash.second);

            return methods;
        }

        FieldAnalysis* Analysis::get_field_analysis(EncodedField* field)
        {
            auto class_analysis = get_class_analysis(reinterpret_cast<Class*>(field->get_field()->get_class_idx())->get_name());

            if (class_analysis)
                return class_analysis->get_field_analysis(field);

            return nullptr;
        }

        std::vector<FieldAnalysis*> Analysis::get_fields()
        {
            std::vector<FieldAnalysis*> fields;

            for (auto & c : classes)
            {
                auto aux = c.second->get_fields();

                fields.insert(std::end(fields), std::begin(aux), std::end(aux));
            }

            return fields;
        }

        std::vector<StringAnalysis*> Analysis::get_strings()
        {
            std::vector<StringAnalysis*> str_vector;

            for (auto & s : strings)
            {
                str_vector.push_back(s.second.get());
            }

            return str_vector;
        }

        std::vector<ClassAnalysis*> Analysis::find_classes(std::string name = ".*", bool no_external = false)
        {
            std::vector<ClassAnalysis*> classes_vector;
            std::regex class_name_regex(name);

            for (auto & c : classes)
            {
                if (no_external && (c.second->is_class_external()))
                    continue;
                if (std::regex_search(c.second->name(), class_name_regex))
                    classes_vector.push_back(c.second.get());
            }

            return classes_vector;
        }

        std::vector<MethodAnalysis*> Analysis::find_methods(std::string class_name = ".*",
                                                                            std::string method_name = ".*",
                                                                            std::string descriptor = ".*",
                                                                            std::string accessflags = ".*",
                                                                            bool no_external = false)
        {
            std::vector<MethodAnalysis*> methods_vector;

            std::regex class_name_regex(class_name),
                method_name_regex(method_name),
                descriptor_regex(descriptor),
                accessflags_regex(accessflags);

            for (auto & c : classes)
            {
                if (std::regex_search(c.second->name(), class_name_regex))
                {
                    auto methods = c.second->get_methods();

                    for (auto m : methods)
                    {
                        if (no_external && m->external())
                            continue;

                        if (std::regex_search(m->name(), method_name_regex) &&
                            std::regex_search(m->descriptor(), descriptor_regex) &&
                            std::regex_search(m->access(), accessflags_regex))
                            methods_vector.push_back(m);
                    }
                }
            }

            return methods_vector;
        }

        std::vector<StringAnalysis*> Analysis::find_strings(std::string string = ".*")
        {
            std::vector<StringAnalysis*> strings_list;
            std::regex str_reg(string);

            for (auto it = strings.begin(); it != strings.end(); it++)
            {
                if (std::regex_search(it->first, str_reg))
                    strings_list.push_back(it->second.get());
            }

            return strings_list;
        }

        std::vector<FieldAnalysis*> Analysis::find_fields(std::string class_name = ".*",
                                                                          std::string field_name = ".*",
                                                                          std::string field_type = ".*",
                                                                          std::string accessflags = ".*")
        {
            std::regex class_name_regex(class_name),
                field_name_regex(field_name),
                field_type_regex(field_type),
                accessflags_regex(accessflags);

            std::vector<FieldAnalysis*> fields_list;

            for (auto & c : classes)
            {
                if (std::regex_search(c.second->name(), class_name_regex))
                {
                    auto fields = c.second->get_fields();
                    for (auto f : fields)
                    {
                        if (std::regex_search(f->name(), field_name_regex) &&
                            std::regex_search(f->get_field()->get_field()->get_type_idx()->get_raw(), field_type_regex) &&
                            std::regex_search(dalvik_opcodes->get_access_flags_string(f->get_field()->get_access_flags()), accessflags_regex))
                        {
                            fields_list.push_back(f);
                        }
                    }
                }
            }

            return fields_list;
        }

        void Analysis::_create_xref(KUNAI::DEX::ClassDef* current_class)
        {
            auto logger = LOGGER::logger();

            auto current_class_name = current_class->get_class_idx()->get_name();
            auto class_data_item = current_class->get_class_data();

            if (!class_data_item)
            {
#ifdef DEBUG
                logger->debug("class_data_item not present for class {}", current_class_name);
#endif

                return;
            }

            // add the methods
            auto& current_methods = class_data_item->get_methods();
            auto& class_working_on = classes[current_class_name];

            for (auto current_method : current_methods)
            {
                auto & current_method_analysis = methods[current_method->full_name()];

#ifdef DEBUG
                static size_t j = 0;
                logger->debug("Analyzing {}({}) method from {}", current_method_analysis->full_name().c_str(), j++, current_methods.size());
#endif

                // now we go parsing each instruction
                auto& instructions = current_method_analysis->get_instructions();

                for (auto& offset_instr : instructions)
                {
                    auto off = offset_instr.first;
                    auto instruction = offset_instr.second;

                    auto op_value = instruction->get_OP();

                    // first check for
                    // const-class
                    // new-instance
                    // instructions.
                    if ((op_value == DVMTypes::Opcode::OP_CONST_CLASS) || (op_value == DVMTypes::Opcode::OP_NEW_INSTANCE))
                    {
                        auto const_class_new_instance = reinterpret_cast<Instruction21c*>(instruction);

                        // we need to check that we get a TYPE from CONST_CLASS
                        // or from NEW_INSTANCE, any other Kind (FIELD, PROTO, etc)
                        // it's not good for us
                        if ((const_class_new_instance->get_kind() != DVMTypes::Kind::TYPE) ||
                            (const_class_new_instance->get_source_typeid()->get_type() != Type::CLASS))
                            continue;

                        auto type_info = reinterpret_cast<Class*>(const_class_new_instance->get_source_typeid())->get_name();

                        // we aren't going to analyze our current class name
                        if (type_info == current_class_name)
                            continue;

                        // if the name of the class is not already in classes
                        // probably is an external class, add it to classes.
                        if (classes.find(type_info) == classes.end())
                            classes[type_info] = std::make_unique<ClassAnalysis>(std::make_shared<ExternalClass>(type_info));

                        auto & oth_cls = classes[type_info];

                        /**
                         * FIXME: xref_to does not work here! current_method is wrong, as it is not the target!
                         * In this case that means, that current_method calls the class oth_class.
                         * Hence, on xref_to the method info is the calling method not the called one,
                         * as there is no called method!
                         * With the _new_instance and _const_class can this be deprecated?
                         * Removing these does not impact tests
                         */
                        class_working_on->add_xref_to(static_cast<DVMTypes::REF_TYPE>(op_value), oth_cls, current_method_analysis, off);
                        oth_cls->add_xref_from(static_cast<DVMTypes::REF_TYPE>(op_value), class_working_on, current_method_analysis, off);

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

                        if (invoke_->get_operands_kind_method()->get_method_class()->get_type() != DEX::Type::CLASS)
                        {
                            logger->warn("Found a call to a method from non class (type found: {})",
                                         invoke_->get_operands_kind_method()->get_method_class()->print_type());
                            continue;
                        }

                        auto class_info = std::dynamic_pointer_cast<Class>(invoke_->get_operands_kind_method()->get_method_class())->get_name();
                        auto method_info = *invoke_->get_operands_kind_method()->get_method_name();
                        auto proto_info = invoke_->get_operands_kind_method()->get_method_prototype()->get_proto_str();

                        auto oth_meth = _resolve_method(class_info, method_info, proto_info);
                        auto oth_cls = classes[class_info];

                        class_working_on->add_method_xref_to(current_method_analysis, oth_cls, oth_meth, off);
                        oth_cls->add_method_xref_from(oth_meth, class_working_on, current_method_analysis, off);

                        class_working_on->add_xref_to(static_cast<DVMTypes::REF_TYPE>(op_value), oth_cls, oth_meth, off);
                        oth_cls->add_xref_from(static_cast<DVMTypes::REF_TYPE>(op_value), class_working_on, current_method_analysis, off);
                    }
                    else if ((DVMTypes::Opcode::OP_INVOKE_VIRTUAL_RANGE <= op_value) && (op_value <= DVMTypes::Opcode::OP_INVOKE_INTERFACE_RANGE))
                    {
                        auto invoke_xxx_range = std::dynamic_pointer_cast<Instruction3rc>(instruction);

                        // get that kind is type, and it's also a class
                        if (invoke_xxx_range->get_kind() != DVMTypes::Kind::METH)
                            continue;

                        auto class_info = std::dynamic_pointer_cast<Class>(invoke_xxx_range->get_operands_method()->get_method_class())->get_name();
                        auto method_info = *invoke_xxx_range->get_operands_method()->get_method_name();
                        auto proto_info = invoke_xxx_range->get_operands_method()->get_method_prototype()->get_proto_str();

                        auto oth_meth = _resolve_method(class_info, method_info, proto_info);
                        auto oth_cls = classes[class_info];

                        class_working_on->add_method_xref_to(current_method_analysis, oth_cls, oth_meth, off);
                        oth_cls->add_method_xref_from(oth_meth, class_working_on, current_method_analysis, off);

                        class_working_on->add_xref_to(static_cast<DVMTypes::REF_TYPE>(op_value), oth_cls, oth_meth, off);
                        oth_cls->add_xref_from(static_cast<DVMTypes::REF_TYPE>(op_value), class_working_on, current_method_analysis, off);
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

                        strings[*string_value]->add_xref_from(class_working_on, current_method_analysis, off);
                    }

                    // check now for field usage, we will first
                    // analyze those from OP_IGET to OP_IPUT_SHORT
                    // then those from OP_SGET to OP_SPUT_SHORT
                    else if ((DVMTypes::Opcode::OP_IGET <= op_value) && (op_value <= DVMTypes::Opcode::OP_IPUT_SHORT))
                    {
                        auto op_i = std::dynamic_pointer_cast<Instruction22c>(instruction);

                        if ((op_i->get_kind() != DVMTypes::Kind::FIELD) || (op_i->get_third_operand_kind() != DVMTypes::Kind::FIELD))
                            continue;

                        if (dalvik_opcodes->get_instruction_operation(op_value) == DVMTypes::Operation::FIELD_READ_DVM_OPCODE)
                        {
                            auto field_item = dalvik_opcodes->get_dalvik_encoded_field_by_fieldid(op_i->get_third_operand_FieldId());

                            classes[current_class_name]->add_field_xref_read(current_method_analysis, class_working_on, field_item, off);

                            // necessary to give a field analysis to the add_xref_read method
                            // we can get the created by the add_field_xref_read.
                            auto field_analysis = classes[current_class_name]->get_field_analysis(field_item);

                            current_method_analysis->add_xref_read(class_working_on, field_analysis, off);
                        }
                        else if (dalvik_opcodes->get_instruction_operation(op_value) == DVMTypes::Operation::FIELD_WRITE_DVM_OPCODE)
                        {
                            auto field_item = dalvik_opcodes->get_dalvik_encoded_field_by_fieldid(op_i->get_third_operand_FieldId());

                            classes[current_class_name]->add_field_xref_write(current_method_analysis, class_working_on, field_item, off);

                            // same as before
                            auto field_analysis = classes[current_class_name]->get_field_analysis(field_item);

                            current_method_analysis->add_xref_write(class_working_on, field_analysis, off);
                        }
                    }
                    else if ((DVMTypes::Opcode::OP_SGET <= op_value) && (op_value <= DVMTypes::Opcode::OP_SPUT_SHORT))
                    {
                        auto op_s = std::dynamic_pointer_cast<Instruction21c>(instruction);

                        if ((op_s->get_kind() != DVMTypes::Kind::FIELD) || (op_s->get_source_kind() != DVMTypes::Kind::FIELD))
                            continue;

                        if (dalvik_opcodes->get_instruction_operation(op_value) == DVMTypes::Operation::FIELD_READ_DVM_OPCODE)
                        {
                            auto field_item = dalvik_opcodes->get_dalvik_encoded_field_by_fieldid(op_s->get_source_static_field());

                            classes[current_class_name]->add_field_xref_read(current_method_analysis, class_working_on, field_item, off);

                            // necessary to give a field analysis to the add_xref_read method
                            // we can get the created by the add_field_xref_read.
                            auto field_analysis = classes[current_class_name]->get_field_analysis(field_item);

                            current_method_analysis->add_xref_read(class_working_on, field_analysis, off);
                        }
                        else if (dalvik_opcodes->get_instruction_operation(op_value) == DVMTypes::Operation::FIELD_WRITE_DVM_OPCODE)
                        {
                            auto field_item = dalvik_opcodes->get_dalvik_encoded_field_by_fieldid(op_s->get_source_static_field());
                            
                            classes[current_class_name]->add_field_xref_write(current_method_analysis, class_working_on, field_item, off);

                            // same as before
                            auto field_analysis = classes[current_class_name]->get_field_analysis(field_item);

                            current_method_analysis->add_xref_write(class_working_on, field_analysis, off);
                        }
                    }
                }
            }
        }

        methodanalysis_t Analysis::_resolve_method(std::string class_name, std::string method_name, std::string method_descriptor)
        {
            std::string m_hash = class_name+method_name+method_descriptor;
            
            std::map<std::uint64_t, instruction_t> empty_instructions;

            if (method_hashes.find(m_hash) == method_hashes.end())
            {
                // create if necessary create a class
                if (classes.find(class_name) == classes.end())
                    // add as external class
                    classes[class_name] = std::make_shared<ClassAnalysis>(std::make_shared<ExternalClass>(class_name));

                auto meth = std::make_shared<ExternalMethod>(class_name, method_name, method_descriptor);
                auto meth_analysis = std::make_shared<MethodAnalysis>(meth, dalvik_opcodes, empty_instructions);

                // add to all the collections we have
                method_hashes[m_hash] = meth_analysis;
                classes[class_name]->add_method(meth_analysis);
                methods[meth->full_name()] = meth_analysis;
            }

            return method_hashes[m_hash];
        }

    } // namespace DEX
} // namespace KUNAI
