//--------------------------------------------------------------------*- C++ -*-
// Kunai-static-analyzer: library for doing analysis of dalvik files
// @author Farenain <kunai.static.analysis@gmail.com>
// @author Ernesto Java <javaernesto@gmail.com>
//
// @file dex_analysis.cpp

#include "Kunai/DEX/analysis/dex_analysis.hpp"
#include "Kunai/Utils/logger.hpp"

using namespace KUNAI::DEX;

void Analysis::add(Parser *parser)
{
    auto logger = LOGGER::logger();

    parsers.push_back(parser);

    auto &class_dex = parser->get_classes();

    logger->debug("Addind to the analysis {} number of classes", class_dex.get_number_of_classes());

    for (auto & class_def_item : class_dex.get_classdefs())
    {
        /// save the class with the name
        auto name = class_def_item->get_class_idx()->get_name();
        classes[name] = std::make_unique<ClassAnalysis>(class_def_item.get());
        auto &new_class = classes[name];

        // get the class data item to retrieve the methods
        auto &class_data_item = class_def_item->get_class_data_item();

        logger->debug("Adding to the class {} direct and {} virtual methods", class_data_item.get_number_of_direct_methods(), class_data_item.get_number_of_static_fields());

        for (auto encoded_method : class_data_item.get_methods())
        {
            auto method_id = encoded_method->getMethodID();
            /// now create a method analysis
            auto method_name = method_id->pretty_method();
            methods[method_name] = std::make_unique<MethodAnalysis>(encoded_method, disassembler->get_dex_instructions()[encoded_method]);
            auto new_method = methods[method_name].get();

            new_class->add_method(new_method);
            // maybe not needed method_hashes
            auto hash = name + method_id->get_name() + method_id->get_proto()->get_shorty_idx();
            method_hashes[hash] = new_method;
        }
    }

    logger->info("Analysis: correctly added parser to analysis object");
}

void Analysis::create_xrefs()
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

    logger->debug("create_xref(): creating xrefs for {} dex files", parsers.size());

    for (auto parser : parsers)
    {
        static size_t i = 0;
        logger->debug("Analyzing {} parser", i++);

        auto &class_dex = parser->get_classes();
        auto &class_def_items = class_dex.get_classdefs();

        logger->debug("Number of classes to analyze: {}", class_dex.get_number_of_classes());

        for (auto &class_def_item : class_def_items)
        {
            static size_t j = 0;
            logger->debug("Analyzing class number {}", j++);

            _create_xrefs(class_def_item.get());
        }
    }

    logger->info("Cross-references correctly created");
}

void Analysis::_create_xrefs(ClassDef *current_class)
{
    auto logger = LOGGER::logger();

    /// take the name of the analyzed class
    auto &current_class_name = current_class->get_class_idx()->get_name();
    auto &class_data_item = current_class->get_class_data_item();

    auto class_analysis_working_on = classes[current_class_name].get();

    /// get all the methods
    auto &current_methods = class_data_item.get_methods();

    for (auto &method : current_methods)
    {
        // Obtain the MethodAnalysis*
        auto current_method_analysis = methods[method->getMethodID()->pretty_method()].get();

        for (auto &instr : current_method_analysis->get_instructions())
        {
            auto off = instr->get_address();
            auto instruction = instr.get();
            auto op_value = instr->get_instruction_opcode();

            // check for: `const-class` and `new-instance` instructions
            if (op_value == TYPES::opcodes::OP_CONST_CLASS ||
                op_value == TYPES::opcodes::OP_NEW_INSTANCE)
            {
                auto const_class_new_instance = reinterpret_cast<Instruction21c *>(instruction);

                // check we get a TYPE from CONST_CLASS
                // or from NEW_INSTANCE, any other Kind (FIELD, PROTO, etc)
                // it is not valid in this case
                if (const_class_new_instance->get_kind() != TYPES::Kind::TYPE ||
                    const_class_new_instance->get_source_dvmtype()->get_type() != DVMType::CLASS)
                    continue;

                auto cls_name = const_class_new_instance->get_source_dvmclass()->get_name();

                // avoid analyzing our own class name
                if (cls_name == current_class_name)
                    continue;

                // if the name of the class is not already in the classes,
                // probably we are treating with an external class
                if (classes.find(cls_name) == classes.end())
                {
                    external_classes[cls_name] = std::make_unique<ExternalClass>(cls_name);
                    classes[cls_name] = std::make_unique<ClassAnalysis>(external_classes[cls_name].get());
                }

                auto oth_cls = classes[cls_name].get();

                /// add the cross references
                class_analysis_working_on->add_xref_to(static_cast<TYPES::REF_TYPE>(op_value),
                                                       oth_cls, current_method_analysis, off);
                oth_cls->add_xref_from(static_cast<TYPES::REF_TYPE>(op_value),
                                       class_analysis_working_on, current_method_analysis, off);

                /// check if is a const-class
                if (op_value == TYPES::opcodes::OP_CONST_CLASS)
                {
                    current_method_analysis->add_xrefconstclass(oth_cls, off);
                    oth_cls->add_xref_const_class(current_method_analysis, off);
                }
                /// check if it is a new instance
                else if (op_value == TYPES::opcodes::OP_NEW_INSTANCE)
                {
                    current_method_analysis->add_xrefnewinstance(oth_cls, off);
                    oth_cls->add_xref_new_instance(current_method_analysis, off);
                }
            }

            /// check for instructions like: invoke-*
            else if (TYPES::opcodes::OP_INVOKE_VIRTUAL <= op_value &&
                     op_value <= TYPES::opcodes::OP_INVOKE_INTERFACE)
            {
                auto invoke_ = reinterpret_cast<Instruction35c *>(instruction);

                // get the invoke of method
                if (invoke_->get_kind() != TYPES::Kind::METH)
                    continue;

                /// check that called method comes from a class
                /// (not from other type like an Array)
                if (invoke_->get_method()->get_class()->get_type() != DVMType::CLASS)
                {
                    logger->warn("Found a call to a method from non class (type found {})",
                                 invoke_->get_method()->get_class()->print_type());
                    continue;
                }

                auto method_called = invoke_->get_method();

                auto &cls_name = reinterpret_cast<DVMClass *>(method_called->get_class())->get_name();
                auto &method_name = method_called->get_name();
                auto &proto = method_called->get_proto()->get_shorty_idx();

                /// information of method and class called
                auto oth_meth = _resolve_method(cls_name, method_name, proto);
                auto oth_cls = classes[cls_name].get();

                class_analysis_working_on->add_method_xref_to(current_method_analysis, oth_cls, oth_meth, off);
                oth_cls->add_method_xref_from(oth_meth, class_analysis_working_on, current_method_analysis, off);

                class_analysis_working_on->add_xref_to(static_cast<TYPES::REF_TYPE>(op_value), oth_cls, oth_meth, off);
                oth_cls->add_xref_from(static_cast<TYPES::REF_TYPE>(op_value), class_analysis_working_on, current_method_analysis, off);
            }
            /// check for instructions like: invoke-xxx/range
            else if (TYPES::opcodes::OP_INVOKE_VIRTUAL_RANGE <= op_value &&
                     op_value <= TYPES::opcodes::OP_INVOKE_INTERFACE_RANGE)
            {
                auto invoke_xxx_range = reinterpret_cast<Instruction3rc *>(instruction);

                // check if we are calling something different to a method
                if (invoke_xxx_range->get_kind() != TYPES::Kind::METH)
                    continue;

                auto method_id = invoke_xxx_range->get_operand_method();

                auto &cls_name = reinterpret_cast<DVMClass *>(method_id->get_class())->get_name();
                auto &method_name = method_id->get_name();
                auto &proto = method_id->get_proto()->get_shorty_idx();

                auto oth_meth = _resolve_method(cls_name, method_name, proto);
                auto oth_cls = classes[cls_name].get();

                class_analysis_working_on->add_method_xref_to(
                    current_method_analysis, oth_cls, oth_meth, off);
                oth_cls->add_method_xref_from(
                    oth_meth, class_analysis_working_on, current_method_analysis, off);

                class_analysis_working_on->add_xref_to(
                    static_cast<TYPES::REF_TYPE>(op_value), oth_cls, oth_meth, off);
                oth_cls->add_xref_from(
                    static_cast<TYPES::REF_TYPE>(op_value), class_analysis_working_on, current_method_analysis, off);
            }

            // now check for string usage: const-string
            else if (op_value == TYPES::opcodes::OP_CONST_STRING)
            {
                auto const_string = reinterpret_cast<Instruction21c *>(instruction);

                if (!const_string->is_source_string())
                    continue;

                auto &string_value = const_string->get_source_str();

                if (strings.find(string_value) == strings.end())
                    strings[string_value] = std::make_unique<StringAnalysis>(string_value);

                strings[string_value]->add_xreffrom(class_analysis_working_on, current_method_analysis, off);
            }

            /// check now for field usage, we first
            /// analyze those from OP_IGET to OP_IPUT_SHORT
            /// then those from OP_SGET to OP_SPUT_SHORT
            else if (TYPES::opcodes::OP_IGET <= op_value &&
                     op_value <= TYPES::opcodes::OP_IPUT_SHORT)
            {
                auto op_i = reinterpret_cast<Instruction22c *>(instruction);
                auto checked_field = op_i->get_checked_field();

                if (op_i->get_kind() != TYPES::Kind::FIELD ||
                    checked_field == nullptr)
                    continue;

                auto operation = DalvikOpcodes::get_instruction_operation(op_i->get_instruction_opcode());

                /// is a read operation?
                if (operation == TYPES::Operation::FIELD_READ_DVM_OPCODE)
                {
                    // retrieve the encoded field from the FieldID
                    auto field_item = checked_field->get_encoded_field();

                    classes[current_class_name]->add_field_xref_read(
                        current_method_analysis, class_analysis_working_on, field_item, off);

                    // necessary to give a field analysis to the add_xref_read method
                    // we can get the created by the add_field_xref_read.
                    auto field_analysis = classes[current_class_name]->get_field_analysis(field_item);
                    current_method_analysis->add_xrefread(class_analysis_working_on, field_analysis, off);
                }
                /// is a write operation?
                else if (operation == TYPES::Operation::FIELD_WRITE_DVM_OPCODE)
                {
                    // retrieve the encoded field from the FieldID
                    auto field_item = checked_field->get_encoded_field();

                    classes[current_class_name]->add_field_xref_write(
                        current_method_analysis, class_analysis_working_on, field_item, off);

                    // same as before
                    auto field_analysis = classes[current_class_name]->get_field_analysis(field_item);

                    current_method_analysis->add_xrefwrite(class_analysis_working_on, field_analysis, off);
                }
            }
            /// now time to check OP_SGET to OP_SPUT_SHORT
            else if (TYPES::opcodes::OP_SGET <= op_value &&
                     op_value <= TYPES::opcodes::OP_SPUT_SHORT)
            {
                auto op_s = reinterpret_cast<Instruction21c *>(instruction);
                auto checked_field = op_s->get_source_field();

                /// if the instruction is not a FIELD Kind instruction
                /// if there are not checked field, or the EncodedField
                /// of the Field is nullptr (an external field), leave!
                if (op_s->get_kind() != TYPES::Kind::FIELD ||
                    checked_field == nullptr ||
                    checked_field->get_encoded_field() == nullptr)
                    continue;

                auto operation = DalvikOpcodes::get_instruction_operation(op_s->get_instruction_opcode());

                // is read operation
                if (operation == TYPES::Operation::FIELD_READ_DVM_OPCODE)
                {
                    auto field_item = checked_field->get_encoded_field();

                    classes[current_class_name]->add_field_xref_read(
                        current_method_analysis, class_analysis_working_on, field_item, off);

                    /// necessary to give a field analysis to the add_xref_read method
                    /// we can get the created by the add_field_xref_read.
                    auto field_analysis = classes[current_class_name]->get_field_analysis(field_item);
                    current_method_analysis->add_xrefread(class_analysis_working_on, field_analysis, off);
                }
                else if (operation == TYPES::Operation::FIELD_WRITE_DVM_OPCODE)
                {
                    auto field_item = checked_field->get_encoded_field();

                    classes[current_class_name]->add_field_xref_write(
                        current_method_analysis, class_analysis_working_on, field_item, off);

                    /// same as before
                    auto field_analysis = classes[current_class_name]->get_field_analysis(field_item);
                    current_method_analysis->add_xrefwrite(class_analysis_working_on, field_analysis, off);
                }
            }
        }
    }
}

MethodAnalysis *Analysis::_resolve_method(std::string &class_name,
                                          std::string &method_name, std::string &method_descriptor)
{
    std::string m_hash = class_name+method_name+method_descriptor;
    std::vector<std::unique_ptr<Instruction>> empty_instructions;

    auto it = method_hashes.find(m_hash);

    if (it != method_hashes.end())
        return it->second;
    
    // create if necessary a class
    if (classes.find(class_name) == classes.end())
    {
        external_classes[class_name] = std::make_unique<ExternalClass>(class_name);
        // add external class
        classes[class_name] = std::make_unique<ClassAnalysis>(external_classes[class_name].get());
    }

    external_methods[m_hash] = std::make_unique<ExternalMethod>(class_name, method_name, method_descriptor);
    auto meth_analysis = std::make_unique<MethodAnalysis>(external_methods[m_hash].get(), empty_instructions);
    auto meth_analysis_p_ = meth_analysis.get();
    // add to all the collections we have
    method_hashes[m_hash] = meth_analysis_p_;    
    classes[class_name]->add_method(meth_analysis_p_);
    methods[external_methods[m_hash]->pretty_method_name()] = std::move(meth_analysis);

    return method_hashes[m_hash];
}