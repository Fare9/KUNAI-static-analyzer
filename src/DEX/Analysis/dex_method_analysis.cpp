#include "dex_analysis.hpp"

namespace KUNAI
{
    namespace DEX
    {
        /**
         * MethodAnalysis methods
         */

        MethodAnalysis::MethodAnalysis(std::shared_ptr<ParentMethod> method_encoded, std::shared_ptr<DalvikOpcodes> dalvik_opcodes, std::map<std::uint64_t, std::shared_ptr<Instruction>> instructions) : method_encoded(method_encoded),
                                                                                                                                                                                                          dalvik_opcodes(dalvik_opcodes),
                                                                                                                                                                                                          instructions(instructions)
        {
            this->is_external = method_encoded->is_external();
            this->exceptions = std::make_shared<Exception>();

            if (this->instructions.size() > 0)
                this->create_basic_block();
        }

        bool MethodAnalysis::is_android_api()
        {
            if (!is_external)
                return false;

            std::string class_name = this->class_name();

            for (const auto &known_api : known_apis)
            {
                if (class_name.find(known_api) == 0)
                    return true;
            }

            return false;
        }

        std::string MethodAnalysis::name()
        {
            std::string name;

            if (is_external)
            {
                std::shared_ptr<ExternalMethod> method = std::dynamic_pointer_cast<ExternalMethod>(method_encoded);
                name = method->get_name();
            }
            else
            {
                std::shared_ptr<EncodedMethod> method = std::dynamic_pointer_cast<EncodedMethod>(method_encoded);
                name = *method->get_method()->get_method_name();
            }

            return name;
        }

        std::string MethodAnalysis::descriptor()
        {
            std::string descriptor;

            if (is_external)
            {
                std::shared_ptr<ExternalMethod> method = std::dynamic_pointer_cast<ExternalMethod>(method_encoded);
                descriptor = method->get_descriptor();
            }
            else
            {
                std::shared_ptr<EncodedMethod> method = std::dynamic_pointer_cast<EncodedMethod>(method_encoded);
                descriptor = method->get_method()->get_method_prototype()->get_proto_str();
            }

            return descriptor;
        }

        std::string MethodAnalysis::access()
        {
            std::string access_flag;

            if (is_external)
            {
                std::shared_ptr<ExternalMethod> method = std::dynamic_pointer_cast<ExternalMethod>(method_encoded);
                access_flag = this->dalvik_opcodes->get_access_flags_string(method->get_access_flags());
            }
            else
            {
                std::shared_ptr<EncodedMethod> method = std::dynamic_pointer_cast<EncodedMethod>(method_encoded);
                access_flag = this->dalvik_opcodes->get_access_flags_string(method->get_access_flags());
            }

            return access_flag;
        }

        std::string MethodAnalysis::class_name()
        {
            std::string class_name;

            if (is_external)
            {
                std::shared_ptr<ExternalMethod> method = std::dynamic_pointer_cast<ExternalMethod>(method_encoded);
                class_name = method->get_class_name();
            }
            else
            {
                std::shared_ptr<EncodedMethod> method = std::dynamic_pointer_cast<EncodedMethod>(method_encoded);
                class_name = method->get_method()->get_method_class()->get_raw();
            }

            return class_name;
        }

        std::string MethodAnalysis::full_name()
        {
            std::string class_name = this->class_name();
            std::string descriptor = this->descriptor();
            std::string name = this->name();

            return class_name + " " + name + " " + descriptor;
        }

        void MethodAnalysis::add_xref_read(std::shared_ptr<ClassAnalysis> class_object, std::shared_ptr<FieldAnalysis> field_object, std::uint64_t offset)
        {
            xrefread.push_back({class_object, field_object, offset});
        }

        void MethodAnalysis::add_xref_write(std::shared_ptr<ClassAnalysis> class_object, std::shared_ptr<FieldAnalysis> field_object, std::uint64_t offset)
        {
            xrefwrite.push_back({class_object, field_object, offset});
        }

        void MethodAnalysis::add_xref_to(std::shared_ptr<ClassAnalysis> class_object, std::shared_ptr<MethodAnalysis> method_object, std::uint64_t offset)
        {
            xrefto.push_back({class_object, method_object, offset});
        }

        void MethodAnalysis::add_xref_from(std::shared_ptr<ClassAnalysis> class_object, std::shared_ptr<MethodAnalysis> method_object, std::uint64_t offset)
        {
            xreffrom.push_back({class_object, method_object, offset});
        }

        void MethodAnalysis::add_xref_new_instance(std::shared_ptr<ClassAnalysis> class_object, std::uint64_t offset)
        {
            xrefnewinstance.push_back({class_object, offset});
        }

        void MethodAnalysis::add_xref_const_class(std::shared_ptr<ClassAnalysis> class_object, std::uint64_t offset)
        {
            xrefconstclass.push_back({class_object, offset});
        }

        void MethodAnalysis::create_basic_block()
        {
            auto logger = LOGGER::logger();

            basic_blocks = std::make_shared<BasicBlocks>();
            std::shared_ptr<DVMBasicBlock> current_basic = std::make_shared<DVMBasicBlock>(0, dalvik_opcodes, basic_blocks, std::dynamic_pointer_cast<EncodedMethod>(method_encoded), instructions);

            // push the first basic block
            basic_blocks->push_basic_block(current_basic);

            std::vector<std::int64_t> l;
            std::map<std::uint64_t, std::vector<std::int64_t>> h;

            logger->debug("create_basic_block: creating basic blocks for method {}.", std::dynamic_pointer_cast<EncodedMethod>(method_encoded)->full_name());

            for (auto const &instruction : instructions)
            {
                auto idx = std::get<0>(instruction);
                auto ins = std::get<1>(instruction);

                if (dalvik_opcodes->get_instruction_operation(ins->get_OP()) ==
                    DVMTypes::Operation::BRANCH_DVM_OPCODE)
                {
                    auto v = determine_next(ins, idx, instructions);
                    h[idx] = v;
                    l.insert(l.end(), v.begin(), v.end());
                }
            }

            logger->debug("create_basic_block: parsing method exceptions.");

            auto excepts = determine_exception(dalvik_opcodes, std::dynamic_pointer_cast<EncodedMethod>(method_encoded));

            for (const auto &except : excepts)
            {
                l.push_back(except.try_value_start_addr);

                for (auto &handler : except.handler)
                {
                    l.push_back(handler.handler_start_addr);
                }
            }

            logger->debug("create_basic_block: creating the basic blocks with references.");

            for (const auto &instruction : instructions)
            {
                auto idx = std::get<0>(instruction);
                auto ins = std::get<1>(instruction);

                if (std::find(l.begin(), l.end(), static_cast<std::int64_t>(idx)) != l.end())
                {
                    if (current_basic->get_nb_instructions() != 0)
                    {
                        current_basic = std::make_shared<DVMBasicBlock>(current_basic->get_end(), dalvik_opcodes, basic_blocks, std::dynamic_pointer_cast<EncodedMethod>(method_encoded), instructions);
                        basic_blocks->push_basic_block(current_basic);
                    }
                }

                current_basic->push(ins);

                if (h.find(idx) != h.end())
                {
                    current_basic = std::make_shared<DVMBasicBlock>(current_basic->get_end(), dalvik_opcodes, basic_blocks, std::dynamic_pointer_cast<EncodedMethod>(method_encoded), instructions);
                    basic_blocks->push_basic_block(current_basic);
                }
            }

            if (current_basic->get_nb_instructions() == 0)
            {
                basic_blocks->pop_basic_block();
            }

            logger->debug("create_basic_blocks: setting basic blocks childs.");
            
            auto bbs = basic_blocks->get_basic_blocks();
            for (auto bb : bbs)
            {
                bb->set_child(h[bb->get_end() - bb->get_last_length()]);
            }

            logger->debug("create_basic_blocks: creating exceptions.");
            
            this->exceptions->add(excepts, this->basic_blocks);

            for (auto bb : bbs)
            {
                bb->set_exception_analysis(this->exceptions->get_exception(bb->get_start(), bb->get_end()));
            }
        }
    }
}