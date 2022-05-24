#include "dex_analysis.hpp"

namespace KUNAI
{
    namespace DEX
    {
        /**
         * MethodAnalysis methods
         */

        MethodAnalysis::MethodAnalysis(std::variant<encodedmethod_t, externalmethod_t> method_encoded, dalvikopcodes_t dalvik_opcodes, std::map<std::uint64_t, instruction_t> instructions) : method_encoded(method_encoded),
                                                                                                                                                                                                          dalvik_opcodes(dalvik_opcodes),
                                                                                                                                                                                                          instructions(instructions)
        {
            this->is_external = method_encoded.index() == 0 ? false : true;
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
            if (is_external)
            {
                return std::get<externalmethod_t>(method_encoded)->get_name();
            }
            else
            {
                return *std::get<encodedmethod_t>(method_encoded)->get_method()->get_method_name();
            }
        }

        std::string MethodAnalysis::descriptor()
        {
            if (is_external)
            {
                return std::get<externalmethod_t>(method_encoded)->get_descriptor();
            }
            else
            {
                return std::get<encodedmethod_t>(method_encoded)->get_method()->get_method_prototype()->get_proto_str();
            }
        }

        std::string MethodAnalysis::access()
        {
            std::string access_flag;

            if (is_external)
            {
                externalmethod_t method = std::get<externalmethod_t>(method_encoded);
                access_flag = this->dalvik_opcodes->get_access_flags_string(method->get_access_flags());
            }
            else
            {
                encodedmethod_t method = std::get<encodedmethod_t>(method_encoded);
                access_flag = this->dalvik_opcodes->get_access_flags_string(method->get_access_flags());
            }

            return access_flag;
        }

        std::string MethodAnalysis::class_name()
        {
            std::string class_name;

            if (is_external)
            {
                externalmethod_t method = std::get<externalmethod_t>(method_encoded);
                class_name = method->get_class_name();
            }
            else
            {
                encodedmethod_t method = std::get<encodedmethod_t>(method_encoded);
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

        void MethodAnalysis::add_xref_read(classanalysis_t class_object, fieldanalysis_t field_object, std::uint64_t offset)
        {
            xrefread.push_back({class_object, field_object, offset});
        }

        void MethodAnalysis::add_xref_write(classanalysis_t class_object, fieldanalysis_t field_object, std::uint64_t offset)
        {
            xrefwrite.push_back({class_object, field_object, offset});
        }

        void MethodAnalysis::add_xref_to(classanalysis_t class_object, methodanalysis_t method_object, std::uint64_t offset)
        {
            xrefto.push_back({class_object, method_object, offset});
        }

        void MethodAnalysis::add_xref_from(classanalysis_t class_object, methodanalysis_t method_object, std::uint64_t offset)
        {
            xreffrom.push_back({class_object, method_object, offset});
        }

        void MethodAnalysis::add_xref_new_instance(classanalysis_t class_object, std::uint64_t offset)
        {
            xrefnewinstance.push_back({class_object, offset});
        }

        void MethodAnalysis::add_xref_const_class(classanalysis_t class_object, std::uint64_t offset)
        {
            xrefconstclass.push_back({class_object, offset});
        }

        void MethodAnalysis::create_basic_block()
        {
            auto logger = LOGGER::logger();

            basic_blocks = std::make_shared<BasicBlocks>();
            dvmbasicblock_t current_basic = std::make_shared<DVMBasicBlock>(0, dalvik_opcodes, basic_blocks, std::get<encodedmethod_t>(method_encoded), instructions);

            // push the first basic block
            basic_blocks->push_basic_block(current_basic);

            std::vector<std::int64_t> l;
            std::map<std::uint64_t, std::vector<std::int64_t>> h;


            logger->debug("create_basic_block: creating basic blocks for method {}.", std::get<encodedmethod_t>(method_encoded)->full_name());

            for (auto const &instruction : instructions)
            {
                auto idx = std::get<0>(instruction);
                auto ins = std::get<1>(instruction);

                if (dalvik_opcodes->get_instruction_operation(ins->get_OP()) ==
                    DVMTypes::Operation::BRANCH_DVM_OPCODE)
                {
                    auto v = determine_next(ins, idx);
                    h[idx] = v;
                    l.insert(l.end(), v.begin(), v.end());
                }
            }

            #ifdef DEBUG
            logger->debug("create_basic_block: parsing method exceptions.");
            #endif

            auto excepts = determine_exception(dalvik_opcodes, std::get<encodedmethod_t>(method_encoded));

            for (const auto &except : excepts)
            {
                l.push_back(except.try_value_start_addr);

                for (auto &handler : except.handler)
                {
                    l.push_back(handler.handler_start_addr);
                }
            }

            #ifdef DEBUG
            logger->debug("create_basic_block: creating the basic blocks with references.");
            #endif

            for (const auto &instruction : instructions)
            {
                auto idx = std::get<0>(instruction);
                auto ins = std::get<1>(instruction);

                if (std::find(l.begin(), l.end(), static_cast<std::int64_t>(idx)) != l.end())
                {
                    if (current_basic->get_nb_instructions() != 0)
                    {
                        current_basic = std::make_shared<DVMBasicBlock>(current_basic->get_end(), dalvik_opcodes, basic_blocks, std::get<encodedmethod_t>(method_encoded), instructions);
                        basic_blocks->push_basic_block(current_basic);
                    }
                }

                current_basic->push(ins);

                if (h.find(idx) != h.end())
                {
                    current_basic = std::make_shared<DVMBasicBlock>(current_basic->get_end(), dalvik_opcodes, basic_blocks, std::get<encodedmethod_t>(method_encoded), instructions);
                    basic_blocks->push_basic_block(current_basic);
                }
            }

            if (current_basic->get_nb_instructions() == 0)
            {
                basic_blocks->pop_basic_block();
            }

            #ifdef DEBUG
            logger->debug("create_basic_blocks: setting basic blocks childs.");
            #endif

            auto bbs = basic_blocks->get_basic_blocks();
            for (auto bb : bbs)
            {
                bb->set_child(h[bb->get_end() - bb->get_last_length()]);
            }

            #ifdef DEBUG
            logger->debug("create_basic_blocks: creating exceptions.");
            #endif
            
            this->exceptions->add(excepts, this->basic_blocks);

            for (auto bb : bbs)
            {
                bb->set_exception_analysis(this->exceptions->get_exception(bb->get_start(), bb->get_end()));
            }
        }
    }
}