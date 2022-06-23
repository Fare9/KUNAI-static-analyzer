#include "KUNAI/DEX/DVM/dex_disassembler.hpp"

namespace KUNAI
{
    namespace DEX
    {
        DexDisassembler::DexDisassembler(bool parsing_correct, dexparser_t dex_parser, dalvikopcodes_t dalvik_opcodes) : parsing_correct(parsing_correct),
                                                                                                                                                       dex_parser(dex_parser),
                                                                                                                                                       disassembly_correct(false),
                                                                                                                                                       dalvik_opcodes(dalvik_opcodes)
        {
            this->dalvik_disassembler = std::make_shared<LinearSweepDisassembler>(dalvik_opcodes);
        }

        void DexDisassembler::disassembly_analysis()
        {
            auto logger = LOGGER::logger();

            if (!parsing_correct)
            {
                logger->info("Incorrect parsing of dex header, not possible to disassembly methods.");
                return;
            }

            try
            {
                this->disassembly_methods();
                this->disassembly_correct = true;
                logger->info("DexDisassembler disassembly of methods was correct.");
            }
            catch (const std::exception &e)
            {
                logger->info("DexDisassembler, error in disassembly = '{}'", e.what());
                this->disassembly_correct = false;
            }
        }

        void DexDisassembler::disassembly_methods()
        {
            auto logger = LOGGER::logger();

            auto dex_classes = dex_parser->get_classes();

#ifdef DEBUG
            logger->debug("DexDisassembler disassembly a total of {} DEX classes", dex_classes->get_number_of_classes());
#endif

            // for (size_t i = 0, n_of_classes = static_cast<size_t>(dex_classes->get_number_of_classes()); i < n_of_classes; i++)

            auto class_defs = dex_classes->get_classes();

            for (auto class_def : class_defs)
            {
                // get ClassDataItem
                auto class_data_item = class_def->get_class_data();

                // in case of interfaces, or classes with
                // only virtual methods, no "classess_off"
                // in the analysis.
                if (class_data_item == nullptr)
                    continue;

#ifdef DEBUG
                static size_t i = 0;
                logger->debug("For class number {}, disassembly a total of {} DEX direct methods", i++, class_data_item->get_methods().size());
#endif

                for (auto method : class_data_item->get_methods())
                {
                    auto code_item_struct = method->get_code_item();

                    if (!code_item_struct)
                        continue;

                    auto instructions = this->dalvik_disassembler->disassembly(code_item_struct->get_all_raw_instructions());

                    for (auto instruction : instructions)
                    {
                        instruction.second->set_number_of_registers(code_item_struct->get_number_of_registers_in_code());

                        if (method->get_access_flags() & DVMTypes::ACCESS_FLAGS::ACC_STATIC)
                            instruction.second->set_number_of_parameters(method->get_method()->get_method_prototype()->get_number_of_parameters());
                        else
                            instruction.second->set_number_of_parameters(method->get_method()->get_method_prototype()->get_number_of_parameters() + 1);
                    }

                    this->method_instructions[{class_def, method}] = instructions;
                }
            }
        }

        void DexDisassembler::add_disassembly(dexdisassembler_t disas)
        {
            method_instructions.insert(disas->get_instructions().begin(), disas->get_instructions().end());
            disassembly_correct &= disas->get_disassembly_correct();
        }

        std::ostream &operator<<(std::ostream &os, const DexDisassembler &entry)
        {
            auto logger = LOGGER::logger();

            for (auto it = entry.method_instructions.begin(); it != entry.method_instructions.end(); it++)
            {
                auto class_def = std::get<0>(it->first);
                auto direct_method = std::get<1>(it->first);

                if (!class_def || !direct_method || !class_def->get_class_idx() || !direct_method->get_method() || !direct_method->get_method()->get_method_name())
                {
                    logger->warn("Error class_def object, or direct_method object are null or one of their fields is null, going to next one");
                    continue;
                }

                os << "Disassembly of method " << class_def->get_class_idx()->get_raw() << "->" << *direct_method->get_method()->get_method_name() << std::endl;

                for (auto instr = it->second.begin(); instr != it->second.end(); instr++)
                {
                    os << std::right << std::setfill('0') << std::setw(8) << std::hex << instr->first << "  ";
                    instr->second->give_me_instruction(os);
                    os << std::endl;
                }

                os << std::endl;
            }

            return os;
        }

        /**
         * @brief Operator + to join two disassemblers, this will join the two maps
         * as well as we will & the disassembly_correct variable.
         *
         * @param other_disassembler
         * @return DexDisassembler&
         */
        DexDisassembler &operator+(DexDisassembler &first_disassembler, DexDisassembler &other_disassembler)
        {
            first_disassembler.method_instructions.insert(other_disassembler.method_instructions.begin(), other_disassembler.method_instructions.end());
            first_disassembler.disassembly_correct &= other_disassembler.disassembly_correct;
            first_disassembler.parsing_correct &= other_disassembler.parsing_correct;

            return first_disassembler;
        }
    }
}