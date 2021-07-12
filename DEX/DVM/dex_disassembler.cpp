#include "dex_disassembler.hpp"

namespace KUNAI
{
    namespace DEX
    {
        DexDisassembler::DexDisassembler(bool parsing_correct, std::shared_ptr<DexParser> dex_parser, std::shared_ptr<DalvikOpcodes> dalvik_opcodes)
        {
            this->parsing_correct = parsing_correct;
            this->dex_parser = dex_parser;
            this->disassembly_correct = false;
            this->dalvik_opcodes = dalvik_opcodes;
            this->dalvik_disassembler = std::make_shared<LinearSweepDisassembler>(dalvik_opcodes);
        }

        DexDisassembler::~DexDisassembler() {}

        /**
         * @brief get if the disassembly was correct or not.
         * @return bool.
         */
        bool DexDisassembler::get_disassembly_correct()
        {
            return disassembly_correct;
        }


        /**
         * @brief Public method for disassembly with error checking.
         * @return void.
         */
        void DexDisassembler::disassembly_analysis()
        {
            if (!parsing_correct)
                std::cerr << "[-] DEX was not correctly parsed, cannot disassembly it" << std::endl;
            
            try
            {
                this->disassembly_methods();
                this->disassembly_correct = true;
            }
            catch(const std::exception& e)
            {
                std::cerr << "[-] Disassembly error: " << e.what() << '\n';
                this->disassembly_correct = false;
            }
        }

        /**
         * @brief Get the linear sweep disassembler object.
         * @return std::shared_ptr<LinearSweepDisassembler>
         */
        std::shared_ptr<LinearSweepDisassembler> DexDisassembler::get_linear_sweep_disassembler()
        {
            return dalvik_disassembler;
        }

        /**
         * @brief Disassembly DEX methods using linear sweep disassembly, store instructions with its class and method.
         * @return void.
         */
        void DexDisassembler::disassembly_methods()
        {
            auto dex_classes = dex_parser->get_classes();

            for (size_t i = 0; i < dex_classes->get_number_of_classes(); i++)
            {
                // get class def
                auto class_def = dex_classes->get_class_by_pos(i);

                // get ClassDataItem
                auto class_data_item = class_def->get_class_data();

                // in case of interfaces, or classes with
                // only virtual methods, no "classess_off"
                // in the analysis.
                if (class_data_item == nullptr)
                    continue;

                // now get direct method
                // for each one we will start the disassembly.
                for (size_t j = 0; j < class_data_item->get_number_of_direct_methods(); j++)
                {
                    auto direct_method = class_data_item->get_direct_method_by_pos(j);

                    // get code item struct, here is the instruction buffer
                    // here we will apply the disassembly.
                    auto code_item_struct = direct_method->get_code_item();

                    if (!code_item_struct)
                        continue;

                    dalvik_opcodes->set_number_of_registers(code_item_struct->get_number_of_registers_in_code());

                    if (direct_method->get_access_flags() & DVMTypes::ACCESS_FLAGS::ACC_STATIC)
                        dalvik_opcodes->set_number_of_parameters(direct_method->get_method()->get_method_prototype()->get_number_of_parameters());
                    else
                        dalvik_opcodes->set_number_of_parameters(direct_method->get_method()->get_method_prototype()->get_number_of_parameters() + 1);
                    
                    
                    
                    auto instructions = this->dalvik_disassembler->disassembly(code_item_struct->get_all_raw_instructions());

                    this->method_instructions[{class_def, direct_method}] = instructions;
                }

                for (size_t j = 0; j < class_data_item->get_number_of_virtual_methods(); j++)
                {
                    auto virtual_method = class_data_item->get_virtual_method_by_pos(j);

                    auto code_item_struct = virtual_method->get_code_item();

                    if (!code_item_struct)
                        continue;

                    dalvik_opcodes->set_number_of_registers(code_item_struct->get_number_of_registers_in_code());
                    
                    if (virtual_method->get_access_flags() & DVMTypes::ACCESS_FLAGS::ACC_STATIC)
                        dalvik_opcodes->set_number_of_parameters(virtual_method->get_method()->get_method_prototype()->get_number_of_parameters());
                    else
                        dalvik_opcodes->set_number_of_parameters(virtual_method->get_method()->get_method_prototype()->get_number_of_parameters() + 1);

                    auto instructions = this->dalvik_disassembler->disassembly(code_item_struct->get_all_raw_instructions());

                    this->method_instructions[{class_def, virtual_method}] = instructions;
                }
            }
        }


        std::ostream& operator<<(std::ostream& os, const DexDisassembler& entry)
        {
            for (auto it = entry.method_instructions.begin(); it != entry.method_instructions.end(); it++)
            {
                auto class_def = std::get<0>(it->first);
                auto direct_method = std::get<1>(it->first);

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
    }
}