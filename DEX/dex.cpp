#include "dex.hpp"

namespace KUNAI
{
    namespace DEX
    {
        DEX::DEX(std::ifstream &input_file, std::uint64_t file_size)
        {
            try
            {
                this->dex_parser = std::make_shared<DexParser>();
                this->dex_parser->parse_dex_file(input_file, file_size);
                this->dex_parsing_correct = true;
                this->dalvik_opcodes = std::make_shared<DalvikOpcodes>(dex_parser);
                this->dalvik_disassembler = std::make_shared<LinearSweepDisassembler>(dalvik_opcodes);

                this->disassembly_methods();
            }
            catch (const std::exception &e)
            {
                std::cout << e.what();
                this->dex_parsing_correct = false;
            }
        }

        std::shared_ptr<DexParser> DEX::get_parser()
        {
            if (dex_parsing_correct)
                return dex_parser;
            return nullptr;
        }

        std::shared_ptr<DalvikOpcodes> DEX::get_dalvik_opcode_object()
        {
            if (dalvik_opcodes)
                return dalvik_opcodes;
            return nullptr;
        }

        void DEX::disassembly_methods()
        {
            auto dex_classes = dex_parser->get_classes();

            for (size_t i = 0; i < dex_classes->get_number_of_classes(); i++)
            {
                // get class def
                auto class_def = dex_classes->get_class_by_pos(i);

                // get ClassDataItem
                auto class_data_item = class_def->get_class_data();

                // now get direct methods and virtual methods
                // for each one we will start the disassembly.
                for (size_t j = 0; j < class_data_item->get_number_of_direct_methods(); j++)
                {
                    auto direct_method = class_data_item->get_direct_method_by_pos(j);

                    std::cout << "Disassembly of method: " << direct_method->get_method()->get_method_class()->get_raw() << "->" << *direct_method->get_method()->get_method_name() << std::endl;

                    // get code item struct, here is the instruction buffer
                    // here we will apply the disassembly.
                    auto code_item_struct = direct_method->get_code_item();
                    
                    auto instructions = this->dalvik_disassembler->disassembly(code_item_struct->get_all_raw_instructions());

                    for (auto it = instructions.begin(); it != instructions.end(); it++)
                    {
                        std::cout << it->first << ": ";
                        it->second->show_instruction();
                        std::cout << std::endl;
                    }

                    std::cout << "\n\n";
                }
            }
        }

        DEX::~DEX() {}

    }
}