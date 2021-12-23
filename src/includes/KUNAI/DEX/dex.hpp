/***
 * @file dex.hpp
 * @author @Farenain
 *
 * @brief Class to manage a DEX file.
 *
 * This class must serve to the user as an starting
 * point of the analysis of a DEX file, here user should
 * have access to analysis objets for classes, methods,
 * instructions and so on.
 */

#ifndef DEX_HPP
#define DEX_HPP

#include <iostream>
#include <memory>

#include "dex_parser.hpp"
#include "dex_dalvik_opcodes.hpp"
#include "dex_disassembler.hpp"
#include "dex_analysis.hpp"

namespace KUNAI
{
    namespace DEX
    {
        class DEX
        {
        public:
            /**
             * @brief Construct a new DEX::DEX object, this will just apply the parsing,
             *        it will also create the objects for the DalvikOpcodes and the DexDisassembler.
             *        the disassembly will not be applied in the first moment.
             *
             * @param input_file
             * @param file_size
             */
            DEX(std::ifstream &input_file, std::uint64_t file_size);

            /**
             * @brief Destroy the DEX::DEX object
             */
            ~DEX() = default;

            /**
             * @brief Get the DexParser of the DEX file, this will contain all the headers.
             *        Parsing was already applied.
             *
             * @return std::shared_ptr<DexParser>
             */
            std::shared_ptr<DexParser> get_parser()
            {
                if (dex_parsing_correct)
                    return dex_parser;
                return nullptr;
            }

            /**
             * @brief Get a DalvikOpcodes object, this is commonly used internally by
             *        disassembler and other classes for Dalvik information.
             *
             * @return std::shared_ptr<DalvikOpcodes>
             */
            std::shared_ptr<DalvikOpcodes> get_dalvik_opcode_object()
            {
                if (dalvik_opcodes)
                    return dalvik_opcodes;
                return nullptr;
            }

            /**
             * @brief Get the DexDisassembler if you want to apply manually the
             *        disassembly of the instructions.
             *
             * @return std::shared_ptr<DexDisassembler>
             */
            std::shared_ptr<DexDisassembler> get_dex_disassembler()
            {
                if (dex_disassembler)
                    return dex_disassembler;
                return nullptr;
            }

            /**
             * @brief This method returns the analysis object from the DEX
             *        this contains all the analyzed data, together with
             *        ClassAnalysis and MethodAnalysis objects. The instructions
             *        are already disassembled, and the methods are created with
             *        the CFG.
             *
             * @return std::shared_ptr<Analysis>
             */
            std::shared_ptr<Analysis> get_dex_analysis();

            /**
             * @brief Get if parsing was correct or not.
             *
             * @return true
             * @return false
             */
            bool get_parsing_correct()
            {
                return dex_parsing_correct;
            }

        private:
            std::shared_ptr<DexParser> dex_parser;
            std::shared_ptr<DalvikOpcodes> dalvik_opcodes;
            std::shared_ptr<DexDisassembler> dex_disassembler;
            std::shared_ptr<Analysis> dex_analysis;

            std::map<std::tuple<std::shared_ptr<ClassDef>, std::shared_ptr<EncodedMethod>>, std::map<std::uint64_t, std::shared_ptr<Instruction>>> method_instructions;

            bool dex_parsing_correct;
        };

        /**
         * @brief Get the unique dex object object
         *
         * @param input_file
         * @param file_size
         * @return std::unique_ptr<DEX>
         */
        std::unique_ptr<DEX> get_unique_dex_object(std::ifstream &input_file, std::uint64_t file_size);

        /**
         * @brief Get the shared dex object object
         *
         * @param input_file
         * @param file_size
         * @return std::shared_ptr<DEX>
         */
        std::shared_ptr<DEX> get_shared_dex_object(std::ifstream &input_file, std::uint64_t file_size);
    }
}

#endif