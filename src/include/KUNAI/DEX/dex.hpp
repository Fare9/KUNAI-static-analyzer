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

#include "KUNAI/DEX/parser/dex_parser.hpp"
#include "KUNAI/DEX/DVM/dex_dalvik_opcodes.hpp"
#include "KUNAI/DEX/DVM/dex_disassembler.hpp"
#include "KUNAI/DEX/Analysis/dex_analysis.hpp"

namespace KUNAI
{
    namespace DEX
    {
        class DEX;

        /**
         * @brief std::shared_ptr of a DEX class, the recommended type for using
         * this object is dex_t as it allows to keep a shared pointer that will be
         * freed once it is not used more.
         * Dex object is a representation of DEX files, it allows to access a parser
         * and a disassembler as well as an analysis object of the DEX file.
         */
        using dex_t = std::shared_ptr<DEX>;

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
             * @return dexparser_t&
             */
            dexparser_t &get_parser()
            {
                return dex_parser;
            }

            /**
             * @brief Get a DalvikOpcodes object, this is commonly used internally by
             *        disassembler and other classes for Dalvik information.
             *
             * @return dalvikopcodes_t&
             */
            dalvikopcodes_t &get_dalvik_opcode_object()
            {
                return dalvik_opcodes;
            }

            /**
             * @brief Get the DexDisassembler if you want to apply manually the
             *        disassembly of the instructions.
             *
             * @return dexdisassembler_t&
             */
            dexdisassembler_t &get_dex_disassembler()
            {
                return dex_disassembler;
            }

            /**
             * @brief This method returns the analysis object from the DEX
             *        this contains all the analyzed data, together with
             *        ClassAnalysis and MethodAnalysis objects. The instructions
             *        are already disassembled, and the methods are created with
             *        the CFG.
             *
             * @param create_xrefs: create xrefs in the analysis, use it only if necessary.
             * @return analysis_t
             */
            analysis_t get_dex_analysis(bool create_xrefs);

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
            dexparser_t dex_parser;
            dalvikopcodes_t dalvik_opcodes;
            dexdisassembler_t dex_disassembler;
            analysis_t dex_analysis;

            instruction_map_t method_instructions;

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
         * @return dex_t
         */
        dex_t get_shared_dex_object(std::ifstream &input_file, std::uint64_t file_size);
    }
}

#endif