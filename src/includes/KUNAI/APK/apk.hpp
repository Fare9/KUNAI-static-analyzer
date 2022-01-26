/**
 * @file apk.hpp
 * @author @Farenain
 * @brief Class to manage the APK file.
 * 
 * An APK file will be a union of different files,
 * there will be one or more DEX files, .SO files
 * (ELF), an XML file which represents the manifest,
 * etc.
 */

#ifndef APK_HPP
#define APK_HPP

#include <iostream>
#include <memory>
#include <map>
#include <cstdio>
#include <filesystem>

#include "dex.hpp"
#include "dex_disassembler.hpp"
#include "dex_analysis.hpp"

#include "CkZip.h"
#include "CkZipEntry.h"

#include "utils.hpp"

namespace KUNAI
{
    namespace APK
    {
        /**
         * APK definition
         * 
         * @brief An APK will be the union of different components that
         * we will work with, this will involve the use of:
         * - DEX files.
         * - SO (ELF) files (not implemented yet).
         * - XML file: for the AndroidManifest.xml.
         * 
         */
        class APK
        {
        public:

            APK(std::string path_to_apk_file);
            
            ~APK();

            /**
             * @brief Analyze the given apk file for extracting
             * all the DEX files, the AndroidManifest.xml and
             * all the .so files (ELF).
             */
            void analyze_apk_file();

            /**
             * @brief Get the global disassembler object, this object
             * will contain the disassembly from all the DEX files,
             * this is necessary to create a global analysis object
             * too.
             * 
             * @return std::shared_ptr<DEX::DexDisassembler> 
             */
            std::shared_ptr<DEX::DexDisassembler> get_global_disassembler()
            {
                return global_disassembler;
            }

            /**
             * @brief Get the global analysis object, this global
             * analysis object will contain all the DEXParsers and
             * the global disassembler in order to analyze the whole
             * APK.
             * 
             * @return std::shared_ptr<DEX::Analysis> 
             */
            std::shared_ptr<DEX::Analysis> get_global_analysis()
            {
                return global_analysis;
            }

            /**
             * @brief Get the map of the DEX file objects.
             * 
             * @return const std::map<std::string, std::shared_ptr<DEX::DEX>>& 
             */
            const std::map<std::string, std::shared_ptr<DEX::DEX>>& get_dex_files() const
            {
                return dex_files;
            }

            /**
             * @brief Get a dex object by name or nullptr if dex file does not exists.
             * 
             * @param dex_name 
             * @return std::shared_ptr<DEX::DEX> 
             */
            std::shared_ptr<DEX::DEX> get_dex_by_name(std::string dex_name)
            {
                if (dex_files.find(dex_name) == dex_files.end())
                    return nullptr;
                return dex_files[dex_name];
            }

            /**
             * @brief Get the path to apk file given.
             * 
             * @return std::string& 
             */
            std::string& get_path_to_apk_file()
            {
                return path_to_apk_file;
            }

            /**
             * @brief Get the path to where KUNAI will unzip the content.
             * 
             * @return std::string& 
             */
            std::string& get_path_to_unzip_folder()
            {
                return temporal_path;
            }

        private:
            
            /**
             * @brief Extract one DEX file and create the necessary DEX object
             * with it.
             * 
             * @param dex_file 
             * @return std::shared_ptr<DEX::DEX>
             */
            std::shared_ptr<DEX::DEX> manage_dex_files_from_zip_entry(std::shared_ptr<CkZipEntry> dex_file);


            std::map<std::string, std::shared_ptr<DEX::DEX>> dex_files;
            std::shared_ptr<DEX::DexDisassembler> global_disassembler;
            std::shared_ptr<DEX::Analysis> global_analysis;

            std::string path_to_apk_file;
            std::string temporal_path;
        };

        /**
         * @brief Get the unique apk object object
         * 
         * @param path_to_apk_file 
         * @return std::unique_ptr<APK> 
         */
        std::unique_ptr<APK> get_unique_apk_object(std::string path_to_apk_file);

        /**
         * @brief Get the shared apk object object
         * 
         * @param path_to_apk_file 
         * @return std::shared_ptr<APK> 
         */
        std::shared_ptr<APK> get_shared_apk_object(std::string path_to_apk_file);
    }
}

#endif