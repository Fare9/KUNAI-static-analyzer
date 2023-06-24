#include <iostream>
#include <fstream>
#include <regex>

#include <Kunai/DEX/dex.hpp>

namespace KUNAI
{
namespace DEX
{

    class BackwardAnalysis{
        private:
            

        public:
            struct target_info {
                std::string class_name;
                std::string method_name;
                std::string prototype;
                bool is_parsed_correct;
            };

            std::vector<target_info> targets_info;
            std::vector<u_int8_t> parameters_registers;

            // Parse the targets file and extracts the corresponding data from each line
            int8_t parse_targets_file(std::string file_path);

            // Find the invoking instruction in the xreffrom and the registers corresponding to the parameters
            void find_parameters_registers(KUNAI::DEX::MethodAnalysis* xreffrom_method, KUNAI::DEX::MethodAnalysis* target_method);
    };

} // namespace DEX
} // namespace KUNAI