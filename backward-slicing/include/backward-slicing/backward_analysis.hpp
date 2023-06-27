#include <iostream>
#include <fstream>
#include <regex>

#include <Kunai/DEX/dex.hpp>

namespace KUNAI
{
namespace DEX
{
    class BackwardAnalysis{
        public:
            struct target_info {
                std::string class_name;
                std::string method_name;
                std::string prototype;
                bool is_parsed_correct;
            };

            // Vector containing the parsed information of each target method
            std::vector<target_info> targets_info;

            // Vector containing the register number of each of the parameters of a target method
            std::vector<u_int8_t> parameters_registers;

            // Map for register number and a reference to its defining instruction
            std::unordered_map<u_int8_t, KUNAI::DEX::Instruction*> instruction_mapped_registers;

            // Parse the targets file and extracts the corresponding data from each line
            int8_t parse_targets_file(std::string file_path);

            // Find the invoking instruction in the xreffrom and the registers corresponding to the parameters
            void find_parameters_registers(KUNAI::DEX::MethodAnalysis* xreffrom_method, KUNAI::DEX::MethodAnalysis* target_method);

            // Find the instruction that defines each parameter of the target method from its 'xreffrom'
            void find_parameters_definition(KUNAI::DEX::MethodAnalysis *xreffrom_method);

            // Print the instruction where the parameters are first defined or the method name if it was
            // defined as a method parameter
            void print_parameters_definition(KUNAI::DEX::MethodAnalysis *xreffrom_method);
    };

} // namespace DEX
} // namespace KUNAI