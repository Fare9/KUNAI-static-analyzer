#include <iostream>
#include <fstream>
#include <regex>

#include <Kunai/DEX/dex.hpp>
#include <Kunai/DEX/DVM/dvm_types.hpp>
#include "backward-slicing/backward_analysis.hpp"
//#include "KUNAI/DEX/DVM/dex_dvm_types.hpp"



int main(int argc, char **argv)
{

    // check argument
    if (argc != 3)
    {
        std::cerr << "[-] USAGE: " << argv[0] << " <dex_file> <targets_file> \n";
        std::cerr << "\t<dex_file>: dex file to disassemble\n";
        std::cerr << "\t<targets_file>: txt file with one target method per line in the form 'Class->Method(params)ret'\n";
        return -1;
    }

    // get analysis object
    auto dex_file = KUNAI::DEX::Dex::parse_dex_file(argv[1]);
    auto analysis = dex_file->get_analysis(true);
    // create xrefs explicitly
    analysis->create_xrefs();

    // parse targets file
    KUNAI::DEX::BackwardAnalysis backward_analysis;
    if (backward_analysis.parse_targets_file(argv[2]) == 0)
    {
        size_t i = 0;
        for (auto &t : backward_analysis.targets_info)
        {
            if (t.is_parsed_correct)
            {
                std::string access_flags = ".*";

                // Find method analysis object corresponding to the parsed line
                auto method_analysis_vector = analysis->find_methods(t.class_name, t.method_name, t.prototype, access_flags, false);
                if (!method_analysis_vector.empty())
                {
                    for (auto method_analysis : method_analysis_vector)
                    {
                        // Having the MethodAnalysis for our target method, now obtain its xreffrom list
                        // For each xreffrom we must find the instruction that invokes our 
                        // target to retrieve the registers that correspond to the parameters
                        auto &xreffrom = method_analysis->get_xreffrom();
                        for (auto xref : xreffrom)
                        {
                            std::cout << "\nTarget " << i << " xreffrom: " << get<1>(xref)->get_full_name() << "\n";
                            backward_analysis.find_parameters_registers(get<1>(xref), method_analysis);
                            
                            // test to print regs 
                            std::cout << "Parameters registers: ";
                            for (u_int8_t reg : backward_analysis.parameters_registers){
                                std::cout << "v" << unsigned(reg) << " \n";
                            }
                        }
                    }
                }
            }
            else
            {
                std::cout << "Target " << i << " : Couldn't find class and method" << std::endl;
            }
            i++;
            
        }
    }

    return 0;
}