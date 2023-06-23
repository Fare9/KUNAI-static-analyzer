#include "backward_analysis.hpp"

using namespace KUNAI::DEX;

int8_t BackwardAnalysis::parse_targets_file(std::string file_path){
    std::ifstream targetsfs;
    std::string myline;
    std::regex src_pattern("(L.+?;)->(.+?)(\\(.+\\).)");
    std::smatch match;

    targetsfs.open(file_path);
    if (targetsfs.is_open())
    {
        target_info t;
        while (targetsfs)
        {
            std::getline(targetsfs, myline);
            if (myline.length() != 0)
            {
                // Extract class, method and proto using regex
                if (std::regex_match(myline, match, src_pattern)){
                    t.class_name = match[1];
                    t.method_name = match[2];
                    t.prototype = match[3];
                    t.is_parsed_correct = true;
                }
                else
                {
                    t.class_name = "";
                    t.method_name = "";
                    t.prototype = "";
                    t.is_parsed_correct = false;
                }
                targets_info.push_back(t);
            }
        }
        targetsfs.close();
        return 0;
    }
    else
    {
        std::cerr << "Couldn't open file " << file_path << "\n";
        return -1;
    }
}

void BackwardAnalysis::find_parameters_registers(KUNAI::DEX::MethodAnalysis * xreffrom_method, KUNAI::DEX::MethodAnalysis *target_method){
    auto & instructions = xreffrom_method->get_instructions();
    const auto &encoded_method = target_method->get_encoded_method();
    if (std::holds_alternative<KUNAI::DEX::EncodedMethod*>(encoded_method))
    {
        auto &em = std::get<KUNAI::DEX::EncodedMethod*>(encoded_method);
        // read instructions to find all invoke instructions
        for ( auto &instruc : instructions){
            const auto instruc_type = instruc->get_instruction_type();
            if (instruc_type == KUNAI::DEX::dexinsttype_t::DEX_INSTRUCTION35C)
            {
                std::cout << "Invoke instruction in xreffrom method: " << instruc->print_instruction() << std::endl;

                KUNAI::DEX::Instruction35c* instruction35c = reinterpret_cast<KUNAI::DEX::Instruction35c*>(instruc.get());

                // find invoke instruction that calls the target method
                if (instruction35c->get_method() && instruction35c->get_method() == em->getMethodID()){
                    auto &regs = instruction35c->get_registers();
                    parameters_registers = regs;
                    // if instruction is invoke-virtual or invoke-direct params start from the second register
                    // KUNAI::DEX::DVMTypes::Opcode::OP_INVOKE_VIRTUAL || KUNAI::DEX::DVMTypes::Opcode::OP_INVOKE_DIRECT 
                    if (instruction35c->get_instruction_opcode() == 110 || 112) 
                    {   
                        parameters_registers.erase(parameters_registers.begin());
                    }
                    // else instruction is invoke-super, invoke-static or invoke-interface 
                    // params start from the first register
                }
            }
            else if (instruc_type == KUNAI::DEX::dexinsttype_t::DEX_INSTRUCTION45CC ||
            instruc_type == KUNAI::DEX::dexinsttype_t::DEX_INSTRUCTION4RCC)
            {
                // TODO: other types of invoke instructions
                continue;
            }
        
        } 
    }
    else
    {   
        // else not encoded method but external, we don't have the method id but could use method name
        // auto &em = std::get<KUNAI::DEX::ExternalMethod*>(encoded_method);
    
    }
}