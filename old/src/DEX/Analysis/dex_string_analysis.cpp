#include "KUNAI/DEX/Analysis/dex_analysis.hpp"

namespace KUNAI
{
    namespace DEX
    {
        
        StringAnalysis::StringAnalysis(std::string *value) :
            value(value),
            orig_value(value)
        {   
        }

        void StringAnalysis::add_xref_from(ClassAnalysis* class_object, MethodAnalysis* method_object, std::uint64_t offset)
        {
            xreffrom.push_back({class_object, method_object, offset});
        }
        
    } // namespace DEX
} // namespace KUNAI
