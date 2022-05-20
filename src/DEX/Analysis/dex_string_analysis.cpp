#include "dex_analysis.hpp"

namespace KUNAI
{
    namespace DEX
    {
        
        StringAnalysis::StringAnalysis(std::string *value) :
            value(value),
            orig_value(value)
        {   
        }

        StringAnalysis::~StringAnalysis()
        {
            if (!xreffrom.empty())
                xreffrom.clear();
        }

        void StringAnalysis::add_xref_from(classanalysis_t class_object, methodanalysis_t method_object, std::uint64_t offset)
        {
            xreffrom.push_back({class_object, method_object, offset});
        }
        
    } // namespace DEX
} // namespace KUNAI
