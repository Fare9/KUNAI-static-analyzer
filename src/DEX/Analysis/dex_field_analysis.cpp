#include "KUNAI/DEX/Analysis/dex_analysis.hpp"

namespace KUNAI
{
    namespace DEX
    {
        /**
         * FieldAnalysis methods
         */

        FieldAnalysis::FieldAnalysis(encodedfield_t field) : field(field)
        {
        }

        void FieldAnalysis::add_xref_read(classanalysis_t class_object, methodanalysis_t method_object, std::uint64_t offset)
        {
            xrefread.push_back({class_object, method_object, offset});
        }

        void FieldAnalysis::add_xref_write(classanalysis_t class_object, methodanalysis_t method_object, std::uint64_t offset)
        {
            xrefwrite.push_back({class_object, method_object, offset});
        }
    }
}