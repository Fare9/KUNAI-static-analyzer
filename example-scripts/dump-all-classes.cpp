// compile me with: g++ -std=c++17 dump-all-classes.cpp -o dump-all-classes -lkunai

#include <iostream>

#include <KUNAI/APK/apk.hpp>
#include <spdlog/spdlog.h>

int main(int argc, char **argv)
{
    if (argc == 1)
    {
        std::cerr << "[-] USAGE: " << argv[0] << " <file.apk> [<class_name>]" << std::endl;
        return -1;
    }
    std::vector<std::string> all_args;

    all_args.assign(argv + 1, argv + argc);

    // Set the logging level in spdlog, we set to info
    // so we only see info and error messages
    spdlog::set_level(spdlog::level::debug);

    auto logger = KUNAI::LOGGER::logger();

    std::cout << "Starting the analysis of the APK file " << all_args[0] << "\n";

    auto apk_file = KUNAI::APK::get_unique_apk_object(all_args[0], false);

    apk_file->analyze_apk_file();

    std::cout << "Finished the analysis of the APK file\n";

    std::cout << "Obtaining classes...";

    auto analysis_object = apk_file->get_global_analysis();

    auto class_analysis_objects = analysis_object->get_classes();

    /*
     *  Go over all the classes from the APK and print the name
     *  of the classes and the name of their methods.
     */

    for (auto class_analysis_object : class_analysis_objects)
    {
        if (class_analysis_object->is_android_api())
            continue;

        std::cout << "Class name: " << class_analysis_object->name() << "\n";

        auto methods = class_analysis_object->get_methods();

        for (auto method : methods)
        {
            std::cout << "\t|->Method name: " << method->full_name() << "\n";
        }
    }

    if (argc == 3)
    {
        std::cout << "Obtaining class " << all_args[1] << "\n";

        /*
         * Now we take the class analysis given its name.
         * From it, we will extract all the xrefs that the class creates.
         */
        
        auto chosen_class = analysis_object->get_class_analysis(all_args[1]);

        if (!chosen_class)
        {
            std::cerr << "Error class: " << argv[2] << " does not exists\n";
            return 1;
        }

        /*
        * The returned value is a map with the class pointed as key
        * and a set with the reference type, method where class is referenced
        * and finally the address where it is referenced.
        * const std::map<std::shared_ptr<ClassAnalysis>,
                           std::set<std::tuple<DVMTypes::REF_TYPE,
                                               std::shared_ptr<MethodAnalysis>,
                                               std::uint64_t>>>
        */
        auto xrefs_to = chosen_class->get_xref_to();

        std::cout << "Xrefs to:\n";

        for (const auto &xref_to : xrefs_to)
        {
            std::cout << "|->" << xref_to.first->name() << "\n";
            for (const auto &tuple_xref : xref_to.second)
            {

                std::cout << "|\t-> xref type: ";

                switch (std::get<0>(tuple_xref))
                {
                case KUNAI::DEX::DVMTypes::REF_NEW_INSTANCE:
                    std::cout << "REF_NEW_INSTANCE\n";
                    break;
                case KUNAI::DEX::DVMTypes::REF_CLASS_USAGE:
                    std::cout << "REF_CLASS_USAGE\n";
                    break;
                case KUNAI::DEX::DVMTypes::INVOKE_VIRTUAL:
                    std::cout << "INVOKE_VIRTUAL\n";
                    break;
                case KUNAI::DEX::DVMTypes::INVOKE_SUPER:
                    std::cout << "INVOKE_SUPER\n";
                    break;
                case KUNAI::DEX::DVMTypes::INVOKE_DIRECT:
                    std::cout << "INVOKE_DIRECT\n";
                    break;
                case KUNAI::DEX::DVMTypes::INVOKE_STATIC:
                    std::cout << "INVOKE_STATIC\n";
                    break;
                case KUNAI::DEX::DVMTypes::INVOKE_INTERFACE:
                    std::cout << "INVOKE_INTERFACE\n";
                    break;
                case KUNAI::DEX::DVMTypes::INVOKE_VIRTUAL_RANGE:
                    std::cout << "INVOKE_VIRTUAL_RANGE\n";
                    break;
                case KUNAI::DEX::DVMTypes::INVOKE_SUPER_RANGE:
                    std::cout << "INVOKE_SUPER_RANGE\n";
                    break;
                case KUNAI::DEX::DVMTypes::INVOKE_DIRECT_RANGE:
                    std::cout << "INVOKE_DIRECT_RANGE\n";
                    break;
                case KUNAI::DEX::DVMTypes::INVOKE_STATIC_RANGE:
                    std::cout << "INVOKE_STATIC_RANGE\n";
                    break;
                case KUNAI::DEX::DVMTypes::INVOKE_INTERFACE_RANGE:
                    std::cout << "INVOKE_INTERFACE_RANGE\n";
                    break;
                }

                std::cout << "|\t-> Method where is referenced: " << std::get<1>(tuple_xref)->name() << "\n";
                std::cout << "|\t-> Line where xref is done: " << std::get<2>(tuple_xref) << "\n";
            }
        }
    }

    return 0;
}