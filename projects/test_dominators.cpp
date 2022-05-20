#include <iostream>
#include <memory>

#include "lifter_android.hpp"
#include "dex.hpp"

int main(int argc, char **argv)
{
    if (argc != 5)
    {
        std::cout << "USAGE: " << argv[0] << " <dex_file> <class_name> <method_name> <prototype>" << std::endl;
        return 1;
    }

    std::string class_name = argv[2];
    std::string method_name = argv[3];
    std::string prototype = argv[4];

    std::ifstream dex_file;

    dex_file.open(argv[1], std::ios::binary);

    auto fsize = dex_file.tellg();
    dex_file.seekg(0, std::ios::end);
    fsize = dex_file.tellg() - fsize;
    dex_file.seekg(0);

    // get a unique_ptr from a DEX object.
    // this will parse the DEX file, and nothing more.
    auto dex = KUNAI::DEX::get_unique_dex_object(dex_file, fsize);

    // check if parsing was correct.
    if (!dex->get_parsing_correct())
    {
        std::cerr << "[-] Parsing of DEX file was not correct." << std::endl;
        return 1;
    }

    auto dex_analysis = dex->get_dex_analysis(true);

    auto main_method = dex_analysis->get_method_analysis_by_name(class_name, method_name, prototype);

    if (main_method == nullptr)
    {
        std::cerr << "[-] Method selected doesn't exist, or it has not been found." << std::endl;
        return 1;
    }

    if (main_method->external())
    {
        std::cerr << "[-] External method, cannot be lifted to IR..." << std::endl;
        return 1;
    }

    auto lifter_android = std::make_shared<KUNAI::LIFTER::LifterAndroid>();

    auto graph = lifter_android->lift_android_method(main_method, dex_analysis);


    if (graph->get_nodes().size() > 0)
    {
        auto first_node = graph->get_nodes()[0];

        std::cout << "Computing dominators of the graph...\n";

        auto dominators = graph->compute_dominators(first_node);

        for (auto dominator : dominators)
        {
            auto head = dominator.first;
            auto childrens = dominator.second;

            std::cout << head->get_name() << ": ";

            for (auto children : childrens)
            {
                std::cout << children->get_name() << " ";
            }

            std::cout << "\n";
        }

        auto last_node = graph->get_nodes().back();

        std::cout << "Computing predominators of the graph...\n";

        auto predominators = graph->compute_postdominators(last_node);

        for (auto predominator : predominators)
        {
            auto head = predominator.first;
            auto childrens = predominator.second;

            std::cout << head->get_name() << ": ";

            for (auto children : childrens)
            {
                std::cout << children->get_name() << " ";
            }

            std::cout << "\n";
        }

        std::cout << "Computing immediate dominators...\n";

        auto i_domminators = graph->compute_immediate_dominators();

        for (auto idom : i_domminators)
        {
            std::cout << idom.first->get_name() << ": ";

            if (idom.second)
                std::cout << idom.second->get_name();
            
            std::cout << "\n";
        }

        std::cout << "Generating idom.dot with dominator tree...\n";
        
        graph->generate_dominator_tree("idom");
    }

    return 0;
}