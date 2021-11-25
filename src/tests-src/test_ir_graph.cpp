#include <iostream>
#include <memory>

#include "lifter_android.hpp"
#include "dex.hpp"

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        std::cout << "USAGE: " << argv[0] << " <dex_file>" << std::endl;
        return 1;
    }

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

    auto dex_analysis = dex->get_dex_analysis();

    auto main_method = dex_analysis->get_method_analysis_by_name("LMain;", "main", "([Ljava/lang/String;)V");

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

    auto graph = lifter_android->lift_android_method(main_method);

    auto nodes = graph->get_nodes();

    std::cout << "\n\nPrinting nodes from 'LMain;->main([Ljava/lang/String;)V' in MjolnIR\n";

    for (auto node : nodes)
    {
        std::cout << node->to_string();
        std::cout << "\n";
    }

    std::cout << "\nDumping .dot file into directory\n";

    graph->generate_dot_file("main");

    return 0;
}