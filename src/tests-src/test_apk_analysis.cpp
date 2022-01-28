#include <iostream>
#include <memory>

#include <spdlog/spdlog.h>

#include "apk.hpp"

int
main(int argc, char **argv)
{
    if (argc != 2)
    {
        std::cout << "USAGE: " << argv[0] << " <apk file>\n";
        return 1;
    }

    // Set the logging level in spdlog, we set to debug
    spdlog::set_level(spdlog::level::info);

    auto logger = KUNAI::LOGGER::logger();

    logger->info("Started the analysis of file {}", argv[1]);

    auto apk = KUNAI::APK::get_unique_apk_object(argv[1]);

    apk->analyze_apk_file();

    logger->info("Finished the analysis of file {}", argv[1]);

    logger->info("Removing temporal path {} with DEX files...", apk->get_path_to_unzip_folder().c_str());
}