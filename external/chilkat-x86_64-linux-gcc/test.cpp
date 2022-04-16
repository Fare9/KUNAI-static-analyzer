#include "include/CkZip.h"
#include "include/CkZipEntry.h"

#include <iostream>
#include <string>

void ChilkatSample(char * apk_file)
{
    // This example requires the Chilkat API to have been previously unlocked.
    // See Global Unlock Sample for sample code.

    CkZip zip;

    bool success = zip.OpenZip(apk_file);
    if (success != true)
    {
        std::cout << zip.lastErrorText() << "\r\n";
        return;
    }

    const char *unzipDir = "/tmp/unzipDir";

    int n;

    // Get the number of files and directories in the .zip
    n = zip.get_NumEntries();
    std::cout << n << "\r\n";

    CkZipEntry *entry = 0;

    int i;
    for (i = 0; i <= n - 1; i++)
    {

        entry = zip.GetEntryByIndex(i);
        if (entry->get_IsDirectory() == false)
        {
            std::string file_name = entry->fileName();

            if (file_name.find(".dex") == std::string::npos)
                continue;

            // (the filename may include a path)
            std::cout << file_name << "\r\n";

            // Your application may choose to unzip this entry
            // based on the filename.
            // If the entry should be unzipped, then call Extract(unzipDir)
            success = entry->Extract(unzipDir);
            if (success != true)
            {
                std::cout << entry->lastErrorText() << "\r\n";
                return;
            }
        }

        delete entry;
    }
}

int
main(int argc, char **argv)
{
    if (argc != 2)
    {
        std::cout << "Usage: " << argv[0] << " <apk file>\n";
        return 1;
    }

    ChilkatSample(argv[1]);
}