#include <sstream>
#include <iomanip>
#include <vector>
#include <sys/stat.h>
#include <libunwind.h>
#include <cxxabi.h>
#include "app_utils.hpp"




// ============================================================================
std::map<std::string, std::string> cmdline::parse_keyval(int argc, const char* argv[])
{
    std::map<std::string, std::string> items;

    for (int n = 0; n < argc; ++n)
    {
        std::string arg = argv[n];
        std::string::size_type eq_index = arg.find('=');

        if (eq_index != std::string::npos)
        {
            std::string key = arg.substr (0, eq_index);
            std::string val = arg.substr (eq_index + 1);
            items[key] = val;
        }
    }
    return items;
}




// ============================================================================
std::vector<std::string> FileSystem::splitPath (std::string pathName)
{
    auto remaining = pathName;
    auto dirs = std::vector<std::string>();

    while (true)
    {
        auto slash = remaining.find ('/');

        if (slash == std::string::npos)
        {
            dirs.push_back (remaining);
            break;
        }
        dirs.push_back (remaining.substr (0, slash));
        remaining = remaining.substr (slash + 1);
    }
    return dirs;
}

std::string FileSystem::fileExtension (std::string pathName)
{
    auto dot = pathName.rfind ('.');

    if (dot != std::string::npos)
    {
        return pathName.substr (dot);
    }
    return "";
}

std::string FileSystem::getParentDirectory (std::string pathName)
{
    std::string::size_type lastSlash = pathName.find_last_of ("/");
    return pathName.substr (0, lastSlash);
}

void FileSystem::ensureDirectoryExists (std::string dirName)
{
    auto path = std::string (".");

    for (auto dir : splitPath (dirName))
    {
        path += "/" + dir;
        mkdir (path.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    }
}

void FileSystem::ensureParentDirectoryExists (std::string pathName)
{
    std::string parentDir = getParentDirectory (pathName);
    mkdir (parentDir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
}

std::string FileSystem::makeFilename (
    std::string directory,
    std::string base,
    std::string extension,
    int number,
    int rank)
{
    std::stringstream filenameStream;
    filenameStream << directory << "/" << base;

    if (number >= 0)
    {
        filenameStream << "." << std::setfill ('0') << std::setw (4) << number;
    }
    if (rank != -1)
    {
        filenameStream << "." << std::setfill ('0') << std::setw (4) << rank;
    }

    filenameStream << extension;
    return filenameStream.str();
}




// ============================================================================
void Debug::backtrace()
{
    std::cout << std::string(52, '=') << std::endl;
    std::cout << "Backtrace:\n";
    std::cout << std::string(52, '=') << std::endl;

    unw_cursor_t cursor;
    unw_context_t context;

    // Initialize cursor to current frame for local unwinding.
    unw_getcontext(&context);
    unw_init_local(&cursor, &context);

    // Unwind frames one by one, going up the frame stack.
    while (unw_step(&cursor) > 0)
    {
        unw_word_t offset, pc;
        unw_get_reg(&cursor, UNW_REG_IP, &pc);

        if (pc == 0)
        {
            break;
        }
        std::printf("0x%llx:", pc);

        char sym[1024];

        if (unw_get_proc_name(&cursor, sym, sizeof(sym), &offset) == 0)
        {
            int status;
            char* nameptr = sym;
            char* demangled = abi::__cxa_demangle(sym, nullptr, nullptr, &status);

            if (status == 0)
            {
                nameptr = demangled;
            }
            std::printf("(%s+0x%llx)\n", nameptr, offset);
            std::free(demangled);
        }
        else
        {
            std::printf(" -- error: unable to obtain symbol name for this frame\n");
        }
    }
}

void Debug::terminate_with_backtrace()
{
    try {
        auto e = std::current_exception();

        if (e)
        {
            std::rethrow_exception(e);
        }
    }
    catch (std::exception& e)
    {
        std::cout << std::string(52, '=') << std::endl;
        std::cout << "Uncaught exception: "<< e.what() << std::endl;
    }

    backtrace();
    exit(1);
}
