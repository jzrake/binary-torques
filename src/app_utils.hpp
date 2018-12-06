#include <map>
#include <fstream>
#include <iostream>




// ============================================================================
template<typename Writeable>
void tofile(const Writeable& writeable, const std::string& fname)
{
    std::ofstream outfile(fname, std::ofstream::binary | std::ios::out);

    if (! outfile.is_open())
    {
        throw std::invalid_argument("file " + fname + " could not be opened for writing");
    }
    auto s = writeable.dumps();
    outfile.write(s.data(), s.size());
    outfile.close();
}




// ============================================================================
template <class T, std::size_t N>
std::ostream& operator<<(std::ostream& o, const std::array<T, N>& arr)
{
    std::copy(arr.cbegin(), arr.cend(), std::ostream_iterator<T>(o, " "));
    return o;
}




// ============================================================================
class Timer
{
public:
    Timer() : instantiated(std::clock())
    {
    }
    double seconds() const
    {
        return double (std::clock() - instantiated) / CLOCKS_PER_SEC;
    }
private:
    std::clock_t instantiated;
};




// ============================================================================
namespace cmdline 
{
    std::map<std::string, std::string> parse_keyval(int argc, const char* argv[])
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

    template <typename T>
    void set_from_string(std::string source, T& value);

    template <>
    void set_from_string<std::string>(std::string source, std::string& value)
    {
        value = source;
    }

    template <>
    void set_from_string<int>(std::string source, int& value)
    {
        value = std::stoi(source);
    }

    template <>
    void set_from_string<double>(std::string source, double& value)
    {
        value = std::stod(source);
    }
}




// ============================================================================
#include <libunwind.h>
#include <cxxabi.h>

void backtrace()
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

void terminate_with_backtrace()
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