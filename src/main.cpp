#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <map>
#include "ndarray.hpp"
#include "visit_struct.hpp"
#include "ufunc.hpp"
#include "physics.hpp"




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
#define MIN3ABS(a, b, c) std::min(std::min(std::fabs(a), std::fabs(b)), std::fabs(c))
#define SGN(x) std::copysign(1, x)

static double minmod(double ul, double u0, double ur, double theta)
{
    const double a = theta * (u0 - ul);
    const double b =   0.5 * (ur - ul);
    const double c = theta * (ur - u0);
    return 0.25 * std::fabs(SGN(a) + SGN(b)) * (SGN(a) + SGN(c)) * MIN3ABS(a, b, c);
}




// ============================================================================
struct gradient_plm
{
    gradient_plm(double theta) : theta(theta) {}

    double inline operator()(double a, double b, double c) const
    {
        return minmod(a, b, c, theta);
    }
    double theta;
};

struct add_scaled
{
    add_scaled(double c) : c(c) {}

    inline double operator()(double a, double b) const
    {
        return a + b * c;
    }
    double c;
};

struct add_diff_scaled
{
    add_diff_scaled(double c) : c(c) {}

    inline double operator()(double a, double fl, double fr) const
    {
        return a + (fl - fr) * c;
    }
    double c;
};

struct shocktube
{
    inline std::array<double, 5> operator()(std::array<double, 1> x) const
    {
        return x[0] < 0.5
        ? std::array<double, 5>{1.0, 0.0, 0.0, 0.0, 1.000}
        : std::array<double, 5>{0.1, 0.0, 0.0, 0.0, 0.125};
    }
};

struct gaussian
{
    inline std::array<double, 5> operator()(std::array<double, 1> x) const
    {
        auto d = 1 + std::exp(-(x[0] - 0.5) * (x[0] - 0.5) / 0.01);
        return std::array<double, 5>{d, 0.0, 0.0, 0.0, 1.000};
    }
};





// ============================================================================
class Database
{

public:
    Database(int ni) : data(ni, 5)
    {
    }

    /**
     * Merge the given data into the database with the given weighting factor.
     * Setting rk_factor=0.0 corresponds to overwriting the existing data.
     */
    void commit(nd::array<double, 2> new_data, double rk_factor=0.0)
    {
        auto average = ufunc::from([c=rk_factor] (double a, double b)
        {
            return a * (1 - c) + b * c;
        });

        // The only reason to check if rk_factor is zero for performance. The
        // outcome would be the same.

        if (rk_factor == 0.0)
        {
            data = new_data;
        }
        else
        {
            data = average(new_data, data);            
        }
    }

    /**
     * Return a copy of the current data, padded with the given number of
     * guard zones.
     */
    nd::array<double, 2> checkout(int guard=0) const
    {
        auto _ = nd::axis::all();
        auto ni    = data.shape(0);
        auto shape = std::array<int, 2>{ni + 2 * guard, 5};
        auto res   = nd::array<double, 2>(shape);

        res.select(_|guard|ni+guard, _) = data;

        // Temporarily we're using a zero-gradient BC:

        for (int i = 0; i < guard; ++i)
        {
            for (int q = 0; q < 5; ++q)
            {
                res(i, q) = data(0, q);
                res(i + ni + guard, q) = data(ni - 1, q);
            }
        }
        return res;
    }

private:
    nd::array<double, 2> data;
};




// ============================================================================
auto advance(nd::array<double, 2> U0, double dt, double dx)
{
    auto _ = nd::axis::all();

    auto gradient_est = ufunc::from(gradient_plm(1.5));
    auto advance_cons = ufunc::from(add_diff_scaled(-dt / dx));
    auto cons_to_prim = ufunc::vfrom(newtonian_hydro::cons_to_prim());
    auto godunov_flux = ufunc::vfrom(newtonian_hydro::riemann_hlle());
    auto extrap_l = ufunc::from(add_scaled(-0.5));
    auto extrap_r = ufunc::from(add_scaled(+0.5));

    auto mi = U0.shape(0);
    auto P0 = cons_to_prim(U0);

    auto Pa = P0.take<0>(_|0|mi-2);
    auto Pb = P0.take<0>(_|1|mi-1);
    auto Pc = P0.take<0>(_|2|mi-0);
    auto Gb = gradient_est(Pa, Pb, Pc);
    auto Pl = extrap_l(Pb, Gb);
    auto Pr = extrap_r(Pb, Gb);
    auto Fh = godunov_flux(Pr.take<0>(_|0|mi-3), Pl.take<0>(_|1|mi-2));
    auto U1 = advance_cons(
        U0.take<0>(_|2|mi-2),
        Fh.take<0>(_|1|mi-3),
        Fh.take<0>(_|0|mi-4));

    return U1;
}




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
struct run_config
{
    std::string outdir = ".";
    double tfinal = 1.0;


    void print(std::ostream& os) const;
    static run_config from_dict(std::map<std::string, std::string> items);
    static run_config from_argv(int argc, const char* argv[]);
};

VISITABLE_STRUCT(run_config, outdir, tfinal);




// ============================================================================
run_config run_config::from_dict(std::map<std::string, std::string> items)
{
    run_config cfg;

    visit_struct::for_each(cfg, [items] (const char* name, auto& value)
    {
        if (items.find(name) != items.end())
        {
            cmdline::set_from_string(items.at(name), value);
        }
    });
    return cfg;
}

run_config run_config::from_argv(int argc, const char* argv[])
{
    return from_dict(cmdline::parse_keyval(argc, argv));
}

void run_config::print(std::ostream& os) const
{
    using std::left;
    using std::setw;
    using std::setfill;
    using std::showpos;
    const int W = 24;

    os << "\n" << std::string(52, '=') << "\n";

    std::ios orig(nullptr);
    orig.copyfmt(os);

    visit_struct::for_each(*this, [&os] (std::string name, auto& value)
    {
        os << left << setw(W) << setfill('.') << name + " " << " " << value << "\n";
    });

    os << std::string(52, '=') << "\n\n";
    os.copyfmt(orig);
}




// ============================================================================
int main(int argc, const char* argv[])
{
    auto cfg = run_config::from_argv(argc, argv);
    cfg.print(std::cout);

    auto _ = nd::axis::all();
    auto wall = 0.0;
    auto ni   = 1600;
    auto iter = 0;
    auto t    = 0.0;
    auto dx   = 1.0 / ni;
    auto dt   = dx * 0.125;

    auto initial_data = ufunc::vfrom(shocktube());
    auto prim_to_cons = ufunc::vfrom(newtonian_hydro::prim_to_cons());

    auto x_verts = nd::linspace<double>(0.0, 1.0, ni+1).reshape(ni+1, 1);
    auto x_cells = (x_verts.take<0>(_|1|ni+1) + x_verts.take<0>(_|0|ni)) * 0.5;
    auto x_delta = (x_verts.take<0>(_|1|ni+1) - x_verts.take<0>(_|0|ni)) * 1.0;

    auto database = Database(ni);
    database.commit(prim_to_cons(initial_data(x_cells)));

    while (t < cfg.tfinal)
    {
        auto timer = Timer();

        database.commit(advance(database.checkout(2), dt, dx), 0.0);
        database.commit(advance(database.checkout(2), dt, dx), 0.5);

        t    += dt;
        iter += 1;
        wall += timer.seconds();

        std::printf("[%04d] t=%3.2lf kzps=%3.2lf\n", iter, t, ni / 1e3 / timer.seconds());
    }

    std::printf("average kzps=%f\n", ni / 1e3 / wall * iter);

    auto P = prim_to_cons(database.checkout());
    tofile(x_cells.select(_, 0), "x_cells.nd");
    tofile(P.select(_, 0), "rho.nd");

    return 0;
}

