#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <vector>
#include "app_utils.hpp"
#include "ndarray.hpp"
#include "visit_struct.hpp"
#include "ufunc.hpp"
#include "physics.hpp"




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

struct shocktube_2d
{
    inline std::array<double, 5> operator()(std::array<double, 2> x) const
    {
        return x[0] + x[1] < 1.0 + 1e-10
        ? std::array<double, 5>{1.0, 0.0, 0.0, 0.0, 1.000}
        : std::array<double, 5>{0.1, 0.0, 0.0, 0.0, 0.125};
    }
};

struct cylindrical_explosion
{
    inline std::array<double, 5> operator()(std::array<double, 2> X) const
    {
        auto x = X[0] - 0.5;
        auto y = X[1] - 0.5;
        return x * x + y * y < 0.05
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
class Database2
{

public:
    using Index = std::tuple<int, int, int>;
    using Array = nd::array<double, 3>;

    Database2(int ni, int nj) : data(ni, nj, 5)
    {
    }

    void insert(Index index, Array data)
    {
        commit(index, data);
    }

    /**
     * Merge the given data into the database with the given weighting factor.
     * Setting rk_factor=0.0 corresponds to overwriting the existing data.
     */
    void commit(Index, nd::array<double, 3> new_data, double rk_factor=0.0)
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
    nd::array<double, 3> checkout(Index, int guard=0) const
    {
        auto _ = nd::axis::all();
        auto ni    = data.shape(0);
        auto nj    = data.shape(1);
        auto shape = std::array<int, 3>{ni + 2 * guard, nj + 2 * guard, 5};
        auto res   = nd::array<double, 3>(shape);

        res.select(_, _, 0) = 0.1;
        res.select(_, _, 1) = 0.0;
        res.select(_, _, 2) = 0.0;
        res.select(_, _, 3) = 0.0;
        res.select(_, _, 4) = 0.125;
        res.select(_|guard|ni+guard, _|guard|nj+guard, _) = data;

        return res;
    }

private:
    nd::array<double, 3> data;
};




// ============================================================================
enum class Field
{
    cell_centers,
    conserved,
};




// ============================================================================
class DatabaseFMR
{

public:
    using Index = std::tuple<int, int, Field>;
    using Array = nd::array<double, 3>;

    DatabaseFMR(int ni, int nj, std::map<Field, int> field_size)
    : ni(ni)
    , nj(nj)
    , field_size(field_size)
    {
    }

    /**
     * Insert a deep copy of the given array into the database at the given
     * patch index. Any existing data at that location is overwritten.
     */
    void insert(Index index, Array data)
    {
        patches[index].become(check_shape(data, index).copy());
    }

    /**
     * Erase any patch data at the given index.
     */
    auto erase(Index index)
    {
        return patches.erase(index);
    }

    /**
     * Merge the given data into the database at the given index, with the
     * given weighting factor. Setting rk_factor=0.0 corresponds to
     * overwriting the existing data.
     *
     * An exception is throws if a patch does not already exist at the given patch
     * index. Use insert to create a new patch.
     */
    void commit(Index index, Array data, double rk_factor=0.0)
    {
        auto target = patches.at(index);

        if (rk_factor == 0.0)
        {
            target = data;
        }
        else
        {
            auto average = ufunc::from([c=rk_factor] (double a, double b)
            {
                return a * (1 - c) + b * c;
            });
            target = average(data, target);            
        }
    }

    /**
     * Return a deep copy of the data at the patch index, padded with the
     * given number of guard zones. If no data exists at that index, or the
     * data has the wrong size, an exception is thrown.
     */
    Array checkout(Index index, int guard=0) const
    {
        auto _     = nd::axis::all();
        auto ng    = guard;
        auto shape = std::array<int, 3>{ni + 2 * ng, nj + 2 * ng, 5};
        auto res   = nd::array<double, 3>(shape);

        res.select(_, _, 0) = 0.1;
        res.select(_, _, 1) = 0.0;
        res.select(_, _, 2) = 0.0;
        res.select(_, _, 3) = 0.0;
        res.select(_, _, 4) = 0.125;
        res.select(_|ng|ni+ng, _|ng|nj+ng, _) = patches.at(index);

        auto i = std::get<0>(index);
        auto j = std::get<1>(index);
        auto f = std::get<2>(index);

        auto Ri = std::make_tuple(i + 1, j, f);
        auto Li = std::make_tuple(i - 1, j, f);
        auto Rj = std::make_tuple(i, j + 1, f);
        auto Lj = std::make_tuple(i, j - 1, f);

        if (patches.count(Ri))
        {
            res.select(_|ni+ng|ni+2*ng, _|ng|nj+ng, _) = patches.at(Ri).select(_|0|ng, _, _);
        }
        if (patches.count(Rj))
        {
            res.select(_|ng|ni+ng, _|nj+ng|nj+2*ng, _) = patches.at(Rj).select(_, _|0|ng, _);
        }
        if (patches.count(Li))
        {
            res.select(_|0|ng, _|ng|nj+ng, _) = patches.at(Li).select(_|ni-ng|ni, _, _);
        }
        if (patches.count(Lj))
        {
            res.select(_|ng|ni+ng, _|0|ng, _) = patches.at(Lj).select(_, _|nj-ng|nj, _);
        }

        return res;
    }

    std::map<Index, Array> all(Field which) const
    {
        auto res = std::map<Index, Array>();

        for (const auto& patch : patches)
        {
            if (std::get<2>(patch.first) == which)
            {
                res.insert(patch);
            }
        }
        return res;
    }

    auto begin() const
    {
        return patches.begin();
    }

    auto end() const
    {
        return patches.end();
    }

    auto size() const
    {
        return patches.size();
    }

private:
    Array check_shape(Array& array, Index index) const
    {
        auto which = std::get<2>(index);

        if (array.shape() != std::array<int, 3>{ni, nj, field_size.at(which)})
        {
            throw std::invalid_argument("input patch data has the wrong shape");
        }
        return array;
    }
    int ni;
    int nj;
    std::map<Field, int> field_size;
    std::map<Index, Array> patches;
};




// ============================================================================
std::string index_to_string(DatabaseFMR::Index index)
{
    auto i = std::get<0>(index);
    auto j = std::get<1>(index);
    auto f = std::string();

    switch (std::get<2>(index))
    {
        case Field::conserved: f = "conserved"; break;
        case Field::cell_centers: f = "cell_centers"; break;
    }
    return f + "/" + std::to_string(i) + "-" + std::to_string(j);
}

void write_database(const DatabaseFMR& database)
{
    auto parts = std::vector<std::string> {"data", "chkpt.0000.bt"};

    for (const auto& patch : database)
    {
        parts.push_back(index_to_string(patch.first) + ".nd");
        FileSystem::ensureParentDirectoryExists(FileSystem::joinPath(parts));
        tofile(patch.second, FileSystem::joinPath(parts));
        parts.pop_back();
    }
}




// ============================================================================
struct run_config
{
    std::string outdir = ".";
    double tfinal = 1.0;
    int ni = 100;
    int nj = 100;

    void print(std::ostream& os) const;
    static run_config from_dict(std::map<std::string, std::string> items);
    static run_config from_argv(int argc, const char* argv[]);
};

VISITABLE_STRUCT(run_config, outdir, tfinal, ni, nj);




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
auto advance_2d(nd::array<double, 3> U0, double dt, double dx, double dy)
{
    auto _ = nd::axis::all();

    auto update_formula = [dt,dx,dy] (std::array<double, 5> arg)
    {
        double u   = arg[0];
        double fli = arg[1];
        double fri = arg[2];
        double flj = arg[3];
        double frj = arg[4];
        return u + (fri - fli) * dt / dx + (frj - flj) * dt / dy;
    };

    auto gradient_est = ufunc::from(gradient_plm(1.5));
    auto advance_cons = ufunc::nfrom(update_formula);
    auto cons_to_prim = ufunc::vfrom(newtonian_hydro::cons_to_prim());
    auto godunov_flux_i = ufunc::vfrom(newtonian_hydro::riemann_hlle({1, 0, 0}));
    auto godunov_flux_j = ufunc::vfrom(newtonian_hydro::riemann_hlle({0, 1, 0}));
    auto extrap_l = ufunc::from(add_scaled(-0.5));
    auto extrap_r = ufunc::from(add_scaled(+0.5));

    auto mi = U0.shape(0);
    auto mj = U0.shape(1);
    auto P0 = cons_to_prim(U0);

    auto Fhi = [&] ()
    {
        auto Pa = P0.select(_|0|mi-2, _|2|mj-2, _);
        auto Pb = P0.select(_|1|mi-1, _|2|mj-2, _);
        auto Pc = P0.select(_|2|mi-0, _|2|mj-2, _);
        auto Gb = gradient_est(Pa, Pb, Pc);
        auto Pl = extrap_l(Pb, Gb);
        auto Pr = extrap_r(Pb, Gb);
        auto Fh = godunov_flux_i(Pr.take<0>(_|0|mi-3), Pl.take<0>(_|1|mi-2));
        return Fh;
    }();

    auto Fhj = [&] ()
    {
        auto Pa = P0.select(_|2|mi-2, _|0|mj-2, _);
        auto Pb = P0.select(_|2|mi-2, _|1|mj-1, _);
        auto Pc = P0.select(_|2|mi-2, _|2|mj-0, _);
        auto Gb = gradient_est(Pa, Pb, Pc);
        auto Pl = extrap_l(Pb, Gb);
        auto Pr = extrap_r(Pb, Gb);
        auto Fh = godunov_flux_j(Pr.take<1>(_|0|mj-3), Pl.take<1>(_|1|mj-2));
        return Fh;
    }();

    auto U1 = advance_cons(
        std::array<nd::array<double, 3>, 5>{
            U0 .select(_|2|mi-2, _|2|mj-2, _),
            Fhi.take<0>(_|1|mi-3),
            Fhi.take<0>(_|0|mi-4),
            Fhj.take<1>(_|1|mj-3),
            Fhj.take<1>(_|0|mj-4)});

    return U1;
}




// ============================================================================
nd::array<double, 3> mesh_vertices(
    int ni, int nj,
    double x0=0.0, double x1=1.0,
    double y0=0.0, double y1=1.0)
{
    auto X = nd::array<double, 3>(ni + 1, nj + 1, 2);

    for (int i = 0; i < ni + 1; ++i)
    {
        for (int j = 0; j < nj + 1; ++j)
        {
            X(i, j, 0) = x0 + (x1 - x0) * i / ni;
            X(i, j, 1) = y0 + (y1 - y0) * j / nj;
        }
    }
    return X;
}

nd::array<double, 3> mesh_cell_centers(nd::array<double, 3> verts)
{
    auto _ = nd::axis::all();
    auto ni = verts.shape(0) - 1;
    auto nj = verts.shape(1) - 1;

    return (
    verts.select(_|0|ni+0, _|0|nj+0, _) +
    verts.select(_|0|ni+0, _|1|nj+1, _) +
    verts.select(_|1|ni+1, _|0|nj+0, _) +
    verts.select(_|1|ni+1, _|1|nj+1, _)) * 0.25;
}




// ============================================================================
int main_2d(int argc, const char* argv[])
{
    auto cfg = run_config::from_argv(argc, argv);
    cfg.print(std::cout);

    auto wall = 0.0;
    auto ni   = cfg.ni;
    auto nj   = cfg.nj;
    auto iter = 0;
    auto t    = 0.0;
    auto dx   = 1.0 / ni;
    auto dy   = 1.0 / nj;
    auto dt   = std::min(dx, dy) * 0.125;

    auto field_size = std::map<Field, int> {
        {Field::conserved, 5},
        {Field::cell_centers, 2},
    };

    auto initial_data = ufunc::vfrom(cylindrical_explosion());
    auto prim_to_cons = ufunc::vfrom(newtonian_hydro::prim_to_cons());
    auto database = DatabaseFMR(ni, nj, field_size);

    auto Ni = 2;
    auto Nj = 2;

    for (int i = 0; i < Ni; ++i)
    {
        for (int j = 0; j < Nj; ++j)
        {
            double x0 = double(i + 0) / Ni;
            double x1 = double(i + 1) / Ni;
            double y0 = double(j + 0) / Nj;
            double y1 = double(j + 1) / Nj;

            auto x_verts = mesh_vertices(ni, nj, x0, x1, y0, y1);
            auto x_cells = mesh_cell_centers(x_verts);
            auto U = prim_to_cons(initial_data(x_cells));

            database.insert(std::make_tuple(i, j, Field::cell_centers), x_cells);
            database.insert(std::make_tuple(i, j, Field::conserved), U);
        }
    }

    std::map<DatabaseFMR::Index, DatabaseFMR::Array> results;

    while (t < cfg.tfinal)
    {
        auto timer = Timer();


        for (const auto& patch : database.all(Field::conserved))
        {
            auto U = database.checkout(patch.first, 2);
            results[patch.first].become(advance_2d(U, dt, dx, dy));
        }
        for (const auto& res : results)
        {
            database.commit(res.first, res.second, 0.0);
        }


        t    += dt;
        iter += 1;
        wall += timer.seconds();

        std::printf("[%04d] t=%3.2lf kzps=%3.2lf\n", iter, t, ni * nj * database.size() / 1e3 / timer.seconds());
    }

    std::printf("average kzps=%f\n", ni * nj * database.size() / 1e3 / wall * iter);

    write_database(database);
    return 0;
}




// ============================================================================
int main(int argc, const char* argv[])
{
    std::set_terminate(Debug::terminate_with_backtrace);
    return main_2d(argc, argv);
}




// ============================================================================
// class Database
// {

// public:
//     Database(int ni) : data(ni, 5)
//     {
//     }

//     /**
//      * Merge the given data into the database with the given weighting factor.
//      * Setting rk_factor=0.0 corresponds to overwriting the existing data.
//      */
//     void commit(nd::array<double, 2> new_data, double rk_factor=0.0)
//     {
//         auto average = ufunc::from([c=rk_factor] (double a, double b)
//         {
//             return a * (1 - c) + b * c;
//         });

//         // The only reason to check if rk_factor is zero for performance. The
//         // outcome would be the same.

//         if (rk_factor == 0.0)
//         {
//             data = new_data;
//         }
//         else
//         {
//             data = average(new_data, data);            
//         }
//     }

//     /**
//      * Return a copy of the current data, padded with the given number of
//      * guard zones.
//      */
//     nd::array<double, 2> checkout(int guard=0) const
//     {
//         auto _ = nd::axis::all();
//         auto ni    = data.shape(0);
//         auto shape = std::array<int, 2>{ni + 2 * guard, 5};
//         auto res   = nd::array<double, 2>(shape);

//         res.select(_|guard|ni+guard, _) = data;

//         // Temporarily we're using a zero-gradient BC:

//         for (int i = 0; i < guard; ++i)
//         {
//             for (int q = 0; q < 5; ++q)
//             {
//                 res(i, q) = data(0, q);
//                 res(i + ni + guard, q) = data(ni - 1, q);
//             }
//         }
//         return res;
//     }

// private:
//     nd::array<double, 2> data;
// };




// ============================================================================
// auto advance_1d(nd::array<double, 2> U0, double dt, double dx)
// {
//     auto _ = nd::axis::all();

//     auto gradient_est = ufunc::from(gradient_plm(1.5));
//     auto advance_cons = ufunc::from(add_diff_scaled(-dt / dx));
//     auto cons_to_prim = ufunc::vfrom(newtonian_hydro::cons_to_prim());
//     auto godunov_flux = ufunc::vfrom(newtonian_hydro::riemann_hlle({1, 0, 0}));
//     auto extrap_l = ufunc::from(add_scaled(-0.5));
//     auto extrap_r = ufunc::from(add_scaled(+0.5));

//     auto mi = U0.shape(0);
//     auto P0 = cons_to_prim(U0);

//     auto Pa = P0.take<0>(_|0|mi-2);
//     auto Pb = P0.take<0>(_|1|mi-1);
//     auto Pc = P0.take<0>(_|2|mi-0);
//     auto Gb = gradient_est(Pa, Pb, Pc);
//     auto Pl = extrap_l(Pb, Gb);
//     auto Pr = extrap_r(Pb, Gb);
//     auto Fh = godunov_flux(Pr.take<0>(_|0|mi-3), Pl.take<0>(_|1|mi-2));
//     auto U1 = advance_cons(
//         U0.take<0>(_|2|mi-2),
//         Fh.take<0>(_|1|mi-3),
//         Fh.take<0>(_|0|mi-4));

//     return U1;
// }




// ============================================================================
// int main_1d(int argc, const char* argv[])
// {
//     auto cfg = run_config::from_argv(argc, argv);
//     cfg.print(std::cout);

//     auto _ = nd::axis::all();
//     auto wall = 0.0;
//     auto ni   = cfg.ni;
//     auto iter = 0;
//     auto t    = 0.0;
//     auto dx   = 1.0 / ni;
//     auto dt   = dx * 0.125;

//     auto initial_data = ufunc::vfrom(shocktube());
//     auto prim_to_cons = ufunc::vfrom(newtonian_hydro::prim_to_cons());

//     auto x_verts = nd::linspace<double>(0.0, 1.0, ni+1).reshape(ni+1, 1);
//     auto x_cells = (x_verts.take<0>(_|1|ni+1) + x_verts.take<0>(_|0|ni)) * 0.5;
//     auto x_delta = (x_verts.take<0>(_|1|ni+1) - x_verts.take<0>(_|0|ni)) * 1.0;

//     auto database = Database(ni);
//     database.commit(prim_to_cons(initial_data(x_cells)));

//     while (t < cfg.tfinal)
//     {
//         auto timer = Timer();

//         database.commit(advance_1d(database.checkout(2), dt, dx), 0.0);
//         database.commit(advance_1d(database.checkout(2), dt, dx), 0.5);

//         t    += dt;
//         iter += 1;
//         wall += timer.seconds();

//         std::printf("[%04d] t=%3.2lf kzps=%3.2lf\n", iter, t, ni / 1e3 / timer.seconds());
//     }

//     std::printf("average kzps=%f\n", ni / 1e3 / wall * iter);

//     auto P = prim_to_cons(database.checkout());
//     tofile(x_cells.select(_, 0), "x_cells.nd");
//     tofile(P.select(_, 0), "rho.nd");

//     return 0;
// }
