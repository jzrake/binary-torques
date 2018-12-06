#pragma once
#include "ndarray.hpp"




// ============================================================================
namespace ufunc {
    namespace detail
    {
        template <typename T>
        struct function_traits : public function_traits<decltype(&T::operator())>
        {
        };

        template <typename ClassType, typename ReturnType, typename... Args>
        struct function_traits<ReturnType(ClassType::*)(Args...) const>
        {
            typedef ReturnType result_type;
            enum { arity = sizeof...(Args) };

            template <size_t i> struct arg
            {
                typedef typename std::tuple_element<i, std::tuple<Args...>>::type type;
            };
        };

        template <typename T> auto drop_last(const T& container)
        {
            auto res = std::array<typename T::value_type, T().size() - 1>();

            for (std::size_t i = 0; i < res.size(); ++i)
            {
                res[i] = container[i];
            }
            return res;
        }

        template <typename T, typename I> auto replace_last(const T& container, I i)
        {
            auto res = container;
            res[res.size() - 1] = i;
            return res;
        }

        template <std::size_t R, typename Iter> auto take(Iter& iter)
        {
            auto res = std::array<typename Iter::value_type, R>();

            for (std::size_t i = 0; i < R; ++i)
            {
                res[i] = *iter++;
            }
            return res;
        }

        template <typename, typename> struct Ufunc1;
        template <typename, typename> struct Ufunc2;
        template <typename, typename> struct Ufunc3;
        template <typename Callable, typename T, std::size_t Arity> struct Ufuncn;

        template <typename, typename, std::size_t, std::size_t> struct Vfunc1;
        template <typename, typename, std::size_t, std::size_t, std::size_t> struct Vfunc2;
        template <typename, typename, std::size_t, std::size_t, std::size_t, std::size_t> struct Vfunc3;
    }

    template<typename Callable> auto from(Callable f, typename std::enable_if<detail::function_traits<Callable>::arity == 1>::type* = nullptr);
    template<typename Callable> auto from(Callable f, typename std::enable_if<detail::function_traits<Callable>::arity == 2>::type* = nullptr);
    template<typename Callable> auto from(Callable f, typename std::enable_if<detail::function_traits<Callable>::arity == 3>::type* = nullptr);
    template<typename Callable> auto nfrom(Callable f, typename std::enable_if<detail::function_traits<Callable>::arity == 1>::type* = nullptr);

    template<typename Callable> auto vfrom(Callable f, typename std::enable_if<detail::function_traits<Callable>::arity == 1>::type* = nullptr);
    template<typename Callable> auto vfrom(Callable f, typename std::enable_if<detail::function_traits<Callable>::arity == 2>::type* = nullptr);
    template<typename Callable> auto vfrom(Callable f, typename std::enable_if<detail::function_traits<Callable>::arity == 3>::type* = nullptr);
}




// ============================================================================
template <typename Callable, typename T>
struct ufunc::detail::Ufunc1
{
    Ufunc1(Callable F) : F(F) {}

    template <int Rank>
    inline auto operator()(const nd::array<T, Rank>& A) const
    {
        auto R = nd::array<T, Rank>(A.shape());
        auto a = A.begin();
        auto r = R.begin();

        while (r != R.end())
        {
            *r++ = F(*a++);
        }
        return R;
    }
    Callable F;
};




// ============================================================================
template <typename Callable, typename T>
struct ufunc::detail::Ufunc2
{
    Ufunc2(Callable F) : F(F) {}

    template <int Rank>
    inline auto operator()(const nd::array<T, Rank>& A, const nd::array<T, Rank>& B) const
    {
        if (A.shape() != B.shape())
        {
            throw std::invalid_argument("input arrays have different shapes");
        }

        auto R = nd::array<T, Rank>(A.shape());
        auto a = A.begin();
        auto b = B.begin();
        auto r = R.begin();

        while (r != R.end())
        {
            *r++ = F(*a++, *b++);
        }
        return R;
    }
    Callable F;
};




// ============================================================================
template <typename Callable, typename T>
struct ufunc::detail::Ufunc3
{
    Ufunc3(Callable F) : F(F) {}

    template <int Rank>
    inline auto operator()(const nd::array<T, Rank>& A, const nd::array<T, Rank>& B, const nd::array<T, Rank>& C) const
    {
        if (A.shape() != B.shape() || B.shape() != C.shape())
        {
            throw std::invalid_argument("input arrays have different shapes");
        }

        auto R = nd::array<T, Rank>(A.shape());
        auto a = A.begin();
        auto b = B.begin();
        auto c = C.begin();
        auto r = R.begin();

        while (r != R.end())
        {
            *r++ = F(*a++, *b++, *c++);
        }
        return R;
    }
    Callable F;
};




// ============================================================================
template <typename Callable, typename T, std::size_t Arity>
struct ufunc::detail::Ufuncn
{
    Ufuncn(Callable F) : F(F) {}

    template <int Rank>
    inline auto operator()(const std::array<nd::array<T, Rank>, Arity>& args) const
    {
        for (std::size_t n = 0; n < Arity - 1; ++n)
        {
            if (args[n].shape() != args[n + 1].shape())
            {
                throw std::invalid_argument("input arrays have different shapes");
            }            
        }

        auto R = nd::array<T, Rank>(args[0].shape());
        auto r = R.begin();
        auto iters = std::array<typename nd::array<T, Rank>::const_iterator, Arity>();

        for (std::size_t n = 0; n < Arity; ++n)
        {
            iters[n] = args[n].begin();
        }

        while (r != R.end())
        {
            auto a = std::array<T, Arity>();

            for (std::size_t n = 0; n < Arity; ++n)
            {
                a[n] = *iters[n]++;
            }
            *r++ = F(a);
        }
        return R;
    }
    Callable F;
};




// ============================================================================
template <typename Callable, typename T, std::size_t ResSize, std::size_t ArgSize>
struct ufunc::detail::Vfunc1
{
    Vfunc1(Callable F) : F(F) {}

    template <int Rank>
    inline auto operator()(const nd::array<T, Rank>& A) const
    {
        if (A.shape(Rank - 1) != ArgSize)
        {
            throw std::invalid_argument("input array A has wrong last axis size");
        }

        auto R = nd::array<T, Rank>(detail::replace_last(A.shape(), ResSize));
        auto a = A.begin();
        auto r = R.begin();

        while (r != R.end())
        {
            const auto avec = detail::take<ArgSize>(a);
            const auto rvec = F(avec);
            for (std::size_t n = 0; n < ResSize; ++n) *r++ = rvec[n];
        }
        return R;
    }
    Callable F;
};




// ============================================================================
template <typename Callable, typename T, std::size_t ResSize, std::size_t ArgSize1, std::size_t ArgSize2>
struct ufunc::detail::Vfunc2
{
    Vfunc2(Callable F) : F(F) {}

    template <int Rank>
    inline auto operator()(const nd::array<T, Rank>& A, const nd::array<T, Rank>& B) const
    {
        if (A.shape(Rank - 1) != ArgSize1)
        {
            throw std::invalid_argument("input array A has wrong last axis size");
        }
        if (B.shape(Rank - 1) != ArgSize2)
        {
            throw std::invalid_argument("input array B has wrong last axis size");
        }
        if (detail::drop_last(A.shape()) != drop_last(B.shape()))
        {
            throw std::invalid_argument("input arrays have different shapes other than last-axis");
        }

        auto R = nd::array<T, Rank>(detail::replace_last(A.shape(), ResSize));
        auto a = A.begin();
        auto b = B.begin();
        auto r = R.begin();

        while (r != R.end())
        {
            const auto avec = detail::take<ArgSize1>(a);
            const auto bvec = detail::take<ArgSize2>(b);
            const auto rvec = F(avec, bvec);
            for (std::size_t n = 0; n < ResSize; ++n) *r++ = rvec[n];
        }
        return R;
    }
    Callable F;
};




// ============================================================================
template <typename Callable, typename T, std::size_t ResSize, std::size_t ArgSize1, std::size_t ArgSize2, std::size_t ArgSize3>
struct ufunc::detail::Vfunc3
{
    Vfunc3(Callable F) : F(F) {}

    template <int Rank>
    inline auto operator()(const nd::array<T, Rank>& A, const nd::array<T, Rank>& B, const nd::array<T, Rank>& C) const
    {
        if (A.shape(Rank - 1) != ArgSize1)
        {
            throw std::invalid_argument("input array A has wrong last-axis size");
        }
        if (B.shape(Rank - 1) != ArgSize2)
        {
            throw std::invalid_argument("input array B has wrong last-axis size");
        }
        if (C.shape(Rank - 1) != ArgSize3)
        {
            throw std::invalid_argument("input array B has wrong last-axis size");
        }
        if (detail::drop_last(A.shape()) != drop_last(B.shape()) ||
            detail::drop_last(B.shape()) != drop_last(C.shape()))
        {
            throw std::invalid_argument("input arrays have different shapes other than last-axis");
        }

        auto R = nd::array<T, Rank>(detail::replace_last(A.shape(), ResSize));
        auto a = A.begin();
        auto b = B.begin();
        auto c = C.begin();
        auto r = R.begin();

        while (r != R.end())
        {
            const auto avec = detail::take<ArgSize1>(a);
            const auto bvec = detail::take<ArgSize2>(b);
            const auto cvec = detail::take<ArgSize3>(c);
            const auto rvec = F(avec, bvec, cvec);
            for (std::size_t n = 0; n < ResSize; ++n) *r++ = rvec[n];
        }
        return R;
    }
    Callable F;
};




// ============================================================================
template<typename Callable>
auto ufunc::from(Callable f, typename std::enable_if<detail::function_traits<Callable>::arity == 1>::type*)
{
    return detail::Ufunc1<Callable, double>(f);
}

template<typename Callable>
auto ufunc::from(Callable f, typename std::enable_if<detail::function_traits<Callable>::arity == 2>::type*)
{
    return detail::Ufunc2<Callable, double>(f);
}

template<typename Callable>
auto ufunc::from(Callable f, typename std::enable_if<detail::function_traits<Callable>::arity == 3>::type*)
{
    return detail::Ufunc3<Callable, double>(f);
}

template<typename Callable>
auto ufunc::nfrom(Callable f, typename std::enable_if<detail::function_traits<Callable>::arity == 1>::type*)
{
    constexpr std::size_t ArgSize = typename detail::function_traits<Callable>::template arg<0>::type().size();
    return detail::Ufuncn<Callable, double, ArgSize>(f);
}

template<typename Callable>
auto ufunc::vfrom(Callable f, typename std::enable_if<detail::function_traits<Callable>::arity == 1>::type*)
{
    constexpr std::size_t ResSize = typename detail::function_traits<Callable>::result_type().size();
    constexpr std::size_t ArgSize = typename detail::function_traits<Callable>::template arg<0>::type().size();
    return detail::Vfunc1<Callable, double, ResSize, ArgSize>(f);
}

template<typename Callable>
auto ufunc::vfrom(Callable f, typename std::enable_if<detail::function_traits<Callable>::arity == 2>::type*)
{
    constexpr std::size_t ResSize  = typename detail::function_traits<Callable>::result_type().size();
    constexpr std::size_t ArgSize1 = typename detail::function_traits<Callable>::template arg<0>::type().size();
    constexpr std::size_t ArgSize2 = typename detail::function_traits<Callable>::template arg<1>::type().size();
    return detail::Vfunc2<Callable, double, ResSize, ArgSize1, ArgSize2>(f);
}

template<typename Callable>
auto ufunc::vfrom(Callable f, typename std::enable_if<detail::function_traits<Callable>::arity == 3>::type*)
{
    constexpr std::size_t ResSize  = typename detail::function_traits<Callable>::result_type().size();
    constexpr std::size_t ArgSize1 = typename detail::function_traits<Callable>::template arg<0>::type().size();
    constexpr std::size_t ArgSize2 = typename detail::function_traits<Callable>::template arg<1>::type().size();
    constexpr std::size_t ArgSize3 = typename detail::function_traits<Callable>::template arg<2>::type().size();
    return detail::Vfunc3<Callable, double, ResSize, ArgSize1, ArgSize2, ArgSize3>(f);
}
