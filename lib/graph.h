#ifndef GRAPH_H
#define GRAPH_H

#include <array>
#include <cstdint>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <variant>
#include <vector>

namespace jsc {

namespace detail {
template <class T> using remove_cvref_t = std::remove_cv_t<std::remove_reference_t<T>>;
}

template <class TStr = std::string>
using AttrVariant = std::variant<
        std::array<std::int64_t, 4>,
        std::array<double, 4>,
        TStr,
        std::vector<std::int64_t>,
        std::vector<double>>;

template <class TStr = std::string>
class AttrValue {
public:
    using variant_type = AttrVariant<TStr>;

    AttrValue() = default;

    template <class TVal,
              class = std::enable_if_t<!std::is_same_v<detail::remove_cvref_t<TVal>, AttrValue>>>
    explicit AttrValue(TVal &&v) { init(std::forward<TVal>(v)); }

    AttrValue(const AttrValue &) = default;
    AttrValue(AttrValue &&) noexcept = default;
    AttrValue &operator=(const AttrValue &) = default;
    AttrValue &operator=(AttrValue &&) noexcept = default;

    template <class T>       T &get()       { return std::get<T>(_v); }
    template <class T> const T &get() const { return std::get<T>(_v); }

    template <class T>       T &at(std::size_t i);
    template <class T> const T &at(std::size_t i) const { return const_cast<AttrValue*>(this)->at<T>(i); }

    TStr       &str()       { return std::get<TStr>(_v); }
    const TStr &str() const { return std::get<TStr>(_v); }

    bool is_string()  const { return std::holds_alternative<TStr>(_v); }
    bool is_int_arr() const { return std::holds_alternative<std::array<std::int64_t,4>>(_v); }
    bool is_dbl_arr() const { return std::holds_alternative<std::array<double,4>>(_v); }

private:
    variant_type _v{};

    template <class TVal>
    void init(TVal &&val);

    template <class T, class Arr>
    static Arr scalar_to_arr(T x) { Arr a{}; a[0] = static_cast<typename Arr::value_type>(x); return a; }
};

template <class TStr>
template <class T>
T &AttrValue<TStr>::at(std::size_t i) {
    static_assert(std::is_same_v<T, std::int64_t> || std::is_same_v<T, double>,
                  "AttrValue::at<T> → только int64_t или double");
    if (std::holds_alternative<std::array<T,4>>(_v))  return std::get<std::array<T,4>>(_v).at(i);
    if (std::holds_alternative<std::vector<T>>(_v))   return std::get<std::vector<T>>(_v).at(i);
    throw std::bad_variant_access{};
}

template <class TStr>
template <class TVal>
void AttrValue<TStr>::init(TVal &&val) {
    using raw_t = detail::remove_cvref_t<TVal>;
    if constexpr (std::is_same_v<raw_t, variant_type>)      _v = std::forward<TVal>(val);
    else if constexpr (std::is_integral_v<raw_t>)           _v = scalar_to_arr<raw_t, std::array<std::int64_t,4>>(val);
    else if constexpr (std::is_floating_point_v<raw_t>)     _v = scalar_to_arr<raw_t, std::array<double,4>>(val);
    else                                                    _v = std::forward<TVal>(val);
}

template <class TStr = std::string>
struct Attr {
    TStr name{};
    AttrValue<TStr> value{};

    Attr() = default;
    Attr(TStr n, AttrValue<TStr> v) : name(std::move(n)), value(std::move(v)) {}
};

template <class TStr = std::string>
class Widget {
public:
    explicit Widget(const TStr &nm = {}) : _name(nm) {}
    explicit Widget(AttrValue<TStr> nm)  : _name(std::move(nm)) {}

    void            set_attr(TStr k, AttrValue<TStr> v) { _dyn.emplace(std::move(k), std::move(v)); }
    AttrValue<TStr>*get_attr(const TStr &k);
    const AttrValue<TStr>*get_attr(const TStr &k) const;

    Widget &add_child(Widget w) { _children.emplace_back(std::move(w)); return _children.back(); }
    const std::vector<Widget>& children() const { return _children; }

private:
    AttrValue<TStr> _name;
    std::unordered_map<TStr, AttrValue<TStr>> _dyn;
    std::vector<Widget> _children;
};

template <class TStr>
AttrValue<TStr>* Widget<TStr>::get_attr(const TStr &k) {
    auto it = _dyn.find(k);
    return it == _dyn.end() ? nullptr : &it->second;
}
template <class TStr>
const AttrValue<TStr>* Widget<TStr>::get_attr(const TStr &k) const {
    auto it = _dyn.find(k);
    return it == _dyn.end() ? nullptr : &it->second;
}

template <class TStr = std::string>
class Node {
public:
    explicit Node(const TStr &nm = {}) : _name(nm) {}
    explicit Node(AttrValue<TStr> nm)   : _name(std::move(nm)) {}

    Widget<TStr> &add_widget(Widget<TStr> w) { _widgets.emplace_back(std::move(w)); return _widgets.back(); }
    const std::vector<Widget<TStr>>& widgets() const { return _widgets; }

    void            set_attr(TStr k, AttrValue<TStr> v) { _dyn.emplace(std::move(k), std::move(v)); }
    AttrValue<TStr>*get_attr(const TStr &k);
    const AttrValue<TStr>*get_attr(const TStr &k) const;

private:
    AttrValue<TStr> _name;
    std::unordered_map<TStr, AttrValue<TStr>> _dyn;
    std::vector<Widget<TStr>> _widgets;
};

template <class TStr>
AttrValue<TStr>* Node<TStr>::get_attr(const TStr &k) {
    auto it = _dyn.find(k);
    return it == _dyn.end() ? nullptr : &it->second;
}
template <class TStr>
const AttrValue<TStr>* Node<TStr>::get_attr(const TStr &k) const {
    auto it = _dyn.find(k);
    return it == _dyn.end() ? nullptr : &it->second;
}

class Hyperlink {
public:
    Hyperlink(std::size_t from = 0, std::size_t to = 0, double w = 1.0)
        : _from(from), _to(to), _weight(w) {}

    std::size_t from()   const { return _from;   }
    std::size_t to()     const { return _to;     }
    double      weight() const { return _weight; }

private:
    std::size_t _from{};
    std::size_t _to{};
    double      _weight{1.0};
};

}
#endif