#ifndef GRAPH_H
#define GRAPH_H

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <variant>
#include <vector>

namespace jsc {

template <class T>
using decay_t = std::decay_t<T>;

template <class TStr = std::string>
using AttrVariant = std::variant<
    std::array<std::int64_t, 4>,
    std::array<double, 4>,
    TStr,
    std::vector<std::int64_t>,
    std::vector<double>
>;

template <class TStr = std::string>
class AttrValue {
public:
    using variant_type = AttrVariant<TStr>;

    AttrValue() = default;
    AttrValue(std::initializer_list<std::int64_t> il) { _v = to_array(il); }
    AttrValue(std::initializer_list<double> il) { _v = to_array(il); }

    template <class TVal, class = std::enable_if_t<!std::is_same_v<decay_t<TVal>, AttrValue>>>
    explicit AttrValue(TVal&& v) { init(std::forward<TVal>(v)); }

    template <class T> T &get() { return std::get<T>(_v); }
    template <class T> const T &get() const { return std::get<T>(_v); }

    bool is_str() const { return std::holds_alternative<TStr>(_v); }
    bool is_vec_i64() const {
        return std::holds_alternative<std::array<std::int64_t,4>>(_v) ||
               std::holds_alternative<std::vector<std::int64_t>>(_v);
    }
    bool is_vec_f64() const {
        return std::holds_alternative<std::array<double,4>>(_v) ||
               std::holds_alternative<std::vector<double>>(_v);
    }

    TStr &str() { return std::get<TStr>(_v); }
    const TStr &str() const { return std::get<TStr>(_v); }

    int64_t &i64() { return at_i64(0); }
    uint64_t &ui64() { return reinterpret_cast<uint64_t&>(at_i64(0)); }
    double &f64() { return at_f64(0); }

    int64_t &at_i64 (std::size_t i);
    uint64_t &at_ui64(std::size_t i) { return reinterpret_cast<uint64_t&>(at_i64(i)); }
    double &at_f64 (std::size_t i);

    const int64_t &at_i64 (std::size_t i) const { return const_cast<AttrValue*>(this)->at_i64(i); }
    const uint64_t &at_ui64(std::size_t i) const { return reinterpret_cast<const uint64_t&>(at_i64(i)); }
    const double &at_f64 (std::size_t i) const { return const_cast<AttrValue*>(this)->at_f64(i); }

    void push_i64(int64_t v);
    void push_f64(double  v);
    bool pop_i64(int64_t &out);
    bool pop_f64(double  &out);

protected:
    variant_type _v{};

    static constexpr std::int64_t empty_i64() { return std::numeric_limits<std::int64_t>::min(); }
    static double empty_f64() { return std::numeric_limits<double>::quiet_NaN(); }
    static bool is_empty(int64_t x) { return x == empty_i64(); }
    static bool is_empty(double  x) { return std::isnan(x); }

    template <class TVal>
    void init(TVal&& v);

    template <class T>
    static std::array<T,4> to_array(std::initializer_list<T> il);
};

template <class TStr>
int64_t &AttrValue<TStr>::at_i64(std::size_t i) {
    if (std::holds_alternative<std::array<std::int64_t,4>>(_v))
        return std::get<std::array<std::int64_t,4>>(_v).at(i);
    if (std::holds_alternative<std::vector<std::int64_t>>(_v))
        return std::get<std::vector<std::int64_t>>(_v).at(i);
    throw std::bad_variant_access{};
}
template <class TStr>
double &AttrValue<TStr>::at_f64(std::size_t i) {
    if (std::holds_alternative<std::array<double,4>>(_v))
        return std::get<std::array<double,4>>(_v).at(i);
    if (std::holds_alternative<std::vector<double>>(_v))
        return std::get<std::vector<double>>(_v).at(i);
    throw std::bad_variant_access{};
}

template <class TStr>
void AttrValue<TStr>::push_i64(int64_t v) {
    if (std::holds_alternative<std::vector<std::int64_t>>(_v)) {
        std::get<std::vector<std::int64_t>>(_v).push_back(v);
        return;
    }
    if (std::holds_alternative<std::array<std::int64_t,4>>(_v)) {
        const auto &a = std::get<std::array<std::int64_t,4>>(_v);
        std::vector<std::int64_t> vec;
        for (auto x : a) if (!is_empty(x)) vec.push_back(x);
        vec.push_back(v);
        _v = std::move(vec);
        return;
    }
    _v = std::vector<std::int64_t>{v};
}
template <class TStr>
void AttrValue<TStr>::push_f64(double v) {
    if (std::holds_alternative<std::vector<double>>(_v)) {
        std::get<std::vector<double>>(_v).push_back(v);
        return;
    }
    if (std::holds_alternative<std::array<double,4>>(_v)) {
        const auto &a = std::get<std::array<double,4>>(_v);
        std::vector<double> vec;
        for (auto x : a) if (!is_empty(x)) vec.push_back(x);
        vec.push_back(v);
        _v = std::move(vec);
        return;
    }
    _v = std::vector<double>{v};
}
template <class TStr>
bool AttrValue<TStr>::pop_i64(int64_t &out) {
    if (std::holds_alternative<std::array<std::int64_t,4>>(_v)) {
        const auto &a = std::get<std::array<std::int64_t,4>>(_v);
        std::vector<std::int64_t> vec;
        for (auto x : a) if (!is_empty(x)) vec.push_back(x);
        _v = vec;
    }
    if (!std::holds_alternative<std::vector<std::int64_t>>(_v)) return false;
    auto &vec = std::get<std::vector<std::int64_t>>(_v);
    if (vec.empty()) return false;
    out = vec.back(); vec.pop_back(); return true;
}
template <class TStr>
bool AttrValue<TStr>::pop_f64(double &out) {
    if (std::holds_alternative<std::array<double,4>>(_v)) {
        const auto &a = std::get<std::array<double,4>>(_v);
        std::vector<double> vec;
        for (auto x : a) if (!is_empty(x)) vec.push_back(x);
        _v = vec;
    }
    if (!std::holds_alternative<std::vector<double>>(_v)) return false;
    auto &vec = std::get<std::vector<double>>(_v);
    if (vec.empty()) return false;
    out = vec.back(); vec.pop_back(); return true;
}

template <class TStr>
template <class TVal>
void AttrValue<TStr>::init(TVal&& val) {
    using raw = decay_t<TVal>;
    if constexpr (std::is_same_v<raw, variant_type>) {
        _v = std::forward<TVal>(val);
    } else if constexpr (std::is_integral_v<raw>) {
        std::array<std::int64_t,4> a{};
        a.fill(empty_i64());
        a[0] = static_cast<std::int64_t>(val);
        _v = a;
    } else if constexpr (std::is_floating_point_v<raw>) {
        std::array<double,4> a{};
        a.fill(empty_f64());
        a[0] = static_cast<double>(val);
        _v = a;
    } else {
        _v = std::forward<TVal>(val);
    }
}
template <class TStr>
template <class T>
std::array<T,4> AttrValue<TStr>::to_array(std::initializer_list<T> il) {
    std::array<T,4> a{};
    if constexpr (std::is_same_v<T,std::int64_t>) a.fill(static_cast<T>(empty_i64()));
    else a.fill(static_cast<T>(empty_f64()));
    std::size_t i = 0;
    for (T v : il) { if (i < 4) a[i++] = v; else break; }
    return a;
}

template <class TStr = std::string>
struct Attr { TStr name{}; AttrValue<TStr> value{}; };

template <class TStr = std::string>
class Widget {
public:
    explicit Widget(const TStr &n = {}) : _name(n) {}
    explicit Widget(AttrValue<TStr> n) : _name(std::move(n)) {}

    void set_attr(TStr k, AttrValue<TStr> v) { _dyn.emplace(std::move(k), std::move(v)); }
    AttrValue<TStr>* get_attr(const TStr &k);
    const AttrValue<TStr>* get_attr(const TStr &k) const;

    Widget &add_child(Widget w) { _children.emplace_back(std::move(w)); return _children.back(); }
    const std::vector<Widget>& children() const { return _children; }

protected:
    AttrValue<TStr> _name;
    std::unordered_map<TStr, AttrValue<TStr>> _dyn;
    std::vector<Widget> _children;
};
template <class TStr>
AttrValue<TStr>* Widget<TStr>::get_attr(const TStr &k) {
    auto it = _dyn.find(k); return it == _dyn.end() ? nullptr : &it->second;
}
template <class TStr>
const AttrValue<TStr>* Widget<TStr>::get_attr(const TStr &k) const {
    auto it = _dyn.find(k); return it == _dyn.end() ? nullptr : &it->second;
}

template <class TStr = std::string>
class Node {
public:
    explicit Node(const TStr &n = {}) : _name(n) {}
    explicit Node(AttrValue<TStr> n) : _name(std::move(n)) {}

    Widget<TStr>& add_widget(Widget<TStr> w) { _widgets.emplace_back(std::move(w)); return _widgets.back(); }
    const std::vector<Widget<TStr>>& widgets() const { return _widgets; }

    void set_attr(TStr k, AttrValue<TStr> v) { _dyn.emplace(std::move(k), std::move(v)); }
    AttrValue<TStr>* get_attr(const TStr &k);
    const AttrValue<TStr>* get_attr(const TStr &k) const;

protected:
    AttrValue<TStr> _name;
    std::unordered_map<TStr, AttrValue<TStr>> _dyn;
    std::vector<Widget<TStr>> _widgets;
};
template <class TStr>
AttrValue<TStr>* Node<TStr>::get_attr(const TStr &k) {
    auto it = _dyn.find(k); return it == _dyn.end() ? nullptr : &it->second;
}
template <class TStr>
const AttrValue<TStr>* Node<TStr>::get_attr(const TStr &k) const {
    auto it = _dyn.find(k); return it == _dyn.end() ? nullptr : &it->second;
}

class Hyperlink {
public:
    Hyperlink(std::size_t from = 0, std::size_t to = 0, double w = 1.0)
        : _from(from), _to(to), _weight(w) {}

    std::size_t from() const { return _from;   }
    std::size_t to() const { return _to;     }
    double weight() const { return _weight; }

protected:
    std::size_t _from{};
    std::size_t _to{};
    double _weight{1.0};
};
}
#endif