#ifndef JSC_GRAPH_H
#define JSC_GRAPH_H

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <variant>
#include <vector>
#include <cassert>
#include <stdexcept>
#include <iostream>

#include "common.h"

namespace jsc {

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

    AttrValue() : _array_size(0) {}
    explicit AttrValue(std::initializer_list<std::int64_t> il) { to_array(il); }
    explicit AttrValue(std::initializer_list<double> il) { to_array(il); }
    explicit AttrValue(const std::vector<std::int64_t>& vec) : _v(vec) {}
    explicit AttrValue(const std::vector<double>& vec) : _v(vec) {}

    template <class TVal, class = std::enable_if_t<!std::is_same_v<std::decay_t<TVal>, AttrValue>>>
    explicit AttrValue(TVal&& v) { init(std::forward<TVal>(v)); }

    //AttrValue(AttrValue<TStr>&& v) = default;

    template <class T> T &get() { return std::get<T>(_v); }
    template <class T> const T &get() const { return std::get<T>(_v); }

    std::size_t size() const {
        if (std::holds_alternative<std::array<std::int64_t, 4>>(_v) || std::holds_alternative<std::array<double, 4>>(_v))
            return _array_size;
        else {
            if (std::holds_alternative<std::vector<double>>(_v))
                return std::get<std::vector<double>>(_v).size();
            else if (std::holds_alternative<std::vector<std::int64_t>>(_v))
                return std::get<std::vector<std::int64_t>>(_v).size();
            else
                return std::get<TStr>(_v).size();
        }
    }

    bool is_str() const {
        return std::holds_alternative<TStr>(_v);
    }
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

    int64_t &at_i64 (std::size_t i) {
        if (std::holds_alternative<std::array<std::int64_t,4>>(_v))
            return std::get<std::array<std::int64_t,4>>(_v).at(i);
        if (std::holds_alternative<std::vector<std::int64_t>>(_v))
            return std::get<std::vector<std::int64_t>>(_v).at(i);
        throw std::bad_variant_access{}; // The least problematic way to handle bad access
    }

    uint64_t &at_ui64(std::size_t i) { return reinterpret_cast<uint64_t&>(at_i64(i)); }

    double &at_f64 (std::size_t i) {
        if (std::holds_alternative<std::array<double,4>>(_v))
            return std::get<std::array<double,4>>(_v).at(i);
        if (std::holds_alternative<std::vector<double>>(_v))
            return std::get<std::vector<double>>(_v).at(i);
        throw std::bad_variant_access{}; // The least problematic way to handle bad access
    }
    
    const int64_t &at_i64 (std::size_t i) const { return const_cast<AttrValue*>(this)->at_i64(i); }
    const uint64_t &at_ui64(std::size_t i) const { return reinterpret_cast<const uint64_t&>(at_i64(i)); }
    const double &at_f64 (std::size_t i) const { return const_cast<AttrValue*>(this)->at_f64(i); }

    void push_i64(int64_t v) { do_push<std::int64_t>(v); }
    void push_f64(double v) { do_push<double>(v); }

    bool pop() {
        if (std::holds_alternative<std::array<std::int64_t,4>>(_v) ||
            std::holds_alternative<std::vector<std::int64_t>>(_v)) {
            do_pop<std::int64_t>();
            return true;
        }
        if (std::holds_alternative<std::array<double,4>>(_v) ||
            std::holds_alternative<std::vector<double>>(_v)) {
            do_pop<double>();
            return true;
        }

#ifdef ALWAYS_THROW_ON_ERROR
        throw std::bad_variant_access{};
#else
        return false; // Bad variant access
#endif
    }

protected:
    variant_type _v{};
    int _array_size;

    template <class TVal>
    void init(TVal&& v) {
        using raw = std::decay_t<TVal>;
        if constexpr (std::is_integral_v<raw>) {
            std::array<std::int64_t,4> a{};
            a[0] = static_cast<std::int64_t>(v);
            _array_size = 1;
            _v = a;
        } else if constexpr (std::is_floating_point_v<raw>) {
            std::array<double,4> a{};
            _array_size = 1;
            a[0] = static_cast<double>(v);
            _v = a;
        } else {
            // string or vec
            _v = std::forward<TVal>(v);
        }
    }

    template <class T>
    void to_array(std::initializer_list<T> il) {
        if (il.size() <= 4) {
            _array_size = il.size();
            std::array<T,4> a{};
            for (auto it = il.begin(); it != il.end(); it++)
                a[std::distance(il.begin(), it)] = *it;
            _v = a;
        } else {
            _v = std::vector<std::decay_t<T>>(il);
        }
    }

    template <class T>
    bool do_push(T v) {
        if (std::holds_alternative<std::vector<T>>(_v)) {
            std::get<std::vector<T>>(_v).push_back(v);
            return true;
        }
        if (std::holds_alternative<std::array<T,4>>(_v)) {
            const auto &a = std::get<std::array<T,4>>(_v);
            std::vector<T> vec;
            //vec.reserve(a.size() + 1);
            for (std::size_t i = 0; i < _array_size; i++)
                vec.push_back(a[i]);
            vec.push_back(v);
            _v = std::move(vec);
            return true;
        }
#ifdef ALWAYS_THROW_ON_ERROR
        throw std::bad_variant_access{};
#else
        return false; // cannot push to a string
#endif
    }

    template <class T>
    bool do_pop() {
        if (std::holds_alternative<std::array<T,4>>(_v)) {
            auto &a = std::get<std::array<T,4>>(_v);
            if (_array_size > 0)
                _array_size--;
            return true;
        }
        if (std::holds_alternative<std::vector<T>>(_v)) {
            auto &vec = std::get<std::vector<T>>(_v);
            if (!vec.empty()) vec.pop_back();
            return true;
        }
#ifdef ALWAYS_THROW_ON_ERROR
        throw std::bad_variant_access{};
#else
        return false; // cannot pop a string
#endif
        //assert((std::holds_alternative<std::array<T,4>>(_v) ||
        //        std::holds_alternative<std::vector<T>>(_v)) &&
        //        "do_pop<T>: wrong variant alternative");
    }
};


#ifdef OPTIMAL_STRUCTS
    static_assert(sizeof(AttrValue<std::string>) <= 64, "AttrValue size is not optimal for the cache line size of 64 bytes. You may set FORCE_OPTIMAL_STRUCTS to OFF to disable this warning");
#endif



template<class TStr = std::string>
class AttrSet {
public:
    void set(const TStr& k, const AttrValue<TStr>& v) { _dyn.emplace(std::move(k), std::move(v)); }
    void set(const TStr& k, const TStr& v) { _dyn.emplace(std::move(k), AttrValue<TStr>(v)); }
    void set(const TStr& k, std::initializer_list<std::int64_t> v) { _dyn.emplace(std::move(k), AttrValue<TStr>(v)); }
    void set(const TStr& k, std::initializer_list<double> v) { _dyn.emplace(std::move(k), AttrValue<TStr>(v)); }
    void set(const TStr& k, const std::vector<std::int64_t>& v) { _dyn.emplace(std::move(k), AttrValue<TStr>(v)); }
    void set(const TStr& k, const std::vector<double>& v) { _dyn.emplace(std::move(k), AttrValue<TStr>(v)); }

    bool contains(const TStr& k) const { return _dyn.find(k) != _dyn.end(); }
    AttrValue<TStr>* get(const TStr &k) { auto it = _dyn.find(k); return it != _dyn.end() ? &it->second : nullptr; }
    const AttrValue<TStr>* get(const TStr &k) const { auto it = _dyn.find(k); return it != _dyn.end() ? &it->second : nullptr; }

    const std::unordered_map<TStr, AttrValue<TStr>> attrs_map() const { return _dyn; }
    std::unordered_map<TStr, AttrValue<TStr>> attrs_map() { return _dyn; }
    template<class F>
    void each(F func) const { for (const auto& v : _dyn) { func(v.first, v.second); } }

    auto begin() const { return _dyn.cbegin(); }
    auto end() const { return _dyn.cend(); }

protected:
    std::unordered_map<TStr, AttrValue<TStr>> _dyn;
};


template <class TStr = std::string>
class Widget : public AttrSet<TStr> {
public:
    explicit Widget(const TStr &n = {}) : _name(n) {}

    Widget &add_child(const Widget& w) { _children.emplace_back(std::move(w)); return _children.back(); }
    const std::vector<Widget>& children() const { return _children; }
    std::vector<Widget>& children() { return _children; }
    const Widget& child(std::size_t i) const { return _children[i]; }
    Widget& child(std::size_t i) { return _children[i]; }

    const TStr& name() const { return _name; }
    TStr& name() { return _name; }
protected:
    TStr _name;
    std::vector<Widget> _children;
};


template <class TStr = std::string>
class Node : public AttrSet<TStr> {
public:
    explicit Node(const TStr &n = {}) : _id(std::numeric_limits<std::size_t>::max()), _name(n) {}

    /*Widget<TStr>& set_widget(Widget<TStr> w) { _widget = std::move(w); return _widget; }
    const Widget<TStr>& widget() const { return _widget; }
    Widget<TStr>& widget() { return _widget; }*/

    const TStr& name() const { return _name; }
    TStr& name() { return _name; }

    std::size_t _internal_id() const { return _id; }
    void _set_internal_id(std::size_t new_id) { _id = new_id; } // TODO do we need to store ids?
protected:
    std::size_t _id;
    TStr _name;
    Widget<TStr> _widget;
};


template <class TStr = std::string>
class Hyperlink : public AttrSet<TStr> {
public:
    Hyperlink() : _from(std::numeric_limits<std::size_t>::max()), _to(std::numeric_limits<std::size_t>::max()) {}
    Hyperlink(const Node<TStr>& from, const Node<TStr>& to) : _from(from._internal_id()), _to(to._internal_id()) {}

    std::size_t _id_from() const { return _from; }
    std::size_t _id_to() const { return _to; } // TODO do we need to store ids?

    // TODO add linked widget and operator== for explicit comparisons. We may need to implement unique ids for describing multigraphs

protected:
    std::size_t _from;
    std::size_t _to;
};


class NodeRef {
public:
    NodeRef() : _id(std::numeric_limits<std::size_t>::max()) {}
    NodeRef(std::size_t id) : _id(id) {}
    template<class TStr>
    NodeRef(const Node<TStr>& node) : _id(node._internal_id()) {}

    std::size_t _internal_id() const { return _id; }
protected:
    std::size_t _id;
};


class EdgeRef {
public:
    EdgeRef() : _from(std::numeric_limits<std::size_t>::max()), _to(std::numeric_limits<std::size_t>::max()) {}
    EdgeRef(std::size_t from, std::size_t to) : _from(from), _to(to) {}
    template<class TStr>
    EdgeRef(const Hyperlink<TStr>& edge) : _from(edge._id_from()), _to(edge._id_to()) {}

    std::size_t _id_from() const { return _from; }
    std::size_t _id_to() const { return _to; }

protected:
    std::size_t _from;
    std::size_t _to;
};


/**
 * @brief Basic AdjGraph class implementation, supports multigraphs
 *
 * Graph data is stored in a mapping id -> {node, edges, backlinks, widget}, where backlinks is a list of non-unique backlinks.
 */
template <class TStr = std::string>
class AdjGraph {
public:
    AdjGraph() : next_id(0) {}

    // GETTERS
    // =======

    /**
     * @brief Get the root widget of a node. May be invalidated after insertion
     */
    Widget<TStr>& widget(const NodeRef& node)
    {
        if (node._internal_id() == std::numeric_limits<std::size_t>::max()) [[unlikely]]
            throw std::invalid_argument{}; // The node does not exist
        return std::get<3>(data[node._internal_id()]);
    }

    /**
     * @brief Get the starting (edge.from) node of the edge. May be invalidated after insertion. Behavior is undefined if edge.from does not exist
     */
    Node<TStr>& from(const EdgeRef& edge)
    {
        std::size_t id = edge._id_from();
        if (id == std::numeric_limits<std::size_t>::max()) [[unlikely]] {
            throw std::invalid_argument{}; // The node is uninitialized
        }
        return std::get<0>(data[id]);
    }

    /**
     * @brief Get the ending (edge.to) node of the edge. May be invalidated after insertion. Behavior is undefined if edge.to does not exist
     */
    Node<TStr>& to(const EdgeRef& edge)
    {
        std::size_t id = edge._id_to();
        if (id == std::numeric_limits<std::size_t>::max()) [[unlikely]] {
            throw std::invalid_argument{}; // The node is uninitialized
        }
        return std::get<0>(data[id]);
    }

    // MODIFIERS
    // =========

    /**
     * @brief Add a node to the graph. If ALWAYS_THROW_ON_ERROR, throws if the node already has an id
     */
    const NodeRef& add_node(Node<TStr>& node, Widget<TStr>& widget)  // node is now initialized
    {
        if (node._internal_id() != std::numeric_limits<std::size_t>::max()) [[unlikely]] {
            throw std::invalid_argument{}; // The node cannot be added twice
        }
        node._set_internal_id(next_id);
        data.insert({next_id, {std::move(node), std::vector<Hyperlink<TStr>>(), std::vector<std::size_t>(), std::move(widget)}});
        next_id++;
        return NodeRef(node);
    }

    /**
     * @brief Delete a node from the graph. Behavior is undefined if node does not exist
     */
    bool del_node(Node<TStr>& node)
    {
        if (node._internal_id() == std::numeric_limits<std::size_t>::max()) [[unlikely]] {
#ifdef ALWAYS_THROW_ON_ERROR
            throw std::invalid_argument{}; // The node is uninitialized
#else
            return false;
#endif
        }
        std::size_t id = node._internal_id();

        // delete all edges [...] -> [node] using backlinks
        for (std::size_t from : std::get<2>(data[id])) {
            auto& edges = std::get<1>(data[from]);

            for (auto it = edges.begin(); it != edges.end();) {
                if (it._id_to() == id)
                    it = edges.erase(it);
                else it++;
            }
        }

        auto& edges = std::get<1>(data[id]);
        // delete all backlinks of [node] -> [...] (targets)
        for (auto it = edges.begin(); it != edges.end(); it++) {
            // iterate over all N backlinks
            auto& backlinks = std::get<2>(data[it._id_to()]);
            for (auto bl = backlinks.begin(); bl != backlinks.end();) {
                if (*bl == id)
                    bl = backlinks.erase(bl); // don't need to increment
                else
                    bl++;
            }
        }

        data.erase(id);
        node._set_internal_id(std::numeric_limits<std::size_t>::max());
        return true;
    }

    /**
     * @brief Add a new edge to the graph and returns the reference to the created edge. May be invalidated after insertion. Behavior is undefined if edge.from or edge.to do not exist
     */
    EdgeRef add_edge(Hyperlink<TStr>& edge)
    {
        std::size_t from = edge._id_from(), to = edge._id_to();
        if (from == std::numeric_limits<std::size_t>::max() || to == std::numeric_limits<std::size_t>::max()) [[unlikely]] {
            throw std::invalid_argument{}; // The node is uninitialized
        }

        auto& elems = data[from]; // no rehashing will occur
        std::get<1>(elems).push_back(std::move(edge));
        std::get<2>(data[to]).push_back(from); // add backlink, we may have duplicate backlinks
        return EdgeRef(edge);
    }

    /**
     * @brief Delete a single edge in the multigraph. Behavior is undefined if edge.from does not exist
     */
    bool del_edge(const Hyperlink<TStr>& edge)
    {
        std::size_t from = edge._id_from(), to = edge._id_to();
        if (from == std::numeric_limits<std::size_t>::max() || to == std::numeric_limits<std::size_t>::max()) [[unlikely]] {
#ifdef ALWAYS_THROW_ON_ERROR
            throw std::invalid_argument{}; // The node is uninitialized
#else
            return false;
#endif
        }
        auto& edges = std::get<1>(data[from]);

        // Iterate over all links between from->to and perform deletion
        bool found = false;
        for (auto it = edges.begin(); it != edges.end(); it++) {
            if (*it == edge) { // TODO implement explicit operator== for edges, right now it's ambiguous for multigraphs
                edges.erase(it);
                found = true;
                break;
            }
        }
        if (!found) [[unlikely]] {
#ifdef ALWAYS_THROW_ON_ERROR
            throw std::out_of_range{}; // The edge does not exist
#else
            return false;
#endif
        }

        // In a multigraph we would have n backlinks, so we would need to delete exactly one
        auto& backlinks = std::get<2>(data[to]);
        for (auto it = backlinks.begin(); it != backlinks.end(); it++) {
            if (*it == from) {
                backlinks.erase(it);
                return true;
            }
        }
        // Internal error
        [[unlikely]] {
        internal_err_begin();
        std::cerr << "AdjGraph::del_edge() : internal error : missing backlink for an existing edge" << std::endl;
        internal_err("adjgraph_missing_backlinks", __FILE__, __LINE__);
        }
    }
protected:
    // We can theoretically use a vector, but insertions would be not O(1)
    std::unordered_map<std::size_t, std::tuple<Node<TStr>, std::vector<Hyperlink<TStr>>, std::vector<std::size_t>, Widget<TStr>>> data; // TODO use a small vector for backlinks
    std::size_t next_id;
};

}
#endif
