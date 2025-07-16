//
// Created by Flynn on 23.06.2025.
//

#ifndef GRAPH_H
#define GRAPH_H

#include <memory>
#include <vector>
#include <variant>
#include <array>


namespace jsc
{

class Hyperlink; // forward-decl

// Attribute can take the following types :
// Since sizeof(std::string) is 32, we can efficiently store 8*4 bytes
template<class TStr>
using AttrVariant = std::variant<std::array<std::int64_t, 4>, std::array<double, 4>, TStr, std::vector<std::int64_t>, std::vector<double>>;
// Note: do we always need 64 bit int/float values?



/**
 * @brief Variable attribute type. Supports tuples of integral type (4 elements), vector of integral types or a TStr
 * @tparam TStr Variable string class
 */
template<class TStr>
class AttrValue
{
protected:
    AttrVariant<TStr> _value;

public:
    AttrValue() {}

    // Init from type which is known at compile-time
    template<class TVal>
    explicit AttrValue(const TVal& val) { init(val); }

    // Add move, copy ctor
    // ...

    /**
     * @brief Fetch a single integral value
     * @tparam TVal Value type
     */
    template<class TVal>
    void get() -> TVal&
    {
        // Implement logic for fetching a singular integral value
        // ...
    }

    /**
     * @brief Fetch an i-th integral value
     * @param index Element index
     */
    template<class TVal>
    void at(std::size_t index) -> TVal&
    {
        // Determine if _value is an array<4> or a vector
    }

    /**
     * @brief Return the stored string
     */
    TStr& str() { return _value; } // Do we need to

    // Create const versions of these methods

    // Add explicit functions to fetch (i64, ui64, f64, i64(i), ...)

protected:
    /**
     * @brief Initialize empty object with new values
     * @param val Value to set
     */
    template<class TVal>
    void init(const TVal& val)
    {
        if constexpr (std::is_integral_v<std::decay_t<TVal>>)
        {
            // Use std::array for a single value
            _value = std::array<std::decay_t<TVal>, 4>(val);
        } else {
            // Place value on heap

            // ...
        }
    }
};


template<class TStr>
class Attr
{
protected:
    TStr _name;
    AttrValue<TStr> _value;

    // Implement ctors and other methods
};


template<class TStr>
class Widget
{
protected:
    AttrValue<TStr> _name;
    // std::vector<Widget<TStr>> _widgets; // children
    // ...

    // We need to store a hashmap of attributes

public:
    Widget() {}
};


template<class TStr>
class Node // aka Topic
{
protected:
    AttrValue<TStr> _name;
    // Widget<TStr> _root;
    // ...

    // Should also contain a hashmap of attributes

public:
    Node() {}
};


class Hyperlink
{
protected:
    // ...

public:
    Hyperlink() {}
};


}

#endif //GRAPH_H
