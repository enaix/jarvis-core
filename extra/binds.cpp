#include "graph.h"


#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/variant.h>
#include <nanobind/make_iterator.h>
#include <sstream>


namespace nb = nanobind;
using namespace nb::literals;
using namespace jsc;

// Module definition
NB_MODULE(jsc_common, m) {
    m.doc() = "Python bindings for common jarvis-core classes";

    // Bind AttrValue class
    nb::class_<AttrValue<>>(m, "AttrValue")
        // Constructors
        .def(nb::init<>(), "Default constructor")
        .def(nb::init<std::int64_t>(), "value"_a, "Construct from int64")
        .def(nb::init<double>(), "value"_a, "Construct from double")
        .def(nb::init<std::string>(), "value"_a, "Construct from string")
        .def(nb::init<std::vector<std::int64_t>>(), "values"_a, "Construct from vector of int64")
        .def(nb::init<std::vector<double>>(), "values"_a, "Construct from vector of double")

        // Static factory methods for initializer lists
        .def_static("from_i64", [](const std::vector<std::int64_t>& vals) {
            return AttrValue<>(vals);
        }, "values"_a, "Create from list of integers")

        .def_static("from_f64", [](const std::vector<double>& vals) {
            return AttrValue<>(vals);
        }, "values"_a, "Create from list of floats")

        // Size and type checking
        .def("size", &AttrValue<>::size, "Get the size of the stored value")
        .def("is_str", &AttrValue<>::is_str, "Check if value is a string")
        .def("is_vec_i64", &AttrValue<>::is_vec_i64, "Check if value is int64 vector/array")
        .def("is_vec_f64", &AttrValue<>::is_vec_f64, "Check if value is double vector/array")

        // String access
        .def("str", nb::overload_cast<>(&AttrValue<>::str),
             nb::rv_policy::reference_internal,
             "Get string value (mutable)")
        .def("str", nb::overload_cast<>(&AttrValue<>::str, nb::const_),
             nb::rv_policy::reference_internal,
             "Get string value (const)")

        // Scalar access
        .def("i64", &AttrValue<>::i64,
             nb::rv_policy::reference_internal,
             "Get int64 scalar value")
        .def("ui64", &AttrValue<>::ui64,
             nb::rv_policy::reference_internal,
             "Get uint64 scalar value")
        .def("f64", &AttrValue<>::f64,
             nb::rv_policy::reference_internal,
             "Get double scalar value")

        // Indexed access
        .def("at_i64", nb::overload_cast<std::size_t>(&AttrValue<>::at_i64),
             "index"_a,
             nb::rv_policy::reference_internal,
             "Get int64 value at index")
        .def("at_i64", nb::overload_cast<std::size_t>(&AttrValue<>::at_i64, nb::const_),
             "index"_a,
             nb::rv_policy::reference_internal,
             "Get int64 value at index (const)")

        .def("at_ui64", nb::overload_cast<std::size_t>(&AttrValue<>::at_ui64),
             "index"_a,
             nb::rv_policy::reference_internal,
             "Get uint64 value at index")
        .def("at_ui64", nb::overload_cast<std::size_t>(&AttrValue<>::at_ui64, nb::const_),
             "index"_a,
             nb::rv_policy::reference_internal,
             "Get uint64 value at index (const)")

        .def("at_f64", nb::overload_cast<std::size_t>(&AttrValue<>::at_f64),
             "index"_a,
             nb::rv_policy::reference_internal,
             "Get double value at index")
        .def("at_f64", nb::overload_cast<std::size_t>(&AttrValue<>::at_f64, nb::const_),
             "index"_a,
             nb::rv_policy::reference_internal,
             "Get double value at index (const)")

        // Push/pop operations
        .def("push_i64", &AttrValue<>::push_i64, "value"_a, "Push int64 value")
        .def("push_f64", &AttrValue<>::push_f64, "value"_a, "Push double value")
        .def("pop", &AttrValue<>::pop, "Pop last value from vector/array")

        // Python sequence protocol support
        .def("__len__", &AttrValue<>::size)
        .def("__getitem__", [](const AttrValue<>& self, std::size_t i) -> nb::object {
            if (i >= self.size()) {
                throw nb::index_error("Index out of range");
            }
            if (self.is_vec_i64()) {
                return nb::cast(self.at_i64(i));
            } else if (self.is_vec_f64()) {
                return nb::cast(self.at_f64(i));
            } else {
                throw nb::type_error("Cannot index string type");
            }
        }, "index"_a)

        .def("__setitem__", [](AttrValue<>& self, std::size_t i, nb::object value) {
            if (i >= self.size()) {
                throw nb::index_error("Index out of range");
            }
            if (self.is_vec_i64()) {
                self.at_i64(i) = nb::cast<std::int64_t>(value);
            } else if (self.is_vec_f64()) {
                self.at_f64(i) = nb::cast<double>(value);
            } else {
                throw nb::type_error("Cannot index string type");
            }
        }, "index"_a, "value"_a)

        // String representation
        .def("__repr__", [](const AttrValue<>& self) {
            std::ostringstream oss;
            oss << "AttrValue(";
            if (self.is_str()) {
                oss << "str='" << self.str() << "'";
            } else if (self.is_vec_i64()) {
                oss << "i64=[";
                for (std::size_t i = 0; i < self.size(); ++i) {
                    if (i > 0) oss << ", ";
                    oss << self.at_i64(i);
                }
                oss << "]";
            } else if (self.is_vec_f64()) {
                oss << "f64=[";
                for (std::size_t i = 0; i < self.size(); ++i) {
                    if (i > 0) oss << ", ";
                    oss << self.at_f64(i);
                }
                oss << "]";
            }
            oss << ")";
            return oss.str();
        });

    nb::class_<AttrSet<>>(m, "AttrSet")
        .def(nb::init<>(), "Default constructor")

        // Set methods
        .def("set", [](AttrSet<>& self, const std::string& k, const AttrValue<>& v) {
            self.set(k, std::move(v));
        }, "key"_a, "value"_a, "Set attribute with AttrValue")

        .def("set", [](AttrSet<>& self, const std::string& k, const std::string& v) {
            self.set(k, v);
        }, "key"_a, "value"_a, "Set attribute with string")

        .def("set", [](AttrSet<>& self, const std::string& k, std::int64_t v) {
            self.set(k, AttrValue<>(v));
        }, "key"_a, "value"_a, "Set attribute with int64")

        .def("set", [](AttrSet<>& self, const std::string& k, double v) {
            self.set(k, AttrValue<>(v));
        }, "key"_a, "value"_a, "Set attribute with double")

        .def("set", [](AttrSet<>& self, const std::string& k, const std::vector<std::int64_t>& v) {
            self.set(k, v);
        }, "key"_a, "value"_a, "Set attribute with int64 list")

        .def("set", [](AttrSet<>& self, const std::string& k, const std::vector<double>& v) {
            self.set(k, v);
        }, "key"_a, "value"_a, "Set attribute with double list")

        // Get and contains
        .def("contains", &AttrSet<>::contains, "key"_a, "Check if key exists")

        .def("get", nb::overload_cast<const std::string&>(&AttrSet<>::get),
             "key"_a,
             nb::rv_policy::reference_internal,
             "Get attribute value (returns None if not found)")

        .def("get", nb::overload_cast<const std::string&>(&AttrSet<>::get, nb::const_),
             "key"_a,
             nb::rv_policy::reference_internal,
             "Get attribute value (const, returns None if not found)")

        // Python dict-like interface
        .def("__getitem__", [](AttrSet<>& self, const std::string& key) -> AttrValue<>& {
            auto* val = self.get(key);
            if (!val) throw nb::key_error(key.c_str());
            return *val;
        }, "key"_a, nb::rv_policy::reference_internal)

        .def("__setitem__", [](AttrSet<>& self, const std::string& key, const nb::object& value) {
            // Try to determine type from Python object
            if (nb::isinstance<nb::str>(value)) {
                self.set(std::move(key), nb::cast<std::string>(value));
            } else if (nb::isinstance<nb::int_>(value)) {
                self.set(std::move(key), AttrValue<>(nb::cast<std::int64_t>(value)));
            } else if (nb::isinstance<nb::float_>(value)) {
                self.set(std::move(key), AttrValue<>(nb::cast<double>(value)));
            } else if (nb::isinstance<AttrValue<>>(value)) {
                self.set(std::move(key), nb::cast<AttrValue<>>(value));
            } else if (nb::isinstance<nb::list>(value) || nb::isinstance<nb::tuple>(value)) {
                // Try int list first
                try {
                    auto vec = nb::cast<std::vector<std::int64_t>>(value);
                    self.set(std::move(key), vec);
                    return;
                } catch (...) {}
                // Try float list
                try {
                    auto vec = nb::cast<std::vector<double>>(value);
                    self.set(std::move(key), vec);
                    return;
                } catch (...) {
                    throw nb::type_error("List must contain all ints or all floats");
                }
            } else {
                throw nb::type_error("Unsupported value type");
            }
        }, "key"_a, "value"_a)

        .def("__iter__", [](const AttrSet<>& self){
                return nb::make_key_iterator(nb::type<AttrSet<>>(), "key_iterator", self.begin(), self.end());
        }, nb::keep_alive<0, 1>())
        .def("items", [](const AttrSet<>& self){
                return nb::make_iterator(nb::type<AttrSet<>>(), "item_iterator", self.begin(), self.end());
        }, nb::keep_alive<0, 1>())
        .def("values", [](const AttrSet<>& self){
                return nb::make_value_iterator(nb::type<AttrSet<>>(), "value_iterator", self.begin(), self.end());
        }, nb::keep_alive<0, 1>())

        .def("__contains__", &AttrSet<>::contains, "key"_a);

    // Bind Widget class
    nb::class_<Widget<>, AttrSet<>>(m, "Widget")
        // Constructors
        .def(nb::init<>(), "Default constructor")
        .def(nb::init<const std::string&>(), "name"_a, "Construct with name")

        // Name access
        .def_prop_rw("name",
            nb::overload_cast<>(&Widget<>::name, nb::const_),
            nb::overload_cast<>(&Widget<>::name),
            "Widget name")

        // Children management
        .def("add_child", &Widget<>::add_child, "widget"_a,
             nb::rv_policy::reference_internal,
             "Add child widget and return reference to it")

        .def("children", nb::overload_cast<>(&Widget<>::children),
             nb::rv_policy::reference_internal,
             "Get children list (mutable)")

        .def("children", nb::overload_cast<>(&Widget<>::children, nb::const_),
             nb::rv_policy::reference_internal,
             "Get children list (const)")

        .def("child", nb::overload_cast<std::size_t>(&Widget<>::child),
             "index"_a,
             nb::rv_policy::reference_internal,
             "Get child by index (mutable)")

        .def("child", nb::overload_cast<std::size_t>(&Widget<>::child, nb::const_),
             "index"_a,
             nb::rv_policy::reference_internal,
             "Get child by index (const)")

        .def("__len__", [](const Widget<>& self) {
            return self.children().size();
        })

        .def("__repr__", [](const Widget<>& self) {
            return "Widget(name='" + self.name() + "', children=" +
                   std::to_string(self.children().size()) + ")";
        });

    // Bind Node class
    nb::class_<Node<>, AttrSet<>>(m, "Node")
        // Constructors
        .def(nb::init<>(), "Default constructor")
        .def(nb::init<const std::string&>(), "name"_a, "Construct with name")

        // Name access
        .def_prop_rw("name",
            nb::overload_cast<>(&Node<>::name, nb::const_),
            nb::overload_cast<>(&Node<>::name),
            "Node name")

        // Widget access
        .def("set_widget", &Node<>::set_widget, "widget"_a,
             nb::rv_policy::reference_internal,
             "Set widget and return reference to it")

        .def("widget", nb::overload_cast<>(&Node<>::widget),
             nb::rv_policy::reference_internal,
             "Get widget (mutable)")

        .def("widget", nb::overload_cast<>(&Node<>::widget, nb::const_),
             nb::rv_policy::reference_internal,
             "Get widget (const)")

        .def("__repr__", [](const Node<>& self) {
            return "Node(name='" + self.name() + "')";
        });


    // Bind Hyperlink class
    nb::class_<Hyperlink<>, AttrSet<>>(m, "Hyperlink")
        // Constructors
        .def(nb::init<>(), "Default constructor")
        .def(nb::init<std::size_t, std::size_t>(), "from"_a, "to"_a, "Construct with internal node ids")

        .def("__repr__", [](const Widget<>& self) {
            return "Hyperlink()";
        });
}
