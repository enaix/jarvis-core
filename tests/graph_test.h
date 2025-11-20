#pragma once
#include "graph.h"
#include <cassert>
#include <iostream>

inline bool test_graph_init()
{
    using namespace jsc;
    std::cout << "test_graph_init()" << std::endl;

    Node<std::string> topic("Topic A");
    Widget<std::string> w("Widget 1");
    
    topic.set_widget(w);

    AttrValue<std::string> v_i64(42);
    AttrValue<std::string> v_str(std::string("hello"));

    assert(v_i64.is_vec_i64() && v_i64.at_i64(0) == 42);
    assert(v_str.is_str() && v_str.str() == "hello");

    return true;
}

inline bool test_graph_access()
{
    using namespace jsc;
    std::cout << "test_graph_access()" << std::endl;
    Node<std::string> a("A");
    Widget<std::string> b("root"), p("paragraph");

    b.add_child(p).set("geometry", {3.14});

    assert(b.child(0).get("geometry")->size() == 1);
    assert(b.child(0).get("geometry")->at_f64(0) == 3.14);

    b.child(0).get("geometry")->push_f64(42);

    assert(b.child(0).get("geometry")->size() == 2);
    assert(b.child(0).get("geometry")->at_f64(1) == 42);

    a.set_widget(b);
    assert(a.widget().name() == "root");
    return true;
}
