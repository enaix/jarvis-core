#pragma once
#include "graph.h"
#include <cassert>

inline bool test_graph_init()
{
    using namespace jsc;

    Node<std::string> topic("Topic A");
    Widget<std::string> w("Widget 1");
    
    topic.add_widget(w);

    AttrValue<std::string> v_i64(42);
    AttrValue<std::string> v_str(std::string("hello"));

    assert(v_i64.is_vec_i64() && v_i64.at_i64(0) == 42);
    assert(v_str.is_str() && v_str.str() == "hello");

    return true;
}