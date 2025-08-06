//
// Created by Flynn on 23.06.2025.
//

#include "graph.h"
#include <cassert>

int main()
{
    using namespace jsc;

    Node<std::string> topic({"Topic A"});
    Widget<std::string> w({"Widget 1"});
    topic.add_widget(w);

    AttrValue<std::string> val(42);
    assert(val.is_int_arr() && val.at<int64_t>(0) == 42);

    AttrValue<std::string> txt(std::string("hello"));
    assert(txt.is_string() && txt.str() == "hello");

    return 0;
}
