# Topics graph implementation

## Description

The purpose of this module is to provide fast and convenient objects and methods for working with the topics graph. The main priority is the convenience (we don't want to needlessly sacrifice usability for speed), while computational efficiency comes next. 

## Requirements

- `jsc::Node`, `jsc::Hyperlink` and `jsc::Widget` provide base functionality for creating, accessing and modifying the topics graph

- A node contains a vector of widgets and the list of attributes.

- Each widget also contains a vector of widgets and the list of the attributes.

- Hyperlink is a bidirectional link from a widget to a node. It needs to have fast and robust indexing, since we cannot use plain pointers due to vector reallocations.

- An attribute is a value with an optional name and a value. The value is interpreted at a runtime with the use of std::variant. Each attribute can contain either 4 8-byte primitive types, a string or a vector of primitive types.

- Each widget or node contains fixed attributes (example: name) and dynamic ones (interpreted at runtime). A hashmap is used for indexing dynamically added attributes (wip).

## Implementation details

- The underlying implementation should be hidden inside of the class. For example, the user shouldn't even know if he's working with a vector or an array inside of the attribute. There may be different implementations of these classes in the future.

- There are several options regarding the attribute storage: either use one large hashmap for multiple objects, use individual hashmaps or something else (linear indexing).

## Next steps

Create and run some tests, including runtime evaluation (todo).

This code should be callable from Python with the help of either [`nanobind`](https://nanobind.readthedocs.io) or `pybind11` (wip).

## Glossary

- Node represents a vertex of the topics graph. The node contains a tree of widgets.

- Hyperlink is essentially an edge of the graph, but it comes from a widget to a node. Hyperlinks should be bidirectional.


## Useful links

- `cppreference` is essential
- Different ways of implementing DOM: https://simdjson.org/api/0.6.0/md_doc_ondemand.html