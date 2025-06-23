# Topics graph implementation

## Description

The purpose of this module is to provide fast and convenient objects and methods for working with the topics graph. The main priority is the convenience (we don't want to needlessly sacrifice usability for speed), while computational efficiency comes next. 

## Requirements

- `jsc::Node`, `jsc::Hyperlink` and `jsc::Widget` provide base functionality for creating, accessing and modifying the topics graph

WIP!!

## Glossary

- Node represents a vertex of the topics graph. The node contains a tree of widgets.

- Hyperlink is essentially an edge of the graph, but it comes from a widget to a node. Hyperlinks should be bidirectional.
