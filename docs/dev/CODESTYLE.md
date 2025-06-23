# Project codestyle and guidelines

An informal codestyle goes here

## Important stuff

### C/C++

- This project uses C++20 standard, feel free to use metaprogramming features, lambdas, concepts or other modern features. The primary compiler is the latest stable Clang build.

- The standard library is mostly fine unless stated otherwise. Don't include whole namespaces though.

- Please refrain from using virtual classes, since it imposes an overhead. Most of the stuff can be implemented without vtables (look up method hiding). If you absolutely must, consult me first.

- Avoid including new libraries if you can; the less dependencies we have the better.

## Styling

I prefer not to break someone's way of writing code, but if the code is hard to read, I may ask you to change the styling.

### C/C++

- I prefer to describe classes in headers directly, since this approach always works with template classes.

```cpp
class SomeClass : public OtherClass
{
protected:
    std::uint8_t _some_field;
    std::vector<A> _other_field;

public:
    SomeClass() : _some_field(0) {}

    template<class TArg>
    void foo(const TArg& arg)
    {
        if (arg.abc())
        {
            std::cout << "foo" << std::endl;
        } else {
            // ...
        }

        int ok = bar([&](const char* out){
            std::cout << out << std::endl;
        });
    }

    int bar(auto func) const
    {
        func("bar");
        return 1;
    }
};
```

