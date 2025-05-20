#include <cstdio>
#include <torch/extension.h>

int add(int a, int b) 
{
    return a + b;
}

void hello();


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("add", &add, "add of two numbers");
  m.def("hello", &hello, "test output");
}