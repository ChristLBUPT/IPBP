from torch.utils.cpp_extension import CppExtension, load

# ext = CppExtension(name='test_module', sources=['./test_module.h'], )
module = load('test_module', sources=['./test_module.cpp'])
print(module)
print(module.add)
print(module.add(1, 2))