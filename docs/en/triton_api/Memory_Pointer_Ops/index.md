# Memory/Pointer Operations

|api|Brief Description|
|--|--|
|[load](./tl.load.md) |Returns a tensor whose values are loaded from the memory location defined by the pointer|
|[store](./tl.store.md) |Stores a data tensor to the memory location defined by the pointer|
|[make_block_ptr](./tl.make_block_ptr.md) |Returns a pointer to a block within the parent tensor|
|[advance](./tl.advance.md) |Advances a block pointer|
|[load_tensor_descriptor](./load_tensor_descriptor.md) | Loads a data block from a tensor descriptor |
|[make_tensor_descriptor](./make_tensor_descriptor.md) | Creates a tensor descriptor object |

```{toctree}
:maxdepth: 3
:hidden:
tl.load.md
tl.store.md
tl.make_block_ptr.md
tl.advance.md
load_tensor_descriptor.md
make_tensor_descriptor.md
