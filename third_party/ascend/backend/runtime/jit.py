from __future__ import annotations, division

import inspect
import os
from typing import Callable, Iterable, Optional, TypeVar, Union, overload

from triton.runtime.driver import driver
from triton.runtime.jit import mangle_type, JITFunction

T = TypeVar("T")


def create_dynamic_param_function_from_signature(sig, kparams, backend):
    """
    Equivalent to sig.bind followed by apply_defaults. This generates a
    native Python function (using exec) which can be memoized on a per-kernel
    basis to avoid having to run these expensive functions -- which constitute
    much of the kernel launch overhead -- every time we run the kernel.
    """

    assert len(sig.parameters) == len(kparams)

    # Create the function argument list and the dict entries for the return statement
    func_args = []
    dict_entries = []
    constexpr_vals = []
    non_constexpr_vals = []

    for ((name, sp), kp) in zip(sig.parameters.items(), kparams):
        if sp.default is inspect.Parameter.empty:
            func_args.append(name)
            dict_entries.append(f"'{name}': {name}")
        else:
            func_args.append(f"{name}=default_{name}")
            dict_entries.append(f"'{name}': {name}")
        if kp.is_constexpr:
            constexpr_vals.append(name)
        else:
            non_constexpr_vals.append(name)
    constexpr_vals = ', '.join(constexpr_vals)
    non_constexpr_vals = ', '.join(non_constexpr_vals)

    func_args.append('**excess_kwargs')

    # Join all arguments into a function definition string
    args_str = ', '.join(func_args)
    dict_str = ', '.join(dict_entries)
    func_body = "def dynamic_func(%s):\n    return {%s}, (%s), (%s), excess_kwargs" % (
        args_str, dict_str, constexpr_vals, non_constexpr_vals)
    # Prepare defaults to be inserted into function namespace
    func_namespace = {
        f"default_{name}": param.default
        for name, param in sig.parameters.items()
        if param.default is not inspect.Parameter.empty
    }

    func_namespace['mangle_type'] = mangle_type
    func_namespace['compute_spec_key'] = backend.compute_spec_key

    # Execute the function string in func_namespace to create the function
    exec(func_body, func_namespace)

    # Extract the newly created function from the namespace
    return func_namespace['dynamic_func']


def create_static_param_function_from_signature(sig, kparams, backend):
    """
    Equivalent to sig.bind followed by apply_defaults. This generates a
    native Python function (using exec) which can be memoized on a per-kernel
    basis to avoid having to run these expensive functions -- which constitute
    much of the kernel launch overhead -- every time we run the kernel.
    """

    assert len(sig.parameters) == len(kparams)

    # Create the function argument list and the dict entries for the return statement
    func_args = []
    signature_types = []
    specialisations = []

    for ((name, sp), kp) in zip(sig.parameters.items(), kparams):
        if sp.default is inspect.Parameter.empty:
            func_args.append(name)
        else:
            func_args.append(f"{name}=default_{name}")
        if not kp.is_constexpr:
            if not kp.do_not_specialize:
                if not kp.do_not_specialize_on_alignment:
                    specialisations.append('compute_spec_key(%s, align=True)' % name)
                else:
                    specialisations.append('compute_spec_key(%s, align=False)' % name)
            if kp.annotation_type:
                signature_types.append('"%s"' % kp.annotation_type)
            else:
                signature_types.append('mangle_type(%s, %s)' % (name, 'True' if kp.is_const else 'False'))

    cache_key = ', '.join(signature_types + specialisations)

    func_args.append('**excess_kwargs')

    # Join all arguments into a function definition string
    args_str = ', '.join(func_args)
    func_body = "def static_func(%s):\n    return (%s)" % (
        args_str, cache_key)
    # Prepare defaults to be inserted into function namespace
    func_namespace = {
        f"default_{name}": param.default
        for name, param in sig.parameters.items()
        if param.default is not inspect.Parameter.empty
    }

    func_namespace['mangle_type'] = mangle_type
    func_namespace['compute_spec_key'] = backend.compute_spec_key

    # Execute the function string in func_namespace to create the function
    exec(func_body, func_namespace)

    # Extract the newly created function from the namespace
    return func_namespace['static_func']


class JITFunction_Ascend(JITFunction[T]):

    def create_binder(self, backend):
        """
        Precompute as much as possible.
        """
        from triton.compiler import CompiledKernel, compile, ASTSource
        self.CompiledKernel = CompiledKernel
        self.compile = compile
        self.ASTSource = ASTSource
        self.dynamic_binder = create_dynamic_param_function_from_signature(self.signature, self.params, backend)
        self.static_binder = create_static_param_function_from_signature(self.signature, self.params, backend)
        self.constexpr_indices = [i for (i, p) in enumerate(self.params) if p.is_constexpr]
        self.non_constexpr_indices = [i for (i, p) in enumerate(self.params) if not p.is_constexpr]
        self.specialised_indices = [
            i for (i, p) in enumerate(self.params) if (not p.do_not_specialize) and (not p.is_constexpr)
        ]

    def run(self, *args, grid, warmup, **kwargs):
        kwargs["debug"] = kwargs.get("debug", False) or self.triton_debug

        device = driver.active.get_current_device()
        stream = driver.active.get_current_stream(device)
        if self.target is None:
            self.target = driver.active.get_current_target()
        if self.backend is None:
            from triton.compiler import make_backend
            self.backend = make_backend(self.target)

        # Execute pre run hooks with args and kwargs
        for hook in self.pre_run_hooks:
            hook(*args, **kwargs)

        if self.dynamic_binder is None or self.static_binder is None:
            self.create_binder(self.backend)

        bound_args, constexpr_vals, non_constexpr_vals, excess_kwargs = self.dynamic_binder(*args, **kwargs)
        if self.sig_and_spec is None:
            self.sig_and_spec = self.static_binder(*args, **kwargs)

        # compute cache key
        key = ''.join(self.sig_and_spec) + str((constexpr_vals, excess_kwargs))
        kernel = self.cache[device].get(key, None)

        if kernel is None:
            # Kernel is not cached; we have to compile.
            options = self.backend.parse_options(kwargs)

            # deprecated arguments
            assert "device_type" not in kwargs, "device_type option is deprecated; current target will be used"
            assert "device" not in kwargs, "device option is deprecated; current device will be used"
            assert "stream" not in kwargs, "stream option is deprecated; current stream will be used"
            for k in excess_kwargs:
                if k not in options.__dict__:
                    raise KeyError("Keyword argument %s was specified but unrecognised" % k)

            bound_vals = tuple(bound_args.values())

            # `None` is nullptr. Implicitly convert to *i8. This needs to be
            # done here rather than when we build the signature as otherwise
            # the kernel cache key could not distinguish between byte pointers
            # and None arguments, resulting in a downstream mismatch:
            sigkeys = [self.params[i].name for i in self.non_constexpr_indices]
            sigvals = self.sig_and_spec[:len(sigkeys)]
            signature = {k: ('*i8' if (v == 'none') else v) for (k, v) in zip(sigkeys, sigvals)}

            configs = (self.backend.get_attrs_descriptor(self.params, bound_vals),)
            constant_params = configs[0].get_constants()
            constants = {
                p.name: v
                for (v, p) in zip(bound_vals, self.params)
                if p.is_constexpr or (p.num in constant_params) or v is None
            }
            for i, arg in constants.items():
                if callable(arg):
                    raise TypeError(f"Callable constexpr at index {i} is not supported")

            kernel = self._do_compile(key, signature, device, self.backend, self.target, constants, options, configs[0],
                                      warmup)
            if kernel is None:
                return None

        # Check that used global values have not changed.
        not_present = object()
        for (name, _), (val, globals_dict) in self.used_global_vals.items():
            if (newVal := globals_dict.get(name, not_present)) != val:
                raise RuntimeError(
                    f"Global variable {name} has changed since we compiled this kernel, from {val} to {newVal}")

        if not warmup:
            # canonicalize grid
            assert grid is not None
            if callable(grid):
                # Arguments are passed as a dict to `grid`, by contract.
                # second parameter to `grid`.
                grid = grid(bound_args)
            grid_size = len(grid)
            grid_0 = grid[0]
            grid_1 = grid[1] if grid_size > 1 else 1
            grid_2 = grid[2] if grid_size > 2 else 1
            if hasattr(kernel, "result"):
                kernel = kernel.result()

            # launch kernel
            launch_metadata = kernel.launch_metadata(grid, stream, *non_constexpr_vals)
            kernel.run(grid_0, grid_1, grid_2, stream, kernel.function, kernel.packed_metadata, launch_metadata,
                       self.CompiledKernel.launch_enter_hook, self.CompiledKernel.launch_exit_hook, *non_constexpr_vals)
        return kernel

    def __init__(self, fn, version=None, do_not_specialize=None, do_not_specialize_on_alignment=None, debug=None,
                 noinline=None, repr=None, launch_metadata=None):
        super().__init__(fn, version, do_not_specialize, do_not_specialize_on_alignment, debug,
                         noinline, repr, launch_metadata)
        self.debug = debug
        self.dynamic_binder = None
        self.static_binder = None
        self.target = None
        self.backend = None
        self.sig_and_spec = None
        self.triton_debug = os.environ.get("TRITON_DEBUG", "0") == "1"

    def __setattr__(self, name, value):
        super(JITFunction_Ascend, self).__setattr__(name, value)
        # - when `.src` attribute is set, cache path needs
        #   to be reinitialized
        if name == "src":
            self.hash = None

    def __repr__(self):
        return f"JITFunction_Ascend({self.module}:{self.fn.__name__})"


# -----------------------------------------------------------------------------
# `jit` decorator
# -----------------------------------------------------------------------------


@overload
def ascend_jit(fn: T) -> JITFunction_Ascend[T]:
    ...


@overload
def ascend_jit(
        *,
        version=None,
        repr: Optional[Callable] = None,
        launch_metadata: Optional[Callable] = None,
        do_not_specialize: Optional[Iterable[int]] = None,
        do_not_specialize_on_alignment: Optional[Iterable[int]] = None,
        debug: Optional[bool] = None,
        noinline: Optional[bool] = None,
) -> Callable[[T], JITFunction_Ascend[T]]:
    ...


def ascend_jit(
        fn: Optional[T] = None,
        *,
        version=None,
        repr: Optional[Callable] = None,
        launch_metadata: Optional[Callable] = None,
        do_not_specialize: Optional[Iterable[int]] = None,
        do_not_specialize_on_alignment: Optional[Iterable[int]] = None,
        debug: Optional[bool] = None,
        noinline: Optional[bool] = None,
) -> Union[JITFunction_Ascend[T], Callable[[T], JITFunction[T]]]:
    """
    Decorator for JIT-compiling a function using the Triton compiler.

    :note: When a jit'd function is called, arguments are
        implicitly converted to pointers if they have a :code:`.data_ptr()` method
        and a `.dtype` attribute.

    :note: This function will be compiled and run on the GPU. It will only have access to:

           * python primitives,
           * builtins within the triton package,
           * arguments to this function,
           * other jit'd functions

    :param fn: the function to be jit-compiled
    :type fn: Callable
    """

    def decorator(fn: T) -> JITFunction[T]:
        assert callable(fn)
        if os.getenv("TRITON_INTERPRET", "0") == "1":
            from triton.runtime.interpreter import InterpretedFunction
            return InterpretedFunction(fn, version=version, do_not_specialize=do_not_specialize,
                                       do_not_specialize_on_alignment=do_not_specialize_on_alignment, debug=debug,
                                       noinline=noinline, repr=repr, launch_metadata=launch_metadata)
        else:
            return JITFunction_Ascend(
                fn,
                version=version,
                do_not_specialize=do_not_specialize,
                do_not_specialize_on_alignment=do_not_specialize_on_alignment,
                debug=debug,
                noinline=noinline,
                repr=repr,
                launch_metadata=launch_metadata,
            )

    if fn is not None:
        return decorator(fn)

    else:
        return decorator
