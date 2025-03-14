import copy
import functools
import types
import torch
import torch.nn.functional as F
from torch import Tensor
from types import MethodType

def copy_func_with_new_globals(f, globals=None):
    """Based on https://stackoverflow.com/a/13503277/2988730 (@unutbu)"""
    if globals is None:
        globals = f.__globals__
    g = types.FunctionType(f.__code__, globals, name=f.__name__,
                           argdefs=f.__defaults__, closure=f.__closure__)
    g = functools.update_wrapper(g, f)
    g.__module__ = f.__module__
    g.__kwdefaults__ = copy.copy(f.__kwdefaults__)
    return g

def add_wrapper_after_function_call_in_method(module, method_name, function_name, wrapper_fn):
    '''
    This function adds a wrapper after the output of a function call in the method named `method_name`. 
    Only calls directly in the method are affected. Calls by other functions called in the method are not affected.
    '''

    original_method = getattr(module, method_name).__func__
    method_globals = dict(original_method.__globals__)
    wrapper = wrapper_fn(method_globals[function_name])
    method_globals[function_name] = wrapper
    new_method = copy_func_with_new_globals(original_method, globals=method_globals)
    setattr(module, method_name, new_method.__get__(module))
    return wrapper

def add_lora_in_linear(module):
    def get_fwd_fn():
        def lora_forward(self, input: Tensor) -> Tensor:
            lora_A = self.adapters["lora_A"]
            lora_B = self.adapters["lora_B"]
            return lora_A(lora_B(input))
        return lora_forward
    if hasattr(module, "adapters"):
        fwd_fn = get_fwd_fn()
        setattr(module, "lora_forward", MethodType(fwd_fn, module))
