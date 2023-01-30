__all__ = ["useFun3d", "useCaps"]

import importlib

fun3d_loader = importlib.util.find_spec("fun3d")
has_fun3d = fun3d_loader is not None


def usesFun3d(test_method):
    """
    This loader declarator is applied to any fun3d unittest method as @usesFun3d
    above the method. That test is only ran if fun3d can be imported.
    """
    if has_fun3d:
        return test_method
    else:

        def fake_pass_method():
            return

        return fake_pass_method


caps_loader = importlib.util.find_spec("pyCAPS")
has_caps = caps_loader is not None


def usesCaps(test_method):
    """
    This loader declarator is applied to any ESP/CAPS unittest method as @usesCaps
    above the method. That test is only ran if pyCAPS can be imported.
    """
    if has_caps:
        return test_method
    else:

        def fake_pass_method():
            return

        return fake_pass_method
