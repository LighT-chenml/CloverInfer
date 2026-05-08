from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension


setup(
    name="clover_host_reduce",
    ext_modules=[
        CppExtension(
            "clover_host_reduce",
            [
                "python_reducer.cpp",
                "attention_reducer.cc",
            ],
            include_dirs=["."],
            extra_compile_args=["-O2"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
