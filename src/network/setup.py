from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='clover_net',
    ext_modules=[
        CppExtension(
            'clover_net', 
            ['csrc/clover_transport.cpp'],
            libraries=['ibverbs'], # Link against libibverbs
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
