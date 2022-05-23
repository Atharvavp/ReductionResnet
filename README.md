
## Build commands (CMake) (From inside inference)

Torch_DIR=$CONDA_PREFIX/lib/python3.1/site-packages/torch/share/cmake/Torch cmake -DCMAKE_BUILD_TYPE=Release .
cmake --build .

## Build/Install (From inside inference/src)
python setup.py install
