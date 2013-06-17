#CMAKE_PREFIX_PATH=/opt/trilinos/dev/openmpi/1.4.5/gcc/4.7.2/release/shared:$CMAKE_PREFIX_PATH \
#    -DCMAKE_INSTALL_PREFIX:PATH=/opt/nosh/dev/master/openmpi/1.4.5/gcc/4.7.2/release/shared \
CXX=mpicxx \
FC=mpif90 \
cmake \
    ../source/
