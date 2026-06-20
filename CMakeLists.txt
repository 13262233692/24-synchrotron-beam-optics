cmake_minimum_required(VERSION 3.18)
project(synchrotron_beam_optics VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(HDF5 REQUIRED COMPONENTS CXX)
find_package(OpenMP)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development NumPy)

include(FetchContent)
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG        v2.12.0
)
FetchContent_MakeAvailable(pybind11)

set(CORE_SOURCES
    src/core/physics_constants.cpp
    src/core/bragg_solver.cpp
    src/core/crystal_geometry.cpp
    src/core/photon_transport.cpp
    src/core/hdf5_writer.cpp
    src/mpi/mpi_scheduler.cpp
)

add_library(synchrotron_core STATIC ${CORE_SOURCES})
target_include_directories(synchrotron_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/core
    ${CMAKE_CURRENT_SOURCE_DIR}/src/mpi
)
target_link_libraries(synchrotron_core PUBLIC
    MPI::MPI_CXX
    HDF5::HDF5_CXX
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(synchrotron_core PUBLIC OpenMP::OpenMP_CXX)
endif()

if(MSVC)
    target_compile_options(synchrotron_core PRIVATE
        /fp:fast
        /O2
    )
    target_compile_definitions(synchrotron_core PUBLIC
        _CRT_SECURE_NO_WARNINGS
        __SSE__
    )
else()
    target_compile_options(synchrotron_core PRIVATE
        -ffp-contract=fast
        -fno-math-errno
        -fno-trapping-math
        -O2
    )
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(synchrotron_core PRIVATE
            -march=native
            -msse2
        )
    endif()
endif()

pybind11_add_module(_synchrotron src/bindings.cpp)
target_link_libraries(_synchrotron PRIVATE synchrotron_core pybind11::module)

if(MSVC)
    target_compile_options(_synchrotron PRIVATE /fp:fast)
else()
    target_compile_options(_synchrotron PRIVATE
        -ffp-contract=fast
        -fno-math-errno
        -fno-trapping-math
    )
endif()

set_target_properties(_synchrotron PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python/synchrotron
    PREFIX ""
    SUFFIX "${Python3_SOABI}.pyd"
)

if(WIN32)
    set_target_properties(_synchrotron PROPERTIES SUFFIX ".pyd")
endif()
