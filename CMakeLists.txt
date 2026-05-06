cmake_minimum_required(VERSION 3.16)
project(particle_track_reconstruction LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(run_interaction
    scripts/run_interaction.cpp
    src/interaction/interaction.cpp
)

target_include_directories(run_interaction PRIVATE src)

add_executable(run_annealing
    scripts/run_annealing.cpp
    src/annealing/annealing.cpp
    src/annealing/spinglass.cpp
    src/interaction/interaction.cpp
)

target_include_directories(run_annealing PRIVATE src)
