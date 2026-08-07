add_subdirectory(Interfaces)
add_subdirectory(IR)
add_subdirectory(Transforms)

add_library(DFGEnums OBJECT
    DFGEnums.cpp
)
add_dependencies(DFGEnums DFGEnumsIncGen)
