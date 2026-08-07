add_subdirectory(Interfaces)
add_subdirectory(IR)
add_subdirectory(Transforms)

add_library(EmitHLSEnums OBJECT
    EmitHLSEnums.cpp
)
add_dependencies(EmitHLSEnums EmitHLSEnumsIncGen)
