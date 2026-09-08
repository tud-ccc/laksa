find_path(GUROBI_INCLUDE_DIRS
    NAMES gurobi_c.h
    HINTS ${GUROBI_DIR} $ENV{GUROBI_HOME}
    PATH_SUFFIXES include)

find_library(GUROBI_LIBRARY
    NAMES gurobi gurobi120 gurobi130
    HINTS ${GUROBI_DIR} $ENV{GUROBI_HOME}
    PATH_SUFFIXES lib)

find_library(GUROBI_CXX_LIBRARY
    NAMES gurobi_c++
    HINTS ${GUROBI_DIR} $ENV{GUROBI_HOME}
    PATH_SUFFIXES lib)
set(GUROBI_CXX_DEBUG_LIBRARY ${GUROBI_CXX_LIBRARY})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(GUROBI DEFAULT_MSG GUROBI_LIBRARY)

if(GUROBI_FOUND)
    if(NOT TARGET Gurobi::core)
        add_library(Gurobi::core SHARED IMPORTED GLOBAL)
        set_target_properties(Gurobi::core PROPERTIES
            IMPORTED_LOCATION "${GUROBI_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${GUROBI_INCLUDE_DIRS}"
        )
    endif()
    if(NOT TARGET Gurobi::cxx)
        add_library(Gurobi::cxx STATIC IMPORTED GLOBAL)
        set_target_properties(Gurobi::cxx PROPERTIES
            IMPORTED_LOCATION "${GUROBI_CXX_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${GUROBI_INCLUDE_DIRS}"
            INTERFACE_LINK_LIBRARIES Gurobi::core
        )
    endif()
endif()
