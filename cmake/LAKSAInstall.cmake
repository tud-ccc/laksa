include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

# add_mlir_library_install() hardcodes lib${LLVM_LIBDIR_SUFFIX}, so follow it
# rather than CMAKE_INSTALL_LIBDIR, which is lib64 on some distributions.
set(LAKSA_INSTALL_LIBDIR "lib${LLVM_LIBDIR_SUFFIX}")
set(LAKSA_INSTALL_PACKAGEDIR "${LAKSA_INSTALL_LIBDIR}/cmake/laksa")

function(laksa_collect_libraries dir out_var)
    set(result "")
    get_property(subdirs DIRECTORY ${dir} PROPERTY SUBDIRECTORIES)
    foreach(subdir IN LISTS subdirs)
        laksa_collect_libraries(${subdir} subdir_libs)
        list(APPEND result ${subdir_libs})
    endforeach()

    get_property(targets DIRECTORY ${dir} PROPERTY BUILDSYSTEM_TARGETS)
    foreach(target IN LISTS targets)
        if(target MATCHES "^obj\\.")
            continue()
        endif()
        get_target_property(type ${target} TYPE)
        if(type STREQUAL "STATIC_LIBRARY"
                OR type STREQUAL "SHARED_LIBRARY"
                OR type STREQUAL "OBJECT_LIBRARY")
            list(APPEND result ${target})
        endif()
    endforeach()

    set(${out_var} ${result} PARENT_SCOPE)
endfunction()

# Call once, after every add_subdirectory().
macro(laksa_install)
    laksa_collect_libraries(${PROJECT_SOURCE_DIR}/lib _laksa_libs)
    laksa_collect_libraries(${PROJECT_SOURCE_DIR}/lib/CAPI _laksa_capi_libs)
    list(APPEND _laksa_libs ${_laksa_capi_libs})
    list(REMOVE_DUPLICATES _laksa_libs)

    install(TARGETS ${_laksa_libs}
        EXPORT LAKSATargets
        ARCHIVE DESTINATION ${LAKSA_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${LAKSA_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        OBJECTS DESTINATION ${LAKSA_INSTALL_LIBDIR}
        COMPONENT LAKSALibraries
    )

    install(TARGETS
            ladle
            laksa-lsp-server
            laksa-opt
            laksa-translate
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        COMPONENT LAKSATools
    )

    install(DIRECTORY
            ${PROJECT_SOURCE_DIR}/include/laksa-mlir
            ${PROJECT_SOURCE_DIR}/include/laksa-mlir-c
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        COMPONENT LAKSAHeaders
        FILES_MATCHING
            PATTERN "*.h"
            PATTERN "*.td"
    )

    install(DIRECTORY ${PROJECT_BINARY_DIR}/include/laksa-mlir
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        COMPONENT LAKSAHeaders
        FILES_MATCHING
            PATTERN "*.inc"
    )

    install(EXPORT LAKSATargets
        FILE LAKSATargets.cmake
        NAMESPACE LAKSA::
        DESTINATION ${LAKSA_INSTALL_PACKAGEDIR}
        COMPONENT LAKSADevelopment
    )

    configure_package_config_file(
        ${PROJECT_SOURCE_DIR}/cmake/LAKSAConfig.cmake.in
        ${PROJECT_BINARY_DIR}/cmake/LAKSAConfig.cmake
        INSTALL_DESTINATION ${LAKSA_INSTALL_PACKAGEDIR}
        PATH_VARS CMAKE_INSTALL_INCLUDEDIR
    )

    write_basic_package_version_file(
        ${PROJECT_BINARY_DIR}/cmake/LAKSAConfigVersion.cmake
        COMPATIBILITY SameMajorVersion
    )

    install(FILES
            ${PROJECT_BINARY_DIR}/cmake/LAKSAConfig.cmake
            ${PROJECT_BINARY_DIR}/cmake/LAKSAConfigVersion.cmake
            ${PROJECT_SOURCE_DIR}/cmake/FindGUROBI.cmake
        DESTINATION ${LAKSA_INSTALL_PACKAGEDIR}
        COMPONENT LAKSADevelopment
    )
endmacro()
