include_directories(SYSTEM ${DataflowSchedulerDialects_SOURCE_DIR}/include)
include_directories(SYSTEM ${DataflowSchedulerDialects_BINARY_DIR}/include)
include_directories(SYSTEM ${KTIR_INCLUDE_DIRS})

################################################################################
# Generate packages and shared library
################################################################################

set(_source_components
  MLIRPythonSources.Core
  MLIRPythonSources.Dialects.arith
  MLIRPythonSources.Dialects.builtin
  MLIRPythonSources.Dialects.func
  MLIRPythonSources.Dialects.index
  MLIRPythonSources.Dialects.memref
  MLIRPythonSources.Dialects.scf
  
  KTIRPythonSources.Dialects
  KTIRPythonExtensions

  DataflowSchedulerDialectsPythonSources.Dialects
  DataflowSchedulerDialectsPythonExtensions

  DataflowSchedulerPythonSources
  DataflowSchedulerPythonExtensions
)

if(MLIR_PYTHON_STUBGEN_ENABLED)
  list(APPEND _source_components 
    MLIRPythonExtension.Core.type_stub_gen

    KTIRPythonSources.type_stubs

    DataflowSchedulerDialectsPythonSources.type_stubs
  )
endif()

# NOTE: The MLIRPythonSupport module, which is compiled here implicitly, may 
#       contain errors that we can't do anything about.
add_compile_options(-Wno-error)

add_mlir_python_common_capi_library(DataflowSchedulerPythonCAPI
  INSTALL_COMPONENT DataflowSchedulerPythonModules
  INSTALL_DESTINATION "${MLIR_BINDINGS_PYTHON_INSTALL_PREFIX}/_mlir_libs"
  OUTPUT_DIRECTORY "${PYTHON_PACKAGE_DIR}/_mlir_libs"
  DECLARED_SOURCES ${_source_components}
)

add_mlir_python_modules(DataflowSchedulerPythonModules
  ROOT_PREFIX "${PYTHON_PACKAGE_DIR}"
  INSTALL_PREFIX "${MLIR_BINDINGS_PYTHON_INSTALL_PREFIX}"
  DECLARED_SOURCES ${_source_components}
  COMMON_CAPI_LINK_LIBS
    DataflowSchedulerPythonCAPI
)
