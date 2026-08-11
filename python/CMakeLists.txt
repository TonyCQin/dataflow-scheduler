include(AddDataflowSchedulerPython)

# NOTE: These are for tablegen only!
include_directories(${PROJECT_SOURCE_DIR}/include ${PROJECT_BINARY_DIR}/include)

# Scheduler bindings contain their own copy of the MLIR bindings.
add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=${MLIR_PYTHON_PACKAGE_PREFIX}.")

# Where all the Python sources are.
set(PYTHON_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir_scheduler")
# Where all the binding library sources are.
set(PYTHON_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../lib/Bindings/Python")

################################################################################
# Sources
################################################################################

declare_mlir_python_sources(DataflowSchedulerPythonSources
  ROOT_DIR "${PYTHON_ROOT_DIR}"
)
if (MLIR_PYTHON_STUBGEN_ENABLED)
  declare_mlir_python_sources(DataflowSchedulerPythonSources.type_stubs
    ADD_TO_PARENT DataflowSchedulerPythonSources
  )
endif ()
declare_mlir_python_sources(DataflowSchedulerPythonSources.Dialects
  ADD_TO_PARENT DataflowSchedulerPythonSources
)
declare_mlir_python_sources(DataflowSchedulerPythonExtensions
  ROOT_DIR "${PYTHON_ROOT_DIR}"
  SOURCES
    _mlir_libs/_dataflow_scheduler/py.typed
)

################################################################################
# Extensions
################################################################################

declare_mlir_python_extension(DataflowSchedulerPythonExtensions.Main
  MODULE_NAME _dataflow_scheduler
  ADD_TO_PARENT DataflowSchedulerPythonExtensions
  SOURCES
    DataflowSchedulerModule.cpp
  EMBED_CAPI_LINK_LIBS
    MLIRCAPIIR
    MLIRCAPIArith
    MLIRCAPIFunc
    MLIRCAPILinalg
    MLIRCAPIMath
    MLIRCAPISCF
    MLIRCAPITensor
    MLIRCAPITransforms
    KTIR::KTDPCAPI
    DataflowScheduler::CAPIRegisterEverything
  PRIVATE_LINK_LIBS
    LLVMSupport
)

if(MLIR_PYTHON_STUBGEN_ENABLED)
  add_dataflow_scheduler_python_type_stubs(DataflowSchedulerPythonExtensions.Main
    ADD_TO_PARENT DataflowSchedulerPythonSources.type_stubs
    DEPENDS_TARGETS
      DataflowSchedulerPythonModules.extension._mlir.dso
      DataflowSchedulerPythonModules.extension._dataflow_scheduler.dso
      DataflowSchedulerPythonModules.sources.MLIRPythonSources.Core
      DataflowSchedulerPythonModules.sources.MLIRPythonSources.Core.Python
      DataflowSchedulerPythonModules.sources.MLIRPythonExtension.Core.type_stub_gen
    OUTPUTS
      _dataflow_scheduler.pyi
  )
endif()

set(EXCLUDE_FROM_ALL)
if (NOT PROJECT_IS_TOP_LEVEL)
  set(EXCLUDE_FROM_ALL EXCLUDE_FROM_ALL)
endif ()
add_subdirectory(mlir_scheduler ${EXCLUDE_FROM_ALL})
