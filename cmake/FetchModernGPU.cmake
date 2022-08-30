include(FetchContent)
set(FETCHCONTENT_QUIET ON)

message(STATUS "Cloning External Project: ModernGPU")
get_filename_component(FC_BASE "../externals"
                REALPATH BASE_DIR "${CMAKE_BINARY_DIR}")
set(FETCHCONTENT_BASE_DIR ${FC_BASE})

FetchContent_Declare(
    moderngpu
    GIT_REPOSITORY https://github.com/crozhon/moderngpu.git
    GIT_TAG        hip
)

FetchContent_GetProperties(moderngpu)
if(NOT moderngpu_POPULATED)
  FetchContent_Populate(
    moderngpu
  )
endif()
set(MODERNGPU_INCLUDE_DIR "${moderngpu_SOURCE_DIR}/src")
