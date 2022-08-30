## begin /* Set the application name. */
#set(APPLICATION_NAME bfs)
## end /* Set the application name. */
#
## begin /* Add CUDA executables */
#add_executable(${APPLICATION_NAME})
#
#set(SOURCE_LIST 
#    ${APPLICATION_NAME}.cu
#)
#
#target_sources(${APPLICATION_NAME} PRIVATE ${SOURCE_LIST})
#target_link_libraries(${APPLICATION_NAME} PRIVATE essentials)
#get_target_property(ESSENTIALS_ARCHITECTURES essentials CUDA_ARCHITECTURES)
#set_target_properties(${APPLICATION_NAME} 
#    PROPERTIES 
#        CUDA_ARCHITECTURES ${ESSENTIALS_ARCHITECTURES}
#) # XXX: Find a better way to inherit essentials properties.
#
#message(STATUS "Example Added: ${APPLICATION_NAME}")
## end /* Add CUDA executables */
#

add_executable(bfs-HIP bfs.cu)
set_source_files_properties(bfs.cu PROPERTIES LANGUAGE HIP)
target_include_directories(bfs-HIP SYSTEM PRIVATE /opt/rocm/include)
target_link_libraries(bfs-HIP PRIVATE essentials)
target_include_directories(bfs-HIP SYSTEM BEFORE PRIVATE ${CMAKE_SOURCE_DIR}/externals/rocthrust-src)
add_compile_definitions(__HIP_PLATFORM_AMD__)
