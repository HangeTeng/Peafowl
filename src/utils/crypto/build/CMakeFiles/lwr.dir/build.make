# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.24

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /root/Peafowl/src/utils/crypto

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /root/Peafowl/src/utils/crypto/build

# Include any dependencies generated for this target.
include CMakeFiles/lwr.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/lwr.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/lwr.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/lwr.dir/flags.make

CMakeFiles/lwr.dir/lwr.cu.o: CMakeFiles/lwr.dir/flags.make
CMakeFiles/lwr.dir/lwr.cu.o: /root/Peafowl/src/utils/crypto/lwr.cu
CMakeFiles/lwr.dir/lwr.cu.o: CMakeFiles/lwr.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/root/Peafowl/src/utils/crypto/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object CMakeFiles/lwr.dir/lwr.cu.o"
	/usr/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -c /root/Peafowl/src/utils/crypto/lwr.cu -o CMakeFiles/lwr.dir/lwr.cu.o
	/usr/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -M /root/Peafowl/src/utils/crypto/lwr.cu -MT CMakeFiles/lwr.dir/lwr.cu.o -o CMakeFiles/lwr.dir/lwr.cu.o.d

CMakeFiles/lwr.dir/lwr.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/lwr.dir/lwr.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/lwr.dir/lwr.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/lwr.dir/lwr.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target lwr
lwr_OBJECTS = \
"CMakeFiles/lwr.dir/lwr.cu.o"

# External object files for target lwr
lwr_EXTERNAL_OBJECTS =

lwr.cpython-38-x86_64-linux-gnu.so: CMakeFiles/lwr.dir/lwr.cu.o
lwr.cpython-38-x86_64-linux-gnu.so: CMakeFiles/lwr.dir/build.make
lwr.cpython-38-x86_64-linux-gnu.so: CMakeFiles/lwr.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/root/Peafowl/src/utils/crypto/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CUDA shared library lwr.cpython-38-x86_64-linux-gnu.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/lwr.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/lwr.dir/build: lwr.cpython-38-x86_64-linux-gnu.so
.PHONY : CMakeFiles/lwr.dir/build

CMakeFiles/lwr.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/lwr.dir/cmake_clean.cmake
.PHONY : CMakeFiles/lwr.dir/clean

CMakeFiles/lwr.dir/depend:
	cd /root/Peafowl/src/utils/crypto/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /root/Peafowl/src/utils/crypto /root/Peafowl/src/utils/crypto /root/Peafowl/src/utils/crypto/build /root/Peafowl/src/utils/crypto/build /root/Peafowl/src/utils/crypto/build/CMakeFiles/lwr.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/lwr.dir/depend
