# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.19

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
CMAKE_SOURCE_DIR = /home/gautham/Documents/Projects/ROS-SLAM

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/gautham/Documents/Projects/ROS-SLAM/build

# Include any dependencies generated for this target.
include CMakeFiles/optimizationStuff.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/optimizationStuff.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/optimizationStuff.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/optimizationStuff.dir/flags.make

CMakeFiles/optimizationStuff.dir/visualSLAM/src/optimizationStuff.cpp.o: CMakeFiles/optimizationStuff.dir/flags.make
CMakeFiles/optimizationStuff.dir/visualSLAM/src/optimizationStuff.cpp.o: ../visualSLAM/src/optimizationStuff.cpp
CMakeFiles/optimizationStuff.dir/visualSLAM/src/optimizationStuff.cpp.o: CMakeFiles/optimizationStuff.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/gautham/Documents/Projects/ROS-SLAM/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/optimizationStuff.dir/visualSLAM/src/optimizationStuff.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/optimizationStuff.dir/visualSLAM/src/optimizationStuff.cpp.o -MF CMakeFiles/optimizationStuff.dir/visualSLAM/src/optimizationStuff.cpp.o.d -o CMakeFiles/optimizationStuff.dir/visualSLAM/src/optimizationStuff.cpp.o -c /home/gautham/Documents/Projects/ROS-SLAM/visualSLAM/src/optimizationStuff.cpp

CMakeFiles/optimizationStuff.dir/visualSLAM/src/optimizationStuff.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/optimizationStuff.dir/visualSLAM/src/optimizationStuff.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/gautham/Documents/Projects/ROS-SLAM/visualSLAM/src/optimizationStuff.cpp > CMakeFiles/optimizationStuff.dir/visualSLAM/src/optimizationStuff.cpp.i

CMakeFiles/optimizationStuff.dir/visualSLAM/src/optimizationStuff.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/optimizationStuff.dir/visualSLAM/src/optimizationStuff.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/gautham/Documents/Projects/ROS-SLAM/visualSLAM/src/optimizationStuff.cpp -o CMakeFiles/optimizationStuff.dir/visualSLAM/src/optimizationStuff.cpp.s

# Object files for target optimizationStuff
optimizationStuff_OBJECTS = \
"CMakeFiles/optimizationStuff.dir/visualSLAM/src/optimizationStuff.cpp.o"

# External object files for target optimizationStuff
optimizationStuff_EXTERNAL_OBJECTS =

devel/lib/liboptimizationStuff.so: CMakeFiles/optimizationStuff.dir/visualSLAM/src/optimizationStuff.cpp.o
devel/lib/liboptimizationStuff.so: CMakeFiles/optimizationStuff.dir/build.make
devel/lib/liboptimizationStuff.so: CMakeFiles/optimizationStuff.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/gautham/Documents/Projects/ROS-SLAM/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX shared library devel/lib/liboptimizationStuff.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/optimizationStuff.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/optimizationStuff.dir/build: devel/lib/liboptimizationStuff.so
.PHONY : CMakeFiles/optimizationStuff.dir/build

CMakeFiles/optimizationStuff.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/optimizationStuff.dir/cmake_clean.cmake
.PHONY : CMakeFiles/optimizationStuff.dir/clean

CMakeFiles/optimizationStuff.dir/depend:
	cd /home/gautham/Documents/Projects/ROS-SLAM/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/gautham/Documents/Projects/ROS-SLAM /home/gautham/Documents/Projects/ROS-SLAM /home/gautham/Documents/Projects/ROS-SLAM/build /home/gautham/Documents/Projects/ROS-SLAM/build /home/gautham/Documents/Projects/ROS-SLAM/build/CMakeFiles/optimizationStuff.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/optimizationStuff.dir/depend
