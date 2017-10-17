# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.1

# Default target executed when no arguments are given to make.
default_target: all
.PHONY : default_target

# Allow only one "make -f Makefile2" at a time, but pass parallelism.
.NOTPARALLEL:
.PHONY : .NOTPARALLEL

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
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
RM = /usr/local/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/devel/2000/branches-DEV/temp/nn

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/devel/2000/branches-DEV/temp/nn

#=============================================================================
# Targets provided globally by CMake.

# Special rule for the target install/local
install/local: preinstall
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Installing only the local directory..."
	/usr/local/bin/cmake -DCMAKE_INSTALL_LOCAL_ONLY=1 -P cmake_install.cmake
.PHONY : install/local

# Special rule for the target install/local
install/local/fast: install/local
.PHONY : install/local/fast

# Special rule for the target edit_cache
edit_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Running CMake cache editor..."
	/usr/local/bin/ccmake -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
.PHONY : edit_cache

# Special rule for the target edit_cache
edit_cache/fast: edit_cache
.PHONY : edit_cache/fast

# Special rule for the target install
install: preinstall
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Install the project..."
	/usr/local/bin/cmake -P cmake_install.cmake
.PHONY : install

# Special rule for the target install
install/fast: preinstall/fast
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Install the project..."
	/usr/local/bin/cmake -P cmake_install.cmake
.PHONY : install/fast

# Special rule for the target rebuild_cache
rebuild_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Running CMake to regenerate build system..."
	/usr/local/bin/cmake -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
.PHONY : rebuild_cache

# Special rule for the target rebuild_cache
rebuild_cache/fast: rebuild_cache
.PHONY : rebuild_cache/fast

# Special rule for the target list_install_components
list_install_components:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Available install components are: \"Unspecified\""
.PHONY : list_install_components

# Special rule for the target list_install_components
list_install_components/fast: list_install_components
.PHONY : list_install_components/fast

# The main all target
all: cmake_check_build_system
	$(CMAKE_COMMAND) -E cmake_progress_start /home/devel/2000/branches-DEV/temp/nn/CMakeFiles /home/devel/2000/branches-DEV/temp/nn/CMakeFiles/progress.marks
	$(MAKE) -f CMakeFiles/Makefile2 all
	$(CMAKE_COMMAND) -E cmake_progress_start /home/devel/2000/branches-DEV/temp/nn/CMakeFiles 0
.PHONY : all

# The main clean target
clean:
	$(MAKE) -f CMakeFiles/Makefile2 clean
.PHONY : clean

# The main clean target
clean/fast: clean
.PHONY : clean/fast

# Prepare targets for installation.
preinstall: all
	$(MAKE) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall

# Prepare targets for installation.
preinstall/fast:
	$(MAKE) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall/fast

# clear depends
depend:
	$(CMAKE_COMMAND) -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 1
.PHONY : depend

#=============================================================================
# Target rules for targets named nn

# Build rule for target.
nn: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 nn
.PHONY : nn

# fast build rule for target.
nn/fast:
	$(MAKE) -f CMakeFiles/nn.dir/build.make CMakeFiles/nn.dir/build
.PHONY : nn/fast

src/functions.o: src/functions.cpp.o
.PHONY : src/functions.o

# target to build an object file
src/functions.cpp.o:
	$(MAKE) -f CMakeFiles/nn.dir/build.make CMakeFiles/nn.dir/src/functions.cpp.o
.PHONY : src/functions.cpp.o

src/functions.i: src/functions.cpp.i
.PHONY : src/functions.i

# target to preprocess a source file
src/functions.cpp.i:
	$(MAKE) -f CMakeFiles/nn.dir/build.make CMakeFiles/nn.dir/src/functions.cpp.i
.PHONY : src/functions.cpp.i

src/functions.s: src/functions.cpp.s
.PHONY : src/functions.s

# target to generate assembly for a file
src/functions.cpp.s:
	$(MAKE) -f CMakeFiles/nn.dir/build.make CMakeFiles/nn.dir/src/functions.cpp.s
.PHONY : src/functions.cpp.s

src/layer.o: src/layer.cpp.o
.PHONY : src/layer.o

# target to build an object file
src/layer.cpp.o:
	$(MAKE) -f CMakeFiles/nn.dir/build.make CMakeFiles/nn.dir/src/layer.cpp.o
.PHONY : src/layer.cpp.o

src/layer.i: src/layer.cpp.i
.PHONY : src/layer.i

# target to preprocess a source file
src/layer.cpp.i:
	$(MAKE) -f CMakeFiles/nn.dir/build.make CMakeFiles/nn.dir/src/layer.cpp.i
.PHONY : src/layer.cpp.i

src/layer.s: src/layer.cpp.s
.PHONY : src/layer.s

# target to generate assembly for a file
src/layer.cpp.s:
	$(MAKE) -f CMakeFiles/nn.dir/build.make CMakeFiles/nn.dir/src/layer.cpp.s
.PHONY : src/layer.cpp.s

src/main.o: src/main.cpp.o
.PHONY : src/main.o

# target to build an object file
src/main.cpp.o:
	$(MAKE) -f CMakeFiles/nn.dir/build.make CMakeFiles/nn.dir/src/main.cpp.o
.PHONY : src/main.cpp.o

src/main.i: src/main.cpp.i
.PHONY : src/main.i

# target to preprocess a source file
src/main.cpp.i:
	$(MAKE) -f CMakeFiles/nn.dir/build.make CMakeFiles/nn.dir/src/main.cpp.i
.PHONY : src/main.cpp.i

src/main.s: src/main.cpp.s
.PHONY : src/main.s

# target to generate assembly for a file
src/main.cpp.s:
	$(MAKE) -f CMakeFiles/nn.dir/build.make CMakeFiles/nn.dir/src/main.cpp.s
.PHONY : src/main.cpp.s

src/net.o: src/net.cpp.o
.PHONY : src/net.o

# target to build an object file
src/net.cpp.o:
	$(MAKE) -f CMakeFiles/nn.dir/build.make CMakeFiles/nn.dir/src/net.cpp.o
.PHONY : src/net.cpp.o

src/net.i: src/net.cpp.i
.PHONY : src/net.i

# target to preprocess a source file
src/net.cpp.i:
	$(MAKE) -f CMakeFiles/nn.dir/build.make CMakeFiles/nn.dir/src/net.cpp.i
.PHONY : src/net.cpp.i

src/net.s: src/net.cpp.s
.PHONY : src/net.s

# target to generate assembly for a file
src/net.cpp.s:
	$(MAKE) -f CMakeFiles/nn.dir/build.make CMakeFiles/nn.dir/src/net.cpp.s
.PHONY : src/net.cpp.s

src/tools.o: src/tools.cpp.o
.PHONY : src/tools.o

# target to build an object file
src/tools.cpp.o:
	$(MAKE) -f CMakeFiles/nn.dir/build.make CMakeFiles/nn.dir/src/tools.cpp.o
.PHONY : src/tools.cpp.o

src/tools.i: src/tools.cpp.i
.PHONY : src/tools.i

# target to preprocess a source file
src/tools.cpp.i:
	$(MAKE) -f CMakeFiles/nn.dir/build.make CMakeFiles/nn.dir/src/tools.cpp.i
.PHONY : src/tools.cpp.i

src/tools.s: src/tools.cpp.s
.PHONY : src/tools.s

# target to generate assembly for a file
src/tools.cpp.s:
	$(MAKE) -f CMakeFiles/nn.dir/build.make CMakeFiles/nn.dir/src/tools.cpp.s
.PHONY : src/tools.cpp.s

# Help Target
help:
	@echo "The following are some of the valid targets for this Makefile:"
	@echo "... all (the default if no target is provided)"
	@echo "... clean"
	@echo "... depend"
	@echo "... install/local"
	@echo "... edit_cache"
	@echo "... install"
	@echo "... rebuild_cache"
	@echo "... nn"
	@echo "... list_install_components"
	@echo "... src/functions.o"
	@echo "... src/functions.i"
	@echo "... src/functions.s"
	@echo "... src/layer.o"
	@echo "... src/layer.i"
	@echo "... src/layer.s"
	@echo "... src/main.o"
	@echo "... src/main.i"
	@echo "... src/main.s"
	@echo "... src/net.o"
	@echo "... src/net.i"
	@echo "... src/net.s"
	@echo "... src/tools.o"
	@echo "... src/tools.i"
	@echo "... src/tools.s"
.PHONY : help



#=============================================================================
# Special targets to cleanup operation of make.

# Special rule to run CMake to check the build system integrity.
# No rule that depends on this can have commands that come from listfiles
# because they might be regenerated.
cmake_check_build_system:
	$(CMAKE_COMMAND) -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 0
.PHONY : cmake_check_build_system

