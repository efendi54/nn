CMAKE_MINIMUM_REQUIRED( VERSION 2.8 )
PROJECT( nn CXX )
#SET( CMAKE_VERBOSE_MAKEFILE ON )

INCLUDE_DIRECTORIES("include")

FILE(GLOB NNSources "src/*.cpp" "src/util/*.cpp" "src/net/*.cpp" "src/data/*.cpp")
ADD_EXECUTABLE( nn ${NNSources} )

#install binaries and resources
#MAKE_EXECUTABLE( "nn" AUTO )
INSTALL(TARGETS "nn" DESTINATION . ) 
