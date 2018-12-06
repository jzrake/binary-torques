-include Makefile.in

CXXFLAGS = -std=c++14 -Wall -Wextra -Wno-missing-braces -O3

THIRD_P = src/visit_struct.hpp src/ndarray.hpp
HEADERS = $(filter-out $(THIRD_P), $(wildcard src/*.hpp))
SOURCES = $(wildcard src/*.cpp)
OBJECTS = $(SOURCES:.cpp=.o)

default: bt
src/main.o: $(HEADERS) $(THIRD_P)
src/app_utils.o: src/app_utils.hpp

bt: $(OBJECTS)
	$(CXX) -o $@ $^

src/visit_struct.hpp: third_party/visit_struct/include/visit_struct/visit_struct.hpp
	cp $< $@

src/ndarray.hpp: third_party/ndarray/include/ndarray.hpp
	cp $< $@

clean:
	$(RM) src/*.o bt
