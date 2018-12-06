-include Makefile.in

CXXFLAGS = -std=c++14 -Wall -Wextra -Wno-missing-braces -O2

THIRD_P = src/visit_struct.hpp src/ndarray.hpp
HEADERS = $(filter-out $(THIRD_P), $(wildcard src/*.hpp))
SOURCES = src/main.cpp
OBJECTS = src/main.o

default: bt

src/main.o: $(HEADERS) $(THIRD_P)

bt: src/main.o
	$(CXX) -o $@ $<

src/visit_struct.hpp: third_party/visit_struct/include/visit_struct/visit_struct.hpp
	cp $< $@

src/ndarray.hpp: third_party/ndarray/include/ndarray.hpp
	cp $< $@

clean:
	$(RM) src/*.o bt
