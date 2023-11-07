# Use the specific version of the iotta-fuzzer image as the base
FROM jeppojeps/iotta-fuzzer:4.3

# Update and install iputils-ping
RUN apt-get update && apt-get install -y iputils-ping && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /

# Clone the KUNAI-static-analyzer repo, checkout the refactoring branch in the 'old' directory, 
# run the build commands for both 'old' and 'kunai-lib' directories.
RUN git clone https://github.com/Fare9/KUNAI-static-analyzer.git && \
    cd KUNAI-static-analyzer/old && \
    git checkout refactoring && \
    cmake -B build -S . && \
    cmake --build build -j && \
    cd ../kunai-lib && \
    CXXFLAGS="-fno-rtti" cmake -S . -B build/ -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=Debug -DUSE_MJOLNIR=ON -DCMAKE_EXE_LINKER_FLAGS="-fuse-ld=ld" -DUNIT_TESTING=ON && \
    cmake --build build/ -j

# Set any default command for the container (if needed)
CMD ["/bin/bash"]

