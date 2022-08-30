CXX ?= g++
AR=ar
# set your paths
JAVAC=javac
D8=d8

# libchilkat data if not set
LIB_ZIP_PATH ?= ./external/zip/build/libzip.so
LIB_ZIP ?= -lzip
INCLUDE_ZIP_PATH ?= ./external/zip/src/
INCLUDE_ZIP = -I ${INCLUDE_ZIP_PATH}
CFLAGS=-std=c++17 -c -fpic
OPTIMIZATION ?= -O3

CODE_FOLDER=src/
CODE_TEST_FOLDER=./projects/
INCLUDE_FOLDER=src/include/KUNAI/
BIN_FOLDER=bin/
BIN_PROJECTS_FOLDER=bin/projects/
OBJ=objs/
BIN_NAME=Kunai
STATIC_LIB_NAME=libkunai.a
SHARED_LIB_NAME=libkunai.so

INCLUDE_PATH=src/include/


DEX_OBJ_FILES = ${OBJ}dex_header.o ${OBJ}dex_strings.o \
			${OBJ}dex_types.o ${OBJ}dex_protos.o ${OBJ}dex_fields.o \
			${OBJ}dex_methods.o ${OBJ}dex_classes.o ${OBJ}dex_encoded.o\
			${OBJ}dex_annotations.o ${OBJ}dex_parser.o\
			${OBJ}dex_dalvik_opcodes.o ${OBJ}dex_instructions.o\
			${OBJ}dex_linear_sweep_disassembly.o ${OBJ}dex_recursive_traversal_disassembly.o ${OBJ}dex_disassembler.o\
			${OBJ}dex_external_methods.o ${OBJ}dex_external_classes.o\
			${OBJ}dex_string_analysis.o ${OBJ}dex_dvm_basic_block.o\
			${OBJ}dex_exception_analysis.o\
			${OBJ}dex_method_analysis.o ${OBJ}dex_field_analysis.o\
			${OBJ}dex_class_analysis.o ${OBJ}dex_analysis.o\
			${OBJ}dex.o

APK_OBJ_FILES = ${OBJ}apk.o

IR_OBJ_FILES = ${OBJ}ir_type.o ${OBJ}ir_expr.o ${OBJ}ir_stmnt.o ${OBJ}ir_blocks.o \
				${OBJ}ir_graph.o ${OBJ}ir_utils.o ${OBJ}single_instruction_optimizations.o ${OBJ}single_block_optimizations.o \
				${OBJ}optimizer.o ${OBJ}reachingDefinition.o ${OBJ}ir_graph_ssa.o
IR_LIFTERS_OBJ_FILES = ${OBJ}lifter_android.o

OBJ_FILES= ${OBJ}utils.o ${DEX_OBJ_FILES} ${APK_OBJ_FILES} ${IR_OBJ_FILES} ${IR_LIFTERS_OBJ_FILES}

.PHONY: clean
.PHONY: tests


all: dirs ${BIN_FOLDER}${BIN_NAME} ${BIN_FOLDER}${STATIC_LIB_NAME} ${BIN_FOLDER}${SHARED_LIB_NAME} \
${BIN_PROJECTS_FOLDER}test_dex_parser ${BIN_PROJECTS_FOLDER}test_dex_disassembler ${BIN_PROJECTS_FOLDER}test_ir ${BIN_PROJECTS_FOLDER}test_dex_lifter \
${BIN_PROJECTS_FOLDER}test_ir_graph ${BIN_PROJECTS_FOLDER}test_dominators ${BIN_PROJECTS_FOLDER}test-optimizations \
${BIN_PROJECTS_FOLDER}test-optimizations2 ${BIN_PROJECTS_FOLDER}test_ssa_form



dirs:
	mkdir -p ${OBJ}
	mkdir -p ${BIN_FOLDER}
	mkdir -p ${BIN_PROJECTS_FOLDER}

${BIN_FOLDER}${BIN_NAME}: ${OBJ}main.o ${OBJ_FILES}
	@echo "Linking $< -> $@"
	${CXX} -o $@ $^ ${LIB_ZIP}
	
${BIN_FOLDER}${STATIC_LIB_NAME}: ${OBJ_FILES}
	@echo "Linking static library $@"
	$(AR) -crv $@ $^
	
${BIN_FOLDER}${SHARED_LIB_NAME}: ${OBJ_FILES}
	@echo "Linking dynamic library $@"
	${CXX} -fpic -shared -Wformat=0 -o $@ $^ ${LIB_ZIP}
	
####################################################################
#  				Test Files
####################################################################

${BIN_PROJECTS_FOLDER}test_dex_parser: ${OBJ}test_dex_parser.o ${OBJ_FILES}
	@echo "Linking $< -> $@"
	${CXX} -o $@ $^ ${LIB_ZIP}
	
${BIN_PROJECTS_FOLDER}test_dex_disassembler: ${OBJ}test_dex_disassembler.o ${OBJ_FILES}
	@echo "Linking $< -> $@"
	${CXX} -o $@ $^ ${LIB_ZIP}
	
${BIN_PROJECTS_FOLDER}test_ir: ${OBJ}test_ir.o ${OBJ_FILES}
	@echo "Linking $< -> $@"
	${CXX} -o $@ $^ ${LIB_ZIP}
	
${BIN_PROJECTS_FOLDER}test_dex_lifter: ${OBJ}test_dex_lifter.o ${OBJ_FILES}
	@echo "Linking $< -> $@"
	${CXX} -o $@ $^ ${LIB_ZIP}

${BIN_PROJECTS_FOLDER}test_ir_graph: ${OBJ}test_ir_graph.o ${OBJ_FILES}
	@echo "Linking $< -> $@"
	${CXX} -o $@ $^ ${LIB_ZIP}

${BIN_PROJECTS_FOLDER}test_dominators: ${OBJ}test_dominators.o ${OBJ_FILES}
	@echo "Linking $< -> $@"
	${CXX} -o $@ $^ ${LIB_ZIP}

${BIN_PROJECTS_FOLDER}test-optimizations: ${OBJ}test-optimizations.o ${OBJ_FILES}
	@echo "Linking $< -> $@"
	${CXX} -o $@ $^ ${LIB_ZIP}

${BIN_PROJECTS_FOLDER}test-optimizations2: ${OBJ}test-optimizations2.o ${OBJ_FILES}
	@echo "Linking $< -> $@"
	${CXX} -o $@ $^ ${LIB_ZIP}

${BIN_PROJECTS_FOLDER}test_ssa_form: ${OBJ}test_ssa_form.o ${OBJ_FILES}
	@echo "Linking $< -> $@"
	${CXX} -o $@ $^ ${LIB_ZIP}

${BIN_TEST_FOLDER}test_apk_analysis: ${OBJ}test_apk_analysis.o ${OBJ_FILES}
	@echo "Linking $< -> $@"
	${CXX} -o $@ $^ ${LIB_ZIP}

####################################################################


# main
${OBJ}main.o: ${CODE_FOLDER}main.cpp
	@echo "Compiling $< -> $@"
	${CXX} -I${INCLUDE_PATH} -o $@ $< ${CFLAGS} $(OPTIMIZATION) $(DEBUG)
	
####################################################################
#  				Test Files
####################################################################

# test_dex_parser
${OBJ}test_dex_parser.o: ${CODE_TEST_FOLDER}test_dex_parser.cpp
	@echo "Compiling $< -> $@"
	${CXX} -I${INCLUDE_PATH} -o $@ $< ${CFLAGS} $(OPTIMIZATION) $(DEBUG)
	
# test_dex_disassembler
${OBJ}test_dex_disassembler.o: ${CODE_TEST_FOLDER}test_dex_disassembler.cpp
	@echo "Compiling $< -> $@"
	${CXX} -I${INCLUDE_PATH} -o $@ $< ${CFLAGS} $(OPTIMIZATION) $(DEBUG)
	
# test IR
${OBJ}test_ir.o: ${CODE_TEST_FOLDER}test_ir.cpp
	@echo "Compiling $< -> $@"
	${CXX} -I${INCLUDE_PATH} -o $@ $< ${CFLAGS} $(OPTIMIZATION) $(DEBUG)

# test dex lifter
${OBJ}test_dex_lifter.o: ${CODE_TEST_FOLDER}test_dex_lifter.cpp
	@echo "Compiling $< -> $@"
	${CXX} -I${INCLUDE_PATH} -o $@ $< ${CFLAGS} $(OPTIMIZATION) $(DEBUG)

${OBJ}test_ir_graph.o: ${CODE_TEST_FOLDER}test_ir_graph.cpp
	@echo "Compiling $< -> $@"
	${CXX} -I${INCLUDE_PATH} -o $@ $< ${CFLAGS} $(OPTIMIZATION) $(DEBUG)

${OBJ}test_dominators.o: ${CODE_TEST_FOLDER}test_dominators.cpp
	@echo "Compiling $< -> $@"
	${CXX} -I${INCLUDE_PATH} -o $@ $< ${CFLAGS} $(OPTIMIZATION) $(DEBUG)

${OBJ}test_apk_analysis.o: ${CODE_TEST_FOLDER}test_apk_analysis.cpp
	@echo "Compiling $< -> $@"
	${CXX} -I${INCLUDE_PATH} ${INCLUDE_ZIP} -o $@ $< ${CFLAGS} $(OPTIMIZATION) $(DEBUG)

${OBJ}test-optimizations.o: ${CODE_TEST_FOLDER}test-optimizations.cpp
	@echo "Compiling $< -> $@"
	${CXX} -I${INCLUDE_PATH} ${INCLUDE_ZIP} -o $@ $< ${CFLAGS} $(OPTIMIZATION) $(DEBUG)

${OBJ}test-optimizations2.o: ${CODE_TEST_FOLDER}test-optimizations2.cpp
	@echo "Compiling $< -> $@"
	${CXX} -I${INCLUDE_PATH} ${INCLUDE_ZIP} -o $@ $< ${CFLAGS} $(OPTIMIZATION) $(DEBUG)

${OBJ}test_ssa_form.o: ${CODE_TEST_FOLDER}test_ssa_form.cpp
	@echo "Compiling $< -> $@"
	${CXX} -I${INCLUDE_PATH} ${INCLUDE_ZIP} -o $@ $< ${CFLAGS} $(OPTIMIZATION) $(DEBUG)

####################################################################
	
# Utils
UTILS_MODULE=Utils/
${OBJ}%.o: ${CODE_FOLDER}${UTILS_MODULE}%.cpp
	@echo "Compiling $< -> $@"
	${CXX} -I${INCLUDE_PATH} -o $@ $< ${CFLAGS} $(OPTIMIZATION) $(DEBUG)
	
# DEX modules here
DEX_MODULE=DEX/
DEX_PARSER=DEX/parser/
DEX_DVM=DEX/DVM/
DEX_ANALYSIS=DEX/Analysis/
${OBJ}dex.o: ${CODE_FOLDER}${DEX_MODULE}dex.cpp
	@echo "Compiling $^ -> $@"
	${CXX} -I${INCLUDE_PATH} -o $@ $^ ${CFLAGS} $(OPTIMIZATION) $(DEBUG)
	
${OBJ}%.o: ${CODE_FOLDER}${DEX_PARSER}%.cpp
	@echo "Compiling $< -> $@"
	${CXX} -I${INCLUDE_PATH} -o $@ $< ${CFLAGS} $(OPTIMIZATION) $(DEBUG)
	
${OBJ}%.o: ${CODE_FOLDER}${DEX_DVM}%.cpp
	@echo "Compiling $< -> $@"
	${CXX} -I${INCLUDE_PATH} -o $@ $< ${CFLAGS} $(OPTIMIZATION) $(DEBUG)	

${OBJ}%.o: ${CODE_FOLDER}${DEX_ANALYSIS}%.cpp
	@echo "Compiling $< -> $@"
	${CXX} -I${INCLUDE_PATH} -o $@ $< ${CFLAGS} $(OPTIMIZATION) $(DEBUG)

# APK modules here
APK_MODULE=APK/
${OBJ}%.o: ${CODE_FOLDER}${APK_MODULE}%.cpp
	@echo "Compiling $< -> $@"
	${CXX} -I${INCLUDE_PATH} -o $@ $< ${CFLAGS} $(OPTIMIZATION) $(DEBUG) ${LIB_ZIP}

# IR modules here
IR_MODULE=mjolnIR/
IR_LIFTERS=mjolnIR/Lifters/
IR_ANALYSIS=mjolnIR/Analysis/
${OBJ}%.o: ${CODE_FOLDER}${IR_MODULE}%.cpp
	@echo "Compiling $< -> $@"
	${CXX} -I${INCLUDE_PATH} -o $@ $< ${CFLAGS} $(OPTIMIZATION) $(DEBUG)
	
${OBJ}%.o: ${CODE_FOLDER}${IR_LIFTERS}%.cpp
	@echo "Compiling $< -> $@"
	${CXX} -I${INCLUDE_PATH} -o $@ $< ${CFLAGS} $(OPTIMIZATION) $(DEBUG)

${OBJ}%.o: ${CODE_FOLDER}${IR_ANALYSIS}%.cpp
	@echo "Compiling $< -> $@"
	${CXX} -I${INCLUDE_PATH} -o $@ $< ${CFLAGS} $(OPTIMIZATION) $(DEBUG)
	
# Compile tests
tests:
	current_dir=$(shell pwd)

	@echo "Compiling test-assignment-arith-logic"
	cd ./tests/test-assignment-arith-logic/ && ${JAVAC} --release 8 Main.java && ${D8} Main.class && mv classes.dex Main.dex

	@echo "Compiling test-const_class"
	cd ./tests/test-const_class/ && ${JAVAC} --release 8 Main.java && ${D8} Main.class && mv classes.dex Main.dex

	@echo "Compiling test-try-catch"
	cd ./tests/test-try-catch/ && ${JAVAC} --release 8 Main.java && ${D8} Main.class && mv classes.dex Main.dex

	@echo "Compiling test-graph"
	cd ./tests/test-graph/ && ${JAVAC} --release 8 Main.java && ${D8} Main.class && mv classes.dex Main.dex

	@echo "Compiling test-cyclomatic-complexity"
	cd ./tests/test-cyclomatic-complexity/ && ${JAVAC} --release 8 Main.java && ${D8} Main.class && mv classes.dex Main.dex

	@echo "Compiling test-vm"
	cd ./tests/test-vm/ && ${JAVAC} --release 8 PCodeVM.java VClass.java && \
		${D8} VClass.class && \
		mv classes.dex VClass.dex &&\
		${D8} PCodeVM.class &&\
		mv classes.dex PCodeVM.dex
	
	@echo "Compiling test-modexp"
	cd ./tests/test-modexp && ${JAVAC} --release 8 Main.java && ${D8} Main.class && mv classes.dex Main.dex

########################################################
clean:
	rm -rf ${OBJ}
	rm -rf ${BIN_FOLDER}
########################################################

########################################################
install:
	@echo "Removing previous folders"
	sudo rm -rf /usr/include/KUNAI
	sudo rm -f /usr/lib/${SHARED_LIB_NAME}
	sudo rm -f /usr/lib/${STATIC_LIB_NAME}
	@echo "Copying libs to /usr/lib"
	sudo cp ${BIN_FOLDER}${STATIC_LIB_NAME} /usr/lib
	sudo cp ${BIN_FOLDER}${SHARED_LIB_NAME} /usr/lib
	sudo cp ${LIB_ZIP_PATH} /usr/lib
	@echo "Creating /usr/include/KUNAI and copying header files"
	sudo cp -r ${INCLUDE_FOLDER} /usr/include/
########################################################

########################################################
remove:
	sudo rm -rf /usr/include/KUNAI
	sudo rm -f /usr/lib/${SHARED_LIB_NAME}
	sudo rm -f /usr/lib/${STATIC_LIB_NAME}
########################################################


debug:
	$(MAKE) $(MAKEFILE) OPTIMIZATION="-O0" DEBUG="-g -D DEBUG" 