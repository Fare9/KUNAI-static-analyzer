CXX ?= g++
AR=ar
# set your paths
JAVAC=javac
DX=dx

# libchilkat data if not set
LIB_CHILKAT ?= -lchilkat-9.5.0
INCLUDE_CHILKAT_PATH ?= ./external/chilkat-x86_64-linux-gcc/include/
INCLUDE_CHILKAT = -I ${INCLUDE_CHILKAT_PATH}
CFLAGS=-std=c++17 -c -fpic
OPTIMIZATION ?= -O3

CODE_FOLDER=src/
CODE_TEST_FOLDER=./projects/
INCLUDE_FOLDER=src/includes/KUNAI/
BIN_FOLDER=bin/
BIN_PROJECTS_FOLDER=bin/projects/
OBJ=objs/
BIN_NAME=Kunai
STATIC_LIB_NAME=libkunai.a
SHARED_LIB_NAME=libkunai.so


DEX_MODULES_INCLUDE = -I ${INCLUDE_FOLDER}DEX/ -I ${INCLUDE_FOLDER}DEX/parser/ -I ${INCLUDE_FOLDER}DEX/DVM/ -I ${INCLUDE_FOLDER}DEX/Analysis/
APK_MODULES_INCLUDE = -I ${INCLUDE_FOLDER}APK/
UTILITIES_INCLUDE = -I ${INCLUDE_FOLDER}Exceptions/ -I ${INCLUDE_FOLDER}Utils/
IR_MODULES_INCLUDE = -I ${INCLUDE_FOLDER}mjolnIR/ -I ${INCLUDE_FOLDER}mjolnIR/Lifters/ -I ${INCLUDE_FOLDER}mjolnIR/arch/
ALL_INCLUDE = ${DEX_MODULES_INCLUDE} ${APK_MODULES_INCLUDE} ${UTILITIES_INCLUDE} ${IR_MODULES_INCLUDE}

DEX_OBJ_FILES = ${OBJ}dex_header.o ${OBJ}dex_strings.o \
			${OBJ}dex_types.o ${OBJ}dex_protos.o ${OBJ}dex_fields.o \
			${OBJ}dex_methods.o ${OBJ}dex_classes.o ${OBJ}dex_encoded.o\
			${OBJ}dex_annotations.o ${OBJ}dex_parser.o\
			${OBJ}dex_dalvik_opcodes.o ${OBJ}dex_instructions.o\
			${OBJ}dex_linear_sweep_disassembly.o ${OBJ}dex_disassembler.o\
			${OBJ}dex_external_methods.o ${OBJ}dex_external_classes.o\
			${OBJ}dex_string_analysis.o ${OBJ}dex_dvm_basic_block.o\
			${OBJ}dex_exception_analysis.o\
			${OBJ}dex_method_analysis.o ${OBJ}dex_field_analysis.o\
			${OBJ}dex_class_analysis.o ${OBJ}dex_analysis.o\
			${OBJ}dex.o

APK_OBJ_FILES = ${OBJ}apk.o

IR_OBJ_FILES = ${OBJ}ir_type.o ${OBJ}ir_expr.o ${OBJ}ir_stmnt.o ${OBJ}ir_blocks.o ${OBJ}ir_graph.o ${OBJ}ir_utils.o ${OBJ}optimizer.o
IR_LIFTERS_OBJ_FILES = ${OBJ}lifter_android.o

OBJ_FILES= ${OBJ}utils.o ${DEX_OBJ_FILES} ${APK_OBJ_FILES} ${IR_OBJ_FILES} ${IR_LIFTERS_OBJ_FILES}

.PHONY: clean
.PHONY: tests


all: dirs ${BIN_FOLDER}${BIN_NAME} ${BIN_FOLDER}${STATIC_LIB_NAME} ${BIN_FOLDER}${SHARED_LIB_NAME} \
${BIN_PROJECTS_FOLDER}test_dex_parser ${BIN_PROJECTS_FOLDER}test_dex_disassembler ${BIN_PROJECTS_FOLDER}test_ir ${BIN_PROJECTS_FOLDER}test_dex_lifter \
${BIN_PROJECTS_FOLDER}test_ir_graph ${BIN_PROJECTS_FOLDER}test_dominators ${BIN_PROJECTS_FOLDER}test-optimizations



dirs:
	mkdir -p ${OBJ}
	mkdir -p ${BIN_FOLDER}
	mkdir -p ${BIN_PROJECTS_FOLDER}

${BIN_FOLDER}${BIN_NAME}: ${OBJ}main.o ${OBJ_FILES}
	@echo "Linking $< -> $@"
	${CXX} -o $@ $^ ${LIB_CHILKAT}
	
${BIN_FOLDER}${STATIC_LIB_NAME}: ${OBJ_FILES}
	@echo "Linking static library $@"
	$(AR) -crv $@ $^
	
${BIN_FOLDER}${SHARED_LIB_NAME}: ${OBJ_FILES}
	@echo "Linking dynamic library $@"
	${CXX} -fpic -shared -Wformat=0 -o $@ $^ ${LIB_CHILKAT}
	
####################################################################
#  				Test Files
####################################################################

${BIN_PROJECTS_FOLDER}test_dex_parser: ${OBJ}test_dex_parser.o ${OBJ_FILES}
	@echo "Linking $< -> $@"
	${CXX} -o $@ $^ ${LIB_CHILKAT}
	
${BIN_PROJECTS_FOLDER}test_dex_disassembler: ${OBJ}test_dex_disassembler.o ${OBJ_FILES}
	@echo "Linking $< -> $@"
	${CXX} -o $@ $^ ${LIB_CHILKAT}
	
${BIN_PROJECTS_FOLDER}test_ir: ${OBJ}test_ir.o ${OBJ_FILES}
	@echo "Linking $< -> $@"
	${CXX} -o $@ $^ ${LIB_CHILKAT}
	
${BIN_PROJECTS_FOLDER}test_dex_lifter: ${OBJ}test_dex_lifter.o ${OBJ_FILES}
	@echo "Linking $< -> $@"
	${CXX} -o $@ $^ ${LIB_CHILKAT}

${BIN_PROJECTS_FOLDER}test_ir_graph: ${OBJ}test_ir_graph.o ${OBJ_FILES}
	@echo "Linking $< -> $@"
	${CXX} -o $@ $^ ${LIB_CHILKAT}

${BIN_PROJECTS_FOLDER}test_dominators: ${OBJ}test_dominators.o ${OBJ_FILES}
	@echo "Linking $< -> $@"
	${CXX} -o $@ $^ ${LIB_CHILKAT}

${BIN_PROJECTS_FOLDER}test-optimizations: ${OBJ}test-optimizations.o ${OBJ_FILES}
	@echo "Linking $< -> $@"
	${CXX} -o $@ $^ ${LIB_CHILKAT}

${BIN_TEST_FOLDER}test_apk_analysis: ${OBJ}test_apk_analysis.o ${OBJ_FILES}
	@echo "Linking $< -> $@"
	${CXX} -o $@ $^ ${LIB_CHILKAT}

####################################################################


# main
${OBJ}main.o: ${CODE_FOLDER}main.cpp
	@echo "Compiling $< -> $@"
	${CXX} ${ALL_INCLUDE} -o $@ $< ${CFLAGS} $(OPTIMIZATION) $(DEBUG)
	
####################################################################
#  				Test Files
####################################################################

# test_dex_parser
${OBJ}test_dex_parser.o: ${CODE_TEST_FOLDER}test_dex_parser.cpp
	@echo "Compiling $< -> $@"
	${CXX} ${ALL_INCLUDE} -o $@ $< ${CFLAGS} $(OPTIMIZATION) $(DEBUG)
	
# test_dex_disassembler
${OBJ}test_dex_disassembler.o: ${CODE_TEST_FOLDER}test_dex_disassembler.cpp
	@echo "Compiling $< -> $@"
	${CXX} ${ALL_INCLUDE} -o $@ $< ${CFLAGS} $(OPTIMIZATION) $(DEBUG)
	
# test IR
${OBJ}test_ir.o: ${CODE_TEST_FOLDER}test_ir.cpp
	@echo "Compiling $< -> $@"
	${CXX} ${ALL_INCLUDE} -o $@ $< ${CFLAGS} $(OPTIMIZATION) $(DEBUG)

# test dex lifter
${OBJ}test_dex_lifter.o: ${CODE_TEST_FOLDER}test_dex_lifter.cpp
	@echo "Compiling $< -> $@"
	${CXX} ${ALL_INCLUDE} -o $@ $< ${CFLAGS} $(OPTIMIZATION) $(DEBUG)

${OBJ}test_ir_graph.o: ${CODE_TEST_FOLDER}test_ir_graph.cpp
	@echo "Compiling $< -> $@"
	${CXX} ${ALL_INCLUDE} -o $@ $< ${CFLAGS} $(OPTIMIZATION) $(DEBUG)

${OBJ}test_dominators.o: ${CODE_TEST_FOLDER}test_dominators.cpp
	@echo "Compiling $< -> $@"
	${CXX} ${ALL_INCLUDE} -o $@ $< ${CFLAGS} $(OPTIMIZATION) $(DEBUG)

${OBJ}test_apk_analysis.o: ${CODE_TEST_FOLDER}test_apk_analysis.cpp
	@echo "Compiling $< -> $@"
	${CXX} ${ALL_INCLUDE} ${INCLUDE_CHILKAT} -o $@ $< ${CFLAGS} $(OPTIMIZATION) $(DEBUG)

${OBJ}test-optimizations.o: ${CODE_TEST_FOLDER}test-optimizations.cpp
	@echo "Compiling $< -> $@"
	${CXX} ${ALL_INCLUDE} ${INCLUDE_CHILKAT} -o $@ $< ${CFLAGS} $(OPTIMIZATION) $(DEBUG)

####################################################################
	
# Utils
UTILS_MODULE=Utils/
${OBJ}%.o: ${CODE_FOLDER}${UTILS_MODULE}%.cpp
	@echo "Compiling $< -> $@"
	${CXX} ${UTILITIES_INCLUDE} -o $@ $< ${CFLAGS} $(OPTIMIZATION) $(DEBUG)
	
# DEX modules here
DEX_MODULE=DEX/
DEX_PARSER=DEX/parser/
DEX_DVM=DEX/DVM/
DEX_ANALYSIS=DEX/Analysis/
${OBJ}dex.o: ${CODE_FOLDER}${DEX_MODULE}dex.cpp
	@echo "Compiling $^ -> $@"
	${CXX} -I${INCLUDE_FOLDER}${DEX_MODULE} -I${INCLUDE_FOLDER}${DEX_PARSER} -I${INCLUDE_FOLDER}${DEX_DVM} -I${INCLUDE_FOLDER}${DEX_ANALYSIS} ${UTILITIES_INCLUDE} -o $@ $^ ${CFLAGS} $(OPTIMIZATION) $(DEBUG)
	
${OBJ}%.o: ${CODE_FOLDER}${DEX_PARSER}%.cpp
	@echo "Compiling $< -> $@"
	${CXX} -I${INCLUDE_FOLDER}${DEX_ANALYSIS} -I${INCLUDE_FOLDER}${DEX_PARSER} -I${INCLUDE_FOLDER}${DEX_DVM} ${UTILITIES_INCLUDE} -o $@ $< ${CFLAGS} $(OPTIMIZATION) $(DEBUG)
	
${OBJ}%.o: ${CODE_FOLDER}${DEX_DVM}%.cpp
	@echo "Compiling $< -> $@"
	${CXX} -I${INCLUDE_FOLDER}${DEX_ANALYSIS} -I${INCLUDE_FOLDER}${DEX_PARSER} -I${INCLUDE_FOLDER}${DEX_DVM} ${UTILITIES_INCLUDE} -o $@ $< ${CFLAGS} $(OPTIMIZATION) $(DEBUG)	

${OBJ}%.o: ${CODE_FOLDER}${DEX_ANALYSIS}%.cpp
	@echo "Compiling $< -> $@"
	${CXX} -I${INCLUDE_FOLDER}${DEX_PARSER} -I${INCLUDE_FOLDER}${DEX_DVM} -I${INCLUDE_FOLDER}${DEX_ANALYSIS} ${UTILITIES_INCLUDE} -o $@ $< ${CFLAGS} $(OPTIMIZATION) $(DEBUG)

# APK modules here
APK_MODULE=APK/
${OBJ}%.o: ${CODE_FOLDER}${APK_MODULE}%.cpp
	@echo "Compiling $< -> $@"
	${CXX} ${DEX_MODULES_INCLUDE} ${UTILITIES_INCLUDE} ${APK_MODULES_INCLUDE} ${INCLUDE_CHILKAT} -o $@ $< ${CFLAGS} $(OPTIMIZATION) $(DEBUG) ${LIB_CHILKAT}

# IR modules here
IR_MODULE=mjolnIR/
IR_LIFTERS=mjolnIR/Lifters/
${OBJ}%.o: ${CODE_FOLDER}${IR_MODULE}%.cpp
	@echo "Compiling $< -> $@"
	${CXX} ${IR_MODULES_INCLUDE} ${DEX_MODULES_INCLUDE} ${UTILITIES_INCLUDE} -o $@ $< ${CFLAGS} $(OPTIMIZATION) $(DEBUG)
	
${OBJ}%.o: ${CODE_FOLDER}${IR_LIFTERS}%.cpp
	@echo "Compiling $< -> $@"
	${CXX} ${IR_MODULES_INCLUDE} ${DEX_MODULES_INCLUDE} ${UTILITIES_INCLUDE} -o $@ $< ${CFLAGS} $(OPTIMIZATION) $(DEBUG)

# Compile tests
tests:
	current_dir=$(shell pwd)

	@echo "Compiling test-assignment-arith-logic"
	cd ./tests/test-assignment-arith-logic/ && ${JAVAC} --release 8 Main.java && ${DX} --dex --output Main.dex Main.class

	@echo "Compiling test-const_class"
	cd ./tests/test-const_class/ && ${JAVAC} --release 8 Main.java && ${DX} --dex --output Main.dex Main.class

	@echo "Compiling test-try-catch"
	cd ./tests/test-try-catch/ && ${JAVAC} --release 8 Main.java && ${DX} --dex --output Main.dex Main.class

	@echo "Compiling test-graph"
	cd ./tests/test-graph/ && ${JAVAC} --release 8 Main.java && ${DX} --dex --output Main.dex Main.class

	@echo "Compiling test-cyclomatic-complexity"
	cd ./tests/test-cyclomatic-complexity/ && ${JAVAC} --release 8 Main.java && ${DX} --dex --output Main.dex Main.class


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
	@echo "Creating /usr/include/KUNAI and copying header files"
	sudo mkdir /usr/include/KUNAI
	sudo find ${INCLUDE_FOLDER} -name '*.hpp' -exec cp "{}" /usr/include/KUNAI \;
########################################################

########################################################
remove:
	sudo rm -rf /usr/include/KUNAI
	sudo rm -f /usr/lib/${SHARED_LIB_NAME}
	sudo rm -f /usr/lib/${STATIC_LIB_NAME}
########################################################


debug:
	$(MAKE) $(MAKEFILE) OPTIMIZATION="-O0" DEBUG="-g -D DEBUG" 