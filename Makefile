CPP=g++
# CFLAGS debugging
#CFLAGS=-std=c++17 -c -g -Wall
# CFLAGS execution
CFLAGS=-std=c++17 -c -O3
BIN_FOLDER=bin/
OBJ=objs/
BIN_NAME=Kunai
FILE_MODULES = -I DEX/ -I DEX/parser/ -I DEX/DVM/ -I DEX/Analysis/
UTILITIES = -I Exceptions/ -I Utils/
ALL_INCLUDE = ${FILE_MODULES} ${UTILITIES}
OBJ_FILES= ${OBJ}utils.o ${OBJ}dex_header.o ${OBJ}dex_strings.o \
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

.PHONY: clean

all: dirs ${BIN_FOLDER}${BIN_NAME} ${BIN_FOLDER}test_dex_parser

dirs:
	mkdir -p ${OBJ}
	mkdir -p ${BIN_FOLDER}

${BIN_FOLDER}${BIN_NAME}: ${OBJ}main.o ${OBJ_FILES}
	@echo "Linking $^ -> $@"
	${CPP} -o $@ $^

${BIN_FOLDER}test_dex_parser: ${OBJ}test_dex_parser.o ${OBJ_FILES}
	@echo "Linking $^ -> $@"
	${CPP} -o $@ $^
	
# main
${OBJ}main.o: main.cpp
	@echo "Compiling $< -> $@"
	${CPP} ${ALL_INCLUDE} -o $@ $< ${CFLAGS}

# test_dex_parser
${OBJ}test_dex_parser.o: test_dex_parser.cpp
	@echo "Compiling $< -> $@"
	${CPP} ${ALL_INCLUDE} -o $@ $< ${CFLAGS}

# Utils
UTILS_MODULE=Utils/
${OBJ}%.o: ${UTILS_MODULE}%.cpp
	@echo "Compiling $< -> $@"
	${CPP} ${UTILITIES} -o $@ $< ${CFLAGS}

# DEX modules here
DEX_MODULE=DEX/
DEX_PARSER=DEX/parser/
DEX_DVM=DEX/DVM/
DEX_ANALYSIS=DEX/Analysis/
${OBJ}dex.o: ${DEX_MODULE}dex.cpp
	@echo "Compiling $^ -> $@"
	${CPP} -I${DEX_MODULE} -I${DEX_PARSER} -I${DEX_DVM} -I${DEX_ANALYSIS} ${UTILITIES} -o $@ $^ ${CFLAGS}

${OBJ}%.o: ${DEX_PARSER}%.cpp
	@echo "Compiling $< -> $@"
	${CPP} -I${DEX_PARSER} -I${DEX_DVM} ${UTILITIES} -o $@ $< ${CFLAGS}

${OBJ}%.o: ${DEX_DVM}%.cpp
	@echo "Compiling $< -> $@"
	${CPP} -I${DEX_PARSER} -I${DEX_DVM} ${UTILITIES} -o $@ $< ${CFLAGS}

${OBJ}%.o: ${DEX_ANALYSIS}%.cpp
	@echo "Compiling $< -> $@"
	${CPP} -I${DEX_PARSER} -I${DEX_DVM} -I${DEX_ANALYSIS} ${UTILITIES} -o $@ $< ${CFLAGS}



########################################################
clean:
	rm -rf ${OBJ}
	rm -rf ${BIN_FOLDER}
########################################################