CPP=g++
AR=ar
# CFLAGS debugging
CFLAGS=-std=c++17 -c -g -Wall -fpic
# CFLAGS execution
#CFLAGS=-std=c++17 -c -O3 -fpic
CODE_FOLDER=src/
INCLUDE_FOLDER=src/includes/KUNAI/
BIN_FOLDER=bin/
OBJ=objs/
BIN_NAME=Kunai
STATIC_LIB_NAME=kunai_lib.a
SHARED_LIB_NAME=kunai_lib.so


FILE_MODULES = -I ${INCLUDE_FOLDER}DEX/ -I ${INCLUDE_FOLDER}DEX/parser/ -I ${INCLUDE_FOLDER}DEX/DVM/ -I ${INCLUDE_FOLDER}DEX/Analysis/
UTILITIES = -I ${INCLUDE_FOLDER}Exceptions/ -I ${INCLUDE_FOLDER}Utils/
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

all: dirs ${BIN_FOLDER}${BIN_NAME} ${BIN_FOLDER}${STATIC_LIB_NAME} ${BIN_FOLDER}${SHARED_LIB_NAME} \
		${BIN_FOLDER}test_dex_parser

dirs:
	mkdir -p ${OBJ}
	mkdir -p ${BIN_FOLDER}

${BIN_FOLDER}${BIN_NAME}: ${OBJ}main.o ${OBJ_FILES}
	@echo "Linking $^ -> $@"
	${CPP} -o $@ $^

${BIN_FOLDER}${STATIC_LIB_NAME}: ${OBJ_FILES}
	@echo "Compiling static library $^ -> $@"
	$(AR) -crv $@ $^

${BIN_FOLDER}${SHARED_LIB_NAME}: ${OBJ_FILES}
	@echo "Compiling dynamic library $^ -> $@"
	$(CPP) -fpic -shared -Wformat=0 -o $@ $^

${BIN_FOLDER}test_dex_parser: ${OBJ}test_dex_parser.o ${OBJ_FILES}
	@echo "Linking $^ -> $@"
	${CPP} -o $@ $^
	
# main
${OBJ}main.o: ${CODE_FOLDER}main.cpp
	@echo "Compiling $< -> $@"
	${CPP} ${ALL_INCLUDE} -o $@ $< ${CFLAGS}

# test_dex_parser
${OBJ}test_dex_parser.o: ${CODE_FOLDER}test_dex_parser.cpp
	@echo "Compiling $< -> $@"
	${CPP} ${ALL_INCLUDE} -o $@ $< ${CFLAGS}

# Utils
UTILS_MODULE=Utils/
${OBJ}%.o: ${CODE_FOLDER}${UTILS_MODULE}%.cpp
	@echo "Compiling $< -> $@"
	${CPP} ${UTILITIES} -o $@ $< ${CFLAGS}

# DEX modules here
DEX_MODULE=DEX/
DEX_PARSER=DEX/parser/
DEX_DVM=DEX/DVM/
DEX_ANALYSIS=DEX/Analysis/
${OBJ}dex.o: ${CODE_FOLDER}${DEX_MODULE}dex.cpp
	@echo "Compiling $^ -> $@"
	${CPP} -I${INCLUDE_FOLDER}${DEX_MODULE} -I${INCLUDE_FOLDER}${DEX_PARSER} -I${INCLUDE_FOLDER}${DEX_DVM} -I${INCLUDE_FOLDER}${DEX_ANALYSIS} ${UTILITIES} -o $@ $^ ${CFLAGS}

${OBJ}%.o: ${CODE_FOLDER}${DEX_PARSER}%.cpp
	@echo "Compiling $< -> $@"
	${CPP} -I${INCLUDE_FOLDER}${DEX_PARSER} -I${INCLUDE_FOLDER}${DEX_DVM} ${UTILITIES} -o $@ $< ${CFLAGS}

${OBJ}%.o: ${CODE_FOLDER}${DEX_DVM}%.cpp
	@echo "Compiling $< -> $@"
	${CPP} -I${INCLUDE_FOLDER}${DEX_PARSER} -I${INCLUDE_FOLDER}${DEX_DVM} ${UTILITIES} -o $@ $< ${CFLAGS}

${OBJ}%.o: ${CODE_FOLDER}${DEX_ANALYSIS}%.cpp
	@echo "Compiling $< -> $@"
	${CPP} -I${INCLUDE_FOLDER}${DEX_PARSER} -I${INCLUDE_FOLDER}${DEX_DVM} -I${INCLUDE_FOLDER}${DEX_ANALYSIS} ${UTILITIES} -o $@ $< ${CFLAGS}



########################################################
clean:
	rm -rf ${OBJ}
	rm -rf ${BIN_FOLDER}
########################################################