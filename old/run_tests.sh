#!/usr/bin/env bash

LOG_FILE=run_tests_output.log

echo "[+] Running KUNAI tests"

if [ ! -d "./bin/projects/" ]; then
    echo "[-] Please compile KUNAI before running tests"
    exit 1
fi

echo "run_tests.sh output" > ${LOG_FILE}

echo "[+] Running 'test_ir' test"

if [ ! -f "./bin/projects/test_ir" ]; then
    echo "[-] File does not exists"
    exit 1
fi

echo "Log 'test_ir'" >> ${LOG_FILE}
./bin/projects/test_ir 2>> ${LOG_FILE}

if [ $? -eq 0 ]; then
    echo "[!] Test succeeded"
else
    echo "[-] Error passing test"
fi

echo "[+] Running 'test_ir_graph' test"

echo "Log 'test_ir_graph'" >> ${LOG_FILE}
./bin/projects/test_ir_graph ./tests/test-graph/Main.dex "LMain;" main "([Ljava/lang/String;)V" 2>> ${LOG_FILE}

if [ -f "./main.dot" ]; then
    echo "[!] Main.dot file was created correctly" >> ${LOG_FILE}
    rm ./main.dot
fi

if [ $? -eq 0 ]; then
    echo "[!] Test succeeded"
else
    echo "[-] Error passing test"
fi

echo "[+] Running 'test_dominators' test"

echo "Log 'test_dominators' test" >> ${LOG_FILE}

./bin/projects/test_dominators ./tests/test-graph/Main.dex "LMain;" main "([Ljava/lang/String;)V" 2>> ${LOG_FILE}

if [ -f "./idom.dot" ]; then
    echo "[!] idom.dot file was created correctly" >> ${LOG_FILE}
    rm ./idom.dot
fi

if [ $? -eq 0 ]; then
    echo "[!] Test succeeded"
else
    echo "[-] Error passing test"
fi