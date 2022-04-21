// This is a generated source file for Chilkat version 9.5.0.89
#ifndef _C_CkByteData_H
#define _C_CkByteData_H
#include "chilkatDefs.h"

#include "Chilkat_C.h"


CK_C_VISIBLE_PUBLIC HCkByteData CkByteData_Create(void);
CK_C_VISIBLE_PUBLIC void CkByteData_Dispose(HCkByteData handle);
CK_C_VISIBLE_PUBLIC BOOL CkByteData_getSecureClear(HCkByteData cHandle);
CK_C_VISIBLE_PUBLIC void CkByteData_putSecureClear(HCkByteData cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC void CkByteData_append(HCkByteData cHandle, HCkByteData db);
CK_C_VISIBLE_PUBLIC void CkByteData_append2(HCkByteData cHandle, const unsigned char *pByteData, unsigned long szByteData);
CK_C_VISIBLE_PUBLIC void CkByteData_appendChar(HCkByteData cHandle, char ch);
CK_C_VISIBLE_PUBLIC void CkByteData_appendCharN(HCkByteData cHandle, char ch, int numTimes);
CK_C_VISIBLE_PUBLIC void CkByteData_appendEncoded(HCkByteData cHandle, const char *str, const char *encoding);
CK_C_VISIBLE_PUBLIC void CkByteData_appendEncodedW(HCkByteData cHandle, const wchar_t * str, const wchar_t * encoding);
CK_C_VISIBLE_PUBLIC BOOL CkByteData_appendFile(HCkByteData cHandle, const char *path);
CK_C_VISIBLE_PUBLIC BOOL CkByteData_appendFileW(HCkByteData cHandle, const wchar_t * path);
CK_C_VISIBLE_PUBLIC void CkByteData_appendInt(HCkByteData cHandle, int intValue, BOOL littleEndian);
CK_C_VISIBLE_PUBLIC void CkByteData_appendRandom(HCkByteData cHandle, int numBytes);
CK_C_VISIBLE_PUBLIC void CkByteData_appendRange(HCkByteData cHandle, HCkByteData byteData, unsigned long index, unsigned long numBytes);
CK_C_VISIBLE_PUBLIC void CkByteData_appendShort(HCkByteData cHandle, short shortValue, BOOL littleEndian);
CK_C_VISIBLE_PUBLIC void CkByteData_appendStr(HCkByteData cHandle, const char *str);
CK_C_VISIBLE_PUBLIC void CkByteData_appendStrW(HCkByteData cHandle, const wchar_t * str, const wchar_t * charset);
CK_C_VISIBLE_PUBLIC BOOL CkByteData_beginsWith(HCkByteData cHandle, HCkByteData byteDataObj);
CK_C_VISIBLE_PUBLIC BOOL CkByteData_beginsWith2(HCkByteData cHandle, const unsigned char *pByteData, unsigned long szByteData);
CK_C_VISIBLE_PUBLIC void CkByteData_borrowData(HCkByteData cHandle, const unsigned char *pByteData, unsigned long szByteData);
CK_C_VISIBLE_PUBLIC void CkByteData_byteSwap4321(HCkByteData cHandle);
CK_C_VISIBLE_PUBLIC void CkByteData_clear(HCkByteData cHandle);
CK_C_VISIBLE_PUBLIC void CkByteData_encode(HCkByteData cHandle, const char *encoding, HCkString str);
CK_C_VISIBLE_PUBLIC void CkByteData_encodeW(HCkByteData cHandle, const wchar_t * encoding, HCkString str);
CK_C_VISIBLE_PUBLIC BOOL CkByteData_ensureBuffer(HCkByteData cHandle, unsigned long expectedNumBytes);
CK_C_VISIBLE_PUBLIC BOOL CkByteData_equals(HCkByteData cHandle, HCkByteData compareBytes);
CK_C_VISIBLE_PUBLIC BOOL CkByteData_equals2(HCkByteData cHandle, const unsigned char *pCompareBytes, unsigned long numBytes);
CK_C_VISIBLE_PUBLIC int CkByteData_findBytes(HCkByteData cHandle, HCkByteData byteDataObj);
CK_C_VISIBLE_PUBLIC int CkByteData_findBytes2(HCkByteData cHandle, const unsigned char *findBytes, unsigned long findBytesLen);
CK_C_VISIBLE_PUBLIC unsigned char CkByteData_getByte(HCkByteData cHandle, unsigned long byteIndex);
CK_C_VISIBLE_PUBLIC const unsigned char *CkByteData_getBytes(HCkByteData cHandle);
CK_C_VISIBLE_PUBLIC char CkByteData_getChar(HCkByteData cHandle, unsigned long byteIndex);
CK_C_VISIBLE_PUBLIC const unsigned char *CkByteData_getData(HCkByteData cHandle);
CK_C_VISIBLE_PUBLIC const unsigned char *CkByteData_getDataAt(HCkByteData cHandle, unsigned long byteIndex);
CK_C_VISIBLE_PUBLIC const wchar_t * CkByteData_getEncodedW(HCkByteData cHandle, const wchar_t * encoding);
CK_C_VISIBLE_PUBLIC int CkByteData_getInt(HCkByteData cHandle, unsigned long byteIndex);
CK_C_VISIBLE_PUBLIC const unsigned char *CkByteData_getRange(HCkByteData cHandle, unsigned long byteIndex, unsigned long numBytes);
CK_C_VISIBLE_PUBLIC short CkByteData_getShort(HCkByteData cHandle, unsigned long byteIndex);
CK_C_VISIBLE_PUBLIC unsigned long CkByteData_getSize(HCkByteData cHandle);
CK_C_VISIBLE_PUBLIC unsigned long CkByteData_getUInt(HCkByteData cHandle, unsigned long byteIndex);
CK_C_VISIBLE_PUBLIC unsigned short CkByteData_getUShort(HCkByteData cHandle, unsigned long byteIndex);
CK_C_VISIBLE_PUBLIC BOOL CkByteData_is7bit(HCkByteData cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkByteData_loadFile(HCkByteData cHandle, const char *path);
CK_C_VISIBLE_PUBLIC BOOL CkByteData_loadFileW(HCkByteData cHandle, const wchar_t * path);
CK_C_VISIBLE_PUBLIC void CkByteData_pad(HCkByteData cHandle, int blockSize, int paddingScheme);
CK_C_VISIBLE_PUBLIC BOOL CkByteData_preAllocate(HCkByteData cHandle, unsigned long expectedNumBytes);
CK_C_VISIBLE_PUBLIC void CkByteData_removeChunk(HCkByteData cHandle, unsigned long startIndex, unsigned long numBytes);
CK_C_VISIBLE_PUBLIC const unsigned char *CkByteData_removeData(HCkByteData cHandle);
CK_C_VISIBLE_PUBLIC void CkByteData_replaceChar(HCkByteData cHandle, unsigned char existingByteValue, unsigned char replacementByteValue);
CK_C_VISIBLE_PUBLIC BOOL CkByteData_saveFile(HCkByteData cHandle, const char *path);
CK_C_VISIBLE_PUBLIC BOOL CkByteData_saveFileW(HCkByteData cHandle, const wchar_t * path);
CK_C_VISIBLE_PUBLIC void CkByteData_shorten(HCkByteData cHandle, unsigned long numBytes);
CK_C_VISIBLE_PUBLIC const wchar_t * CkByteData_to_ws(HCkByteData cHandle, const char *charset);
CK_C_VISIBLE_PUBLIC void CkByteData_unpad(HCkByteData cHandle, int blockSize, int paddingScheme);
#endif
