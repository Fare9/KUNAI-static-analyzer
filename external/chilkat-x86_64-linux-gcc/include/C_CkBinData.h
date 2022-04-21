// This is a generated source file for Chilkat version 9.5.0.89
#ifndef _C_CkBinData_H
#define _C_CkBinData_H
#include "chilkatDefs.h"

#include "Chilkat_C.h"


CK_C_VISIBLE_PUBLIC HCkBinData CkBinData_Create(void);
CK_C_VISIBLE_PUBLIC void CkBinData_Dispose(HCkBinData handle);
CK_C_VISIBLE_PUBLIC const unsigned char *CkBinData_getBinaryDataPtr(HCkBinData handle);
CK_C_VISIBLE_PUBLIC BOOL CkBinData_getLastMethodSuccess(HCkBinData cHandle);
CK_C_VISIBLE_PUBLIC void CkBinData_putLastMethodSuccess(HCkBinData cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC int CkBinData_getNumBytes(HCkBinData cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkBinData_getUtf8(HCkBinData cHandle);
CK_C_VISIBLE_PUBLIC void CkBinData_putUtf8(HCkBinData cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC BOOL CkBinData_AppendBd(HCkBinData cHandle, HCkBinData binData);
CK_C_VISIBLE_PUBLIC BOOL CkBinData_AppendBinary(HCkBinData cHandle, HCkByteData data);
CK_C_VISIBLE_PUBLIC BOOL CkBinData_AppendBinary2(HCkBinData cHandle, const unsigned char *pByteData, unsigned long szByteData);
CK_C_VISIBLE_PUBLIC BOOL CkBinData_AppendBom(HCkBinData cHandle, const char *charset);
CK_C_VISIBLE_PUBLIC BOOL CkBinData_AppendByte(HCkBinData cHandle, int byteValue);
CK_C_VISIBLE_PUBLIC BOOL CkBinData_AppendEncoded(HCkBinData cHandle, const char *encData, const char *encoding);
CK_C_VISIBLE_PUBLIC BOOL CkBinData_AppendEncodedSb(HCkBinData cHandle, HCkStringBuilder sb, const char *encoding);
CK_C_VISIBLE_PUBLIC BOOL CkBinData_AppendInt2(HCkBinData cHandle, int value, BOOL littleEndian);
CK_C_VISIBLE_PUBLIC BOOL CkBinData_AppendInt4(HCkBinData cHandle, int value, BOOL littleEndian);
CK_C_VISIBLE_PUBLIC BOOL CkBinData_AppendPadded(HCkBinData cHandle, const char *str, const char *charset, BOOL padWithSpace, int fieldLen);
CK_C_VISIBLE_PUBLIC BOOL CkBinData_AppendSb(HCkBinData cHandle, HCkStringBuilder sb, const char *charset);
CK_C_VISIBLE_PUBLIC BOOL CkBinData_AppendString(HCkBinData cHandle, const char *str, const char *charset);
CK_C_VISIBLE_PUBLIC BOOL CkBinData_Clear(HCkBinData cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkBinData_ContentsEqual(HCkBinData cHandle, HCkBinData binData);
CK_C_VISIBLE_PUBLIC int CkBinData_FindString(HCkBinData cHandle, const char *str, int startIdx, const char *charset);
CK_C_VISIBLE_PUBLIC BOOL CkBinData_GetBinary(HCkBinData cHandle, HCkByteData outBytes);
CK_C_VISIBLE_PUBLIC BOOL CkBinData_GetBinaryChunk(HCkBinData cHandle, int offset, int numBytes, HCkByteData outBytes);
CK_C_VISIBLE_PUBLIC const unsigned char *CkBinData_GetBytesPtr(HCkBinData cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkBinData_GetEncoded(HCkBinData cHandle, const char *encoding, HCkString outStr);
CK_C_VISIBLE_PUBLIC const char *CkBinData_getEncoded(HCkBinData cHandle, const char *encoding);
CK_C_VISIBLE_PUBLIC BOOL CkBinData_GetEncodedChunk(HCkBinData cHandle, int offset, int numBytes, const char *encoding, HCkString outStr);
CK_C_VISIBLE_PUBLIC const char *CkBinData_getEncodedChunk(HCkBinData cHandle, int offset, int numBytes, const char *encoding);
CK_C_VISIBLE_PUBLIC BOOL CkBinData_GetEncodedSb(HCkBinData cHandle, const char *encoding, HCkStringBuilder sb);
CK_C_VISIBLE_PUBLIC int CkBinData_GetInt2(HCkBinData cHandle, int index, BOOL littleEndian);
CK_C_VISIBLE_PUBLIC int CkBinData_GetInt4(HCkBinData cHandle, int index, BOOL littleEndian);
CK_C_VISIBLE_PUBLIC BOOL CkBinData_GetString(HCkBinData cHandle, const char *charset, HCkString outStr);
CK_C_VISIBLE_PUBLIC const char *CkBinData_getString(HCkBinData cHandle, const char *charset);
CK_C_VISIBLE_PUBLIC BOOL CkBinData_GetTextChunk(HCkBinData cHandle, int startIdx, int numBytes, const char *charset, HCkString outStr);
CK_C_VISIBLE_PUBLIC const char *CkBinData_getTextChunk(HCkBinData cHandle, int startIdx, int numBytes, const char *charset);
CK_C_VISIBLE_PUBLIC unsigned long CkBinData_GetUInt2(HCkBinData cHandle, int index, BOOL littleEndian);
CK_C_VISIBLE_PUBLIC unsigned long CkBinData_GetUInt4(HCkBinData cHandle, int index, BOOL littleEndian);
CK_C_VISIBLE_PUBLIC BOOL CkBinData_LoadBinary(HCkBinData cHandle, HCkByteData data);
CK_C_VISIBLE_PUBLIC BOOL CkBinData_LoadBinary2(HCkBinData cHandle, const unsigned char *pByteData, unsigned long szByteData);
CK_C_VISIBLE_PUBLIC BOOL CkBinData_LoadEncoded(HCkBinData cHandle, const char *encData, const char *encoding);
CK_C_VISIBLE_PUBLIC BOOL CkBinData_LoadFile(HCkBinData cHandle, const char *path);
CK_C_VISIBLE_PUBLIC BOOL CkBinData_RemoveChunk(HCkBinData cHandle, int offset, int numBytes);
CK_C_VISIBLE_PUBLIC BOOL CkBinData_SecureClear(HCkBinData cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkBinData_WriteFile(HCkBinData cHandle, const char *path);
#endif
