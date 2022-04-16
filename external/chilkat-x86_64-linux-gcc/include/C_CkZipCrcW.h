// This is a generated source file for Chilkat version 9.5.0.89
#ifndef _C_CkZipCrcWH
#define _C_CkZipCrcWH
#include "chilkatDefs.h"

#include "Chilkat_C.h"


CK_C_VISIBLE_PUBLIC void CkZipCrcW_setAbortCheck(HCkZipCrcW cHandle, BOOL (*fnAbortCheck)(void));
CK_C_VISIBLE_PUBLIC void CkZipCrcW_setPercentDone(HCkZipCrcW cHandle, BOOL (*fnPercentDone)(int pctDone));
CK_C_VISIBLE_PUBLIC void CkZipCrcW_setProgressInfo(HCkZipCrcW cHandle, void (*fnProgressInfo)(const wchar_t *name, const wchar_t *value));
CK_C_VISIBLE_PUBLIC void CkZipCrcW_setTaskCompleted(HCkZipCrcW cHandle, void (*fnTaskCompleted)(HCkTaskW hTask));

CK_C_VISIBLE_PUBLIC HCkZipCrcW CkZipCrcW_Create(void);
CK_C_VISIBLE_PUBLIC void CkZipCrcW_Dispose(HCkZipCrcW handle);
CK_C_VISIBLE_PUBLIC void CkZipCrcW_getDebugLogFilePath(HCkZipCrcW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void  CkZipCrcW_putDebugLogFilePath(HCkZipCrcW cHandle, const wchar_t *newVal);
CK_C_VISIBLE_PUBLIC const wchar_t *CkZipCrcW_debugLogFilePath(HCkZipCrcW cHandle);
CK_C_VISIBLE_PUBLIC void CkZipCrcW_getLastErrorHtml(HCkZipCrcW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkZipCrcW_lastErrorHtml(HCkZipCrcW cHandle);
CK_C_VISIBLE_PUBLIC void CkZipCrcW_getLastErrorText(HCkZipCrcW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkZipCrcW_lastErrorText(HCkZipCrcW cHandle);
CK_C_VISIBLE_PUBLIC void CkZipCrcW_getLastErrorXml(HCkZipCrcW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkZipCrcW_lastErrorXml(HCkZipCrcW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkZipCrcW_getLastMethodSuccess(HCkZipCrcW cHandle);
CK_C_VISIBLE_PUBLIC void  CkZipCrcW_putLastMethodSuccess(HCkZipCrcW cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC BOOL CkZipCrcW_getVerboseLogging(HCkZipCrcW cHandle);
CK_C_VISIBLE_PUBLIC void  CkZipCrcW_putVerboseLogging(HCkZipCrcW cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC void CkZipCrcW_getVersion(HCkZipCrcW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkZipCrcW_version(HCkZipCrcW cHandle);
CK_C_VISIBLE_PUBLIC void CkZipCrcW_BeginStream(HCkZipCrcW cHandle);
CK_C_VISIBLE_PUBLIC unsigned long CkZipCrcW_CalculateCrc(HCkZipCrcW cHandle, HCkByteData data);
CK_C_VISIBLE_PUBLIC unsigned long CkZipCrcW_CrcBd(HCkZipCrcW cHandle, HCkBinDataW bd);
CK_C_VISIBLE_PUBLIC unsigned long CkZipCrcW_CrcSb(HCkZipCrcW cHandle, HCkStringBuilderW sb, const wchar_t *charset);
CK_C_VISIBLE_PUBLIC unsigned long CkZipCrcW_CrcString(HCkZipCrcW cHandle, const wchar_t *str, const wchar_t *charset);
CK_C_VISIBLE_PUBLIC unsigned long CkZipCrcW_EndStream(HCkZipCrcW cHandle);
CK_C_VISIBLE_PUBLIC unsigned long CkZipCrcW_FileCrc(HCkZipCrcW cHandle, const wchar_t *path);
CK_C_VISIBLE_PUBLIC HCkTaskW CkZipCrcW_FileCrcAsync(HCkZipCrcW cHandle, const wchar_t *path);
CK_C_VISIBLE_PUBLIC BOOL CkZipCrcW_LoadTaskCaller(HCkZipCrcW cHandle, HCkTaskW task);
CK_C_VISIBLE_PUBLIC void CkZipCrcW_MoreData(HCkZipCrcW cHandle, HCkByteData data);
CK_C_VISIBLE_PUBLIC BOOL CkZipCrcW_SaveLastError(HCkZipCrcW cHandle, const wchar_t *path);
CK_C_VISIBLE_PUBLIC BOOL CkZipCrcW_ToHex(HCkZipCrcW cHandle, unsigned long crc, HCkString outStr);
CK_C_VISIBLE_PUBLIC const wchar_t *CkZipCrcW_toHex(HCkZipCrcW cHandle, unsigned long crc);
#endif
