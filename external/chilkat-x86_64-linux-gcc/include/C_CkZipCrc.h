// This is a generated source file for Chilkat version 9.5.0.89
#ifndef _C_CkZipCrc_H
#define _C_CkZipCrc_H
#include "chilkatDefs.h"

#include "Chilkat_C.h"


CK_C_VISIBLE_PUBLIC void CkZipCrc_setAbortCheck(HCkZipCrc cHandle, BOOL (*fnAbortCheck)(void));
CK_C_VISIBLE_PUBLIC void CkZipCrc_setPercentDone(HCkZipCrc cHandle, BOOL (*fnPercentDone)(int pctDone));
CK_C_VISIBLE_PUBLIC void CkZipCrc_setProgressInfo(HCkZipCrc cHandle, void (*fnProgressInfo)(const char *name, const char *value));
CK_C_VISIBLE_PUBLIC void CkZipCrc_setTaskCompleted(HCkZipCrc cHandle, void (*fnTaskCompleted)(HCkTask hTask));

CK_C_VISIBLE_PUBLIC void CkZipCrc_setAbortCheck2(HCkZipCrc cHandle, BOOL (*fnAbortCheck2)(void *pContext));
CK_C_VISIBLE_PUBLIC void CkZipCrc_setPercentDone2(HCkZipCrc cHandle, BOOL (*fnPercentDone2)(int pctDone, void *pContext));
CK_C_VISIBLE_PUBLIC void CkZipCrc_setProgressInfo2(HCkZipCrc cHandle, void (*fnProgressInfo2)(const char *name, const char *value, void *pContext));
CK_C_VISIBLE_PUBLIC void CkZipCrc_setTaskCompleted2(HCkZipCrc cHandle, void (*fnTaskCompleted2)(HCkTask hTask, void *pContext));

// setExternalProgress is for C callback functions defined in the external programming language (such as Go)
CK_C_VISIBLE_PUBLIC void CkZipCrc_setExternalProgress(HCkZipCrc cHandle, BOOL on);
CK_C_VISIBLE_PUBLIC void CkZipCrc_setCallbackContext(HCkZipCrc cHandle, void *pContext);

CK_C_VISIBLE_PUBLIC HCkZipCrc CkZipCrc_Create(void);
CK_C_VISIBLE_PUBLIC void CkZipCrc_Dispose(HCkZipCrc handle);
CK_C_VISIBLE_PUBLIC void CkZipCrc_getDebugLogFilePath(HCkZipCrc cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void CkZipCrc_putDebugLogFilePath(HCkZipCrc cHandle, const char *newVal);
CK_C_VISIBLE_PUBLIC const char *CkZipCrc_debugLogFilePath(HCkZipCrc cHandle);
CK_C_VISIBLE_PUBLIC void CkZipCrc_getLastErrorHtml(HCkZipCrc cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkZipCrc_lastErrorHtml(HCkZipCrc cHandle);
CK_C_VISIBLE_PUBLIC void CkZipCrc_getLastErrorText(HCkZipCrc cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkZipCrc_lastErrorText(HCkZipCrc cHandle);
CK_C_VISIBLE_PUBLIC void CkZipCrc_getLastErrorXml(HCkZipCrc cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkZipCrc_lastErrorXml(HCkZipCrc cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkZipCrc_getLastMethodSuccess(HCkZipCrc cHandle);
CK_C_VISIBLE_PUBLIC void CkZipCrc_putLastMethodSuccess(HCkZipCrc cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC BOOL CkZipCrc_getUtf8(HCkZipCrc cHandle);
CK_C_VISIBLE_PUBLIC void CkZipCrc_putUtf8(HCkZipCrc cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC BOOL CkZipCrc_getVerboseLogging(HCkZipCrc cHandle);
CK_C_VISIBLE_PUBLIC void CkZipCrc_putVerboseLogging(HCkZipCrc cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC void CkZipCrc_getVersion(HCkZipCrc cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkZipCrc_version(HCkZipCrc cHandle);
CK_C_VISIBLE_PUBLIC void CkZipCrc_BeginStream(HCkZipCrc cHandle);
CK_C_VISIBLE_PUBLIC unsigned long CkZipCrc_CalculateCrc(HCkZipCrc cHandle, HCkByteData data);
CK_C_VISIBLE_PUBLIC unsigned long CkZipCrc_CrcBd(HCkZipCrc cHandle, HCkBinData bd);
CK_C_VISIBLE_PUBLIC unsigned long CkZipCrc_CrcSb(HCkZipCrc cHandle, HCkStringBuilder sb, const char *charset);
CK_C_VISIBLE_PUBLIC unsigned long CkZipCrc_CrcString(HCkZipCrc cHandle, const char *str, const char *charset);
CK_C_VISIBLE_PUBLIC unsigned long CkZipCrc_EndStream(HCkZipCrc cHandle);
CK_C_VISIBLE_PUBLIC unsigned long CkZipCrc_FileCrc(HCkZipCrc cHandle, const char *path);
CK_C_VISIBLE_PUBLIC HCkTask CkZipCrc_FileCrcAsync(HCkZipCrc cHandle, const char *path);
CK_C_VISIBLE_PUBLIC BOOL CkZipCrc_LoadTaskCaller(HCkZipCrc cHandle, HCkTask task);
CK_C_VISIBLE_PUBLIC void CkZipCrc_MoreData(HCkZipCrc cHandle, HCkByteData data);
CK_C_VISIBLE_PUBLIC BOOL CkZipCrc_SaveLastError(HCkZipCrc cHandle, const char *path);
CK_C_VISIBLE_PUBLIC BOOL CkZipCrc_ToHex(HCkZipCrc cHandle, unsigned long crc, HCkString outStr);
CK_C_VISIBLE_PUBLIC const char *CkZipCrc_toHex(HCkZipCrc cHandle, unsigned long crc);
#endif
