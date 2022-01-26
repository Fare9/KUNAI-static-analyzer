// This is a generated source file for Chilkat version 9.5.0.89
#ifndef _C_CkBz2_H
#define _C_CkBz2_H
#include "chilkatDefs.h"

#include "Chilkat_C.h"


CK_C_VISIBLE_PUBLIC void CkBz2_setAbortCheck(HCkBz2 cHandle, BOOL (*fnAbortCheck)(void));
CK_C_VISIBLE_PUBLIC void CkBz2_setPercentDone(HCkBz2 cHandle, BOOL (*fnPercentDone)(int pctDone));
CK_C_VISIBLE_PUBLIC void CkBz2_setProgressInfo(HCkBz2 cHandle, void (*fnProgressInfo)(const char *name, const char *value));
CK_C_VISIBLE_PUBLIC void CkBz2_setTaskCompleted(HCkBz2 cHandle, void (*fnTaskCompleted)(HCkTask hTask));

CK_C_VISIBLE_PUBLIC void CkBz2_setAbortCheck2(HCkBz2 cHandle, BOOL (*fnAbortCheck2)(void *pContext));
CK_C_VISIBLE_PUBLIC void CkBz2_setPercentDone2(HCkBz2 cHandle, BOOL (*fnPercentDone2)(int pctDone, void *pContext));
CK_C_VISIBLE_PUBLIC void CkBz2_setProgressInfo2(HCkBz2 cHandle, void (*fnProgressInfo2)(const char *name, const char *value, void *pContext));
CK_C_VISIBLE_PUBLIC void CkBz2_setTaskCompleted2(HCkBz2 cHandle, void (*fnTaskCompleted2)(HCkTask hTask, void *pContext));

// setExternalProgress is for C callback functions defined in the external programming language (such as Go)
CK_C_VISIBLE_PUBLIC void CkBz2_setExternalProgress(HCkBz2 cHandle, BOOL on);
CK_C_VISIBLE_PUBLIC void CkBz2_setCallbackContext(HCkBz2 cHandle, void *pContext);

CK_C_VISIBLE_PUBLIC HCkBz2 CkBz2_Create(void);
CK_C_VISIBLE_PUBLIC void CkBz2_Dispose(HCkBz2 handle);
CK_C_VISIBLE_PUBLIC BOOL CkBz2_getAbortCurrent(HCkBz2 cHandle);
CK_C_VISIBLE_PUBLIC void CkBz2_putAbortCurrent(HCkBz2 cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC void CkBz2_getDebugLogFilePath(HCkBz2 cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void CkBz2_putDebugLogFilePath(HCkBz2 cHandle, const char *newVal);
CK_C_VISIBLE_PUBLIC const char *CkBz2_debugLogFilePath(HCkBz2 cHandle);
CK_C_VISIBLE_PUBLIC int CkBz2_getHeartbeatMs(HCkBz2 cHandle);
CK_C_VISIBLE_PUBLIC void CkBz2_putHeartbeatMs(HCkBz2 cHandle, int newVal);
CK_C_VISIBLE_PUBLIC void CkBz2_getLastErrorHtml(HCkBz2 cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkBz2_lastErrorHtml(HCkBz2 cHandle);
CK_C_VISIBLE_PUBLIC void CkBz2_getLastErrorText(HCkBz2 cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkBz2_lastErrorText(HCkBz2 cHandle);
CK_C_VISIBLE_PUBLIC void CkBz2_getLastErrorXml(HCkBz2 cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkBz2_lastErrorXml(HCkBz2 cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkBz2_getLastMethodSuccess(HCkBz2 cHandle);
CK_C_VISIBLE_PUBLIC void CkBz2_putLastMethodSuccess(HCkBz2 cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC BOOL CkBz2_getUtf8(HCkBz2 cHandle);
CK_C_VISIBLE_PUBLIC void CkBz2_putUtf8(HCkBz2 cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC BOOL CkBz2_getVerboseLogging(HCkBz2 cHandle);
CK_C_VISIBLE_PUBLIC void CkBz2_putVerboseLogging(HCkBz2 cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC void CkBz2_getVersion(HCkBz2 cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkBz2_version(HCkBz2 cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkBz2_CompressFile(HCkBz2 cHandle, const char *inFilename, const char *toPath);
CK_C_VISIBLE_PUBLIC HCkTask CkBz2_CompressFileAsync(HCkBz2 cHandle, const char *inFilename, const char *toPath);
CK_C_VISIBLE_PUBLIC BOOL CkBz2_CompressFileToMem(HCkBz2 cHandle, const char *inFilename, HCkByteData outBytes);
CK_C_VISIBLE_PUBLIC HCkTask CkBz2_CompressFileToMemAsync(HCkBz2 cHandle, const char *inFilename);
CK_C_VISIBLE_PUBLIC BOOL CkBz2_CompressMemory(HCkBz2 cHandle, HCkByteData inData, HCkByteData outBytes);
CK_C_VISIBLE_PUBLIC HCkTask CkBz2_CompressMemoryAsync(HCkBz2 cHandle, HCkByteData inData);
CK_C_VISIBLE_PUBLIC BOOL CkBz2_CompressMemToFile(HCkBz2 cHandle, HCkByteData inData, const char *toPath);
CK_C_VISIBLE_PUBLIC HCkTask CkBz2_CompressMemToFileAsync(HCkBz2 cHandle, HCkByteData inData, const char *toPath);
CK_C_VISIBLE_PUBLIC BOOL CkBz2_LoadTaskCaller(HCkBz2 cHandle, HCkTask task);
CK_C_VISIBLE_PUBLIC BOOL CkBz2_SaveLastError(HCkBz2 cHandle, const char *path);
CK_C_VISIBLE_PUBLIC BOOL CkBz2_UncompressFile(HCkBz2 cHandle, const char *inFilename, const char *toPath);
CK_C_VISIBLE_PUBLIC HCkTask CkBz2_UncompressFileAsync(HCkBz2 cHandle, const char *inFilename, const char *toPath);
CK_C_VISIBLE_PUBLIC BOOL CkBz2_UncompressFileToMem(HCkBz2 cHandle, const char *inFilename, HCkByteData outBytes);
CK_C_VISIBLE_PUBLIC HCkTask CkBz2_UncompressFileToMemAsync(HCkBz2 cHandle, const char *inFilename);
CK_C_VISIBLE_PUBLIC BOOL CkBz2_UncompressMemory(HCkBz2 cHandle, HCkByteData inData, HCkByteData outBytes);
CK_C_VISIBLE_PUBLIC HCkTask CkBz2_UncompressMemoryAsync(HCkBz2 cHandle, HCkByteData inData);
CK_C_VISIBLE_PUBLIC BOOL CkBz2_UncompressMemToFile(HCkBz2 cHandle, HCkByteData inData, const char *toPath);
CK_C_VISIBLE_PUBLIC HCkTask CkBz2_UncompressMemToFileAsync(HCkBz2 cHandle, HCkByteData inData, const char *toPath);
CK_C_VISIBLE_PUBLIC BOOL CkBz2_UnlockComponent(HCkBz2 cHandle, const char *regCode);
#endif
