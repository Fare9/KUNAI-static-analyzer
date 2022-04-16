// This is a generated source file for Chilkat version 9.5.0.89
#ifndef _C_CkUnixCompressWH
#define _C_CkUnixCompressWH
#include "chilkatDefs.h"

#include "Chilkat_C.h"


CK_C_VISIBLE_PUBLIC void CkUnixCompressW_setAbortCheck(HCkUnixCompressW cHandle, BOOL (*fnAbortCheck)(void));
CK_C_VISIBLE_PUBLIC void CkUnixCompressW_setPercentDone(HCkUnixCompressW cHandle, BOOL (*fnPercentDone)(int pctDone));
CK_C_VISIBLE_PUBLIC void CkUnixCompressW_setProgressInfo(HCkUnixCompressW cHandle, void (*fnProgressInfo)(const wchar_t *name, const wchar_t *value));
CK_C_VISIBLE_PUBLIC void CkUnixCompressW_setTaskCompleted(HCkUnixCompressW cHandle, void (*fnTaskCompleted)(HCkTaskW hTask));

CK_C_VISIBLE_PUBLIC HCkUnixCompressW CkUnixCompressW_Create(void);
CK_C_VISIBLE_PUBLIC void CkUnixCompressW_Dispose(HCkUnixCompressW handle);
CK_C_VISIBLE_PUBLIC BOOL CkUnixCompressW_getAbortCurrent(HCkUnixCompressW cHandle);
CK_C_VISIBLE_PUBLIC void  CkUnixCompressW_putAbortCurrent(HCkUnixCompressW cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC void CkUnixCompressW_getDebugLogFilePath(HCkUnixCompressW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void  CkUnixCompressW_putDebugLogFilePath(HCkUnixCompressW cHandle, const wchar_t *newVal);
CK_C_VISIBLE_PUBLIC const wchar_t *CkUnixCompressW_debugLogFilePath(HCkUnixCompressW cHandle);
CK_C_VISIBLE_PUBLIC int CkUnixCompressW_getHeartbeatMs(HCkUnixCompressW cHandle);
CK_C_VISIBLE_PUBLIC void  CkUnixCompressW_putHeartbeatMs(HCkUnixCompressW cHandle, int newVal);
CK_C_VISIBLE_PUBLIC void CkUnixCompressW_getLastErrorHtml(HCkUnixCompressW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkUnixCompressW_lastErrorHtml(HCkUnixCompressW cHandle);
CK_C_VISIBLE_PUBLIC void CkUnixCompressW_getLastErrorText(HCkUnixCompressW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkUnixCompressW_lastErrorText(HCkUnixCompressW cHandle);
CK_C_VISIBLE_PUBLIC void CkUnixCompressW_getLastErrorXml(HCkUnixCompressW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkUnixCompressW_lastErrorXml(HCkUnixCompressW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkUnixCompressW_getLastMethodSuccess(HCkUnixCompressW cHandle);
CK_C_VISIBLE_PUBLIC void  CkUnixCompressW_putLastMethodSuccess(HCkUnixCompressW cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC BOOL CkUnixCompressW_getVerboseLogging(HCkUnixCompressW cHandle);
CK_C_VISIBLE_PUBLIC void  CkUnixCompressW_putVerboseLogging(HCkUnixCompressW cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC void CkUnixCompressW_getVersion(HCkUnixCompressW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkUnixCompressW_version(HCkUnixCompressW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkUnixCompressW_CompressFile(HCkUnixCompressW cHandle, const wchar_t *inFilename, const wchar_t *destPath);
CK_C_VISIBLE_PUBLIC HCkTaskW CkUnixCompressW_CompressFileAsync(HCkUnixCompressW cHandle, const wchar_t *inFilename, const wchar_t *destPath);
CK_C_VISIBLE_PUBLIC BOOL CkUnixCompressW_CompressFileToMem(HCkUnixCompressW cHandle, const wchar_t *inFilename, HCkByteData outData);
CK_C_VISIBLE_PUBLIC HCkTaskW CkUnixCompressW_CompressFileToMemAsync(HCkUnixCompressW cHandle, const wchar_t *inFilename);
CK_C_VISIBLE_PUBLIC BOOL CkUnixCompressW_CompressMemory(HCkUnixCompressW cHandle, HCkByteData inData, HCkByteData outData);
CK_C_VISIBLE_PUBLIC BOOL CkUnixCompressW_CompressMemToFile(HCkUnixCompressW cHandle, HCkByteData inData, const wchar_t *destPath);
CK_C_VISIBLE_PUBLIC BOOL CkUnixCompressW_CompressString(HCkUnixCompressW cHandle, const wchar_t *inStr, const wchar_t *charset, HCkByteData outBytes);
CK_C_VISIBLE_PUBLIC BOOL CkUnixCompressW_CompressStringToFile(HCkUnixCompressW cHandle, const wchar_t *inStr, const wchar_t *charset, const wchar_t *destPath);
CK_C_VISIBLE_PUBLIC BOOL CkUnixCompressW_LoadTaskCaller(HCkUnixCompressW cHandle, HCkTaskW task);
CK_C_VISIBLE_PUBLIC BOOL CkUnixCompressW_SaveLastError(HCkUnixCompressW cHandle, const wchar_t *path);
CK_C_VISIBLE_PUBLIC BOOL CkUnixCompressW_UncompressFile(HCkUnixCompressW cHandle, const wchar_t *inFilename, const wchar_t *destPath);
CK_C_VISIBLE_PUBLIC HCkTaskW CkUnixCompressW_UncompressFileAsync(HCkUnixCompressW cHandle, const wchar_t *inFilename, const wchar_t *destPath);
CK_C_VISIBLE_PUBLIC BOOL CkUnixCompressW_UncompressFileToMem(HCkUnixCompressW cHandle, const wchar_t *inFilename, HCkByteData outData);
CK_C_VISIBLE_PUBLIC HCkTaskW CkUnixCompressW_UncompressFileToMemAsync(HCkUnixCompressW cHandle, const wchar_t *inFilename);
CK_C_VISIBLE_PUBLIC BOOL CkUnixCompressW_UncompressFileToString(HCkUnixCompressW cHandle, const wchar_t *zFilename, const wchar_t *charset, HCkString outStr);
CK_C_VISIBLE_PUBLIC const wchar_t *CkUnixCompressW_uncompressFileToString(HCkUnixCompressW cHandle, const wchar_t *zFilename, const wchar_t *charset);
CK_C_VISIBLE_PUBLIC HCkTaskW CkUnixCompressW_UncompressFileToStringAsync(HCkUnixCompressW cHandle, const wchar_t *zFilename, const wchar_t *charset);
CK_C_VISIBLE_PUBLIC BOOL CkUnixCompressW_UncompressMemory(HCkUnixCompressW cHandle, HCkByteData inData, HCkByteData outData);
CK_C_VISIBLE_PUBLIC BOOL CkUnixCompressW_UncompressMemToFile(HCkUnixCompressW cHandle, HCkByteData inData, const wchar_t *destPath);
CK_C_VISIBLE_PUBLIC BOOL CkUnixCompressW_UncompressString(HCkUnixCompressW cHandle, HCkByteData inCompressedData, const wchar_t *charset, HCkString outStr);
CK_C_VISIBLE_PUBLIC const wchar_t *CkUnixCompressW_uncompressString(HCkUnixCompressW cHandle, HCkByteData inCompressedData, const wchar_t *charset);
CK_C_VISIBLE_PUBLIC BOOL CkUnixCompressW_UnlockComponent(HCkUnixCompressW cHandle, const wchar_t *unlockCode);
CK_C_VISIBLE_PUBLIC BOOL CkUnixCompressW_UnTarZ(HCkUnixCompressW cHandle, const wchar_t *zFilename, const wchar_t *destDir, BOOL bNoAbsolute);
CK_C_VISIBLE_PUBLIC HCkTaskW CkUnixCompressW_UnTarZAsync(HCkUnixCompressW cHandle, const wchar_t *zFilename, const wchar_t *destDir, BOOL bNoAbsolute);
#endif
