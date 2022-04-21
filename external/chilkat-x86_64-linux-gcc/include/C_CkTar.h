// This is a generated source file for Chilkat version 9.5.0.89
#ifndef _C_CkTar_H
#define _C_CkTar_H
#include "chilkatDefs.h"

#include "Chilkat_C.h"


CK_C_VISIBLE_PUBLIC void CkTar_setAbortCheck(HCkTar cHandle, BOOL (*fnAbortCheck)(void));
CK_C_VISIBLE_PUBLIC void CkTar_setPercentDone(HCkTar cHandle, BOOL (*fnPercentDone)(int pctDone));
CK_C_VISIBLE_PUBLIC void CkTar_setProgressInfo(HCkTar cHandle, void (*fnProgressInfo)(const char *name, const char *value));
CK_C_VISIBLE_PUBLIC void CkTar_setTaskCompleted(HCkTar cHandle, void (*fnTaskCompleted)(HCkTask hTask));

CK_C_VISIBLE_PUBLIC void CkTar_setAbortCheck2(HCkTar cHandle, BOOL (*fnAbortCheck2)(void *pContext));
CK_C_VISIBLE_PUBLIC void CkTar_setPercentDone2(HCkTar cHandle, BOOL (*fnPercentDone2)(int pctDone, void *pContext));
CK_C_VISIBLE_PUBLIC void CkTar_setProgressInfo2(HCkTar cHandle, void (*fnProgressInfo2)(const char *name, const char *value, void *pContext));
CK_C_VISIBLE_PUBLIC void CkTar_setTaskCompleted2(HCkTar cHandle, void (*fnTaskCompleted2)(HCkTask hTask, void *pContext));

// setExternalProgress is for C callback functions defined in the external programming language (such as Go)
CK_C_VISIBLE_PUBLIC void CkTar_setExternalProgress(HCkTar cHandle, BOOL on);
CK_C_VISIBLE_PUBLIC void CkTar_setCallbackContext(HCkTar cHandle, void *pContext);

CK_C_VISIBLE_PUBLIC HCkTar CkTar_Create(void);
CK_C_VISIBLE_PUBLIC void CkTar_Dispose(HCkTar handle);
CK_C_VISIBLE_PUBLIC BOOL CkTar_getCaptureXmlListing(HCkTar cHandle);
CK_C_VISIBLE_PUBLIC void CkTar_putCaptureXmlListing(HCkTar cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC void CkTar_getCharset(HCkTar cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void CkTar_putCharset(HCkTar cHandle, const char *newVal);
CK_C_VISIBLE_PUBLIC const char *CkTar_charset(HCkTar cHandle);
CK_C_VISIBLE_PUBLIC void CkTar_getDebugLogFilePath(HCkTar cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void CkTar_putDebugLogFilePath(HCkTar cHandle, const char *newVal);
CK_C_VISIBLE_PUBLIC const char *CkTar_debugLogFilePath(HCkTar cHandle);
CK_C_VISIBLE_PUBLIC int CkTar_getDirMode(HCkTar cHandle);
CK_C_VISIBLE_PUBLIC void CkTar_putDirMode(HCkTar cHandle, int newVal);
CK_C_VISIBLE_PUBLIC void CkTar_getDirPrefix(HCkTar cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void CkTar_putDirPrefix(HCkTar cHandle, const char *newVal);
CK_C_VISIBLE_PUBLIC const char *CkTar_dirPrefix(HCkTar cHandle);
CK_C_VISIBLE_PUBLIC int CkTar_getFileMode(HCkTar cHandle);
CK_C_VISIBLE_PUBLIC void CkTar_putFileMode(HCkTar cHandle, int newVal);
CK_C_VISIBLE_PUBLIC int CkTar_getGroupId(HCkTar cHandle);
CK_C_VISIBLE_PUBLIC void CkTar_putGroupId(HCkTar cHandle, int newVal);
CK_C_VISIBLE_PUBLIC void CkTar_getGroupName(HCkTar cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void CkTar_putGroupName(HCkTar cHandle, const char *newVal);
CK_C_VISIBLE_PUBLIC const char *CkTar_groupName(HCkTar cHandle);
CK_C_VISIBLE_PUBLIC int CkTar_getHeartbeatMs(HCkTar cHandle);
CK_C_VISIBLE_PUBLIC void CkTar_putHeartbeatMs(HCkTar cHandle, int newVal);
CK_C_VISIBLE_PUBLIC void CkTar_getLastErrorHtml(HCkTar cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkTar_lastErrorHtml(HCkTar cHandle);
CK_C_VISIBLE_PUBLIC void CkTar_getLastErrorText(HCkTar cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkTar_lastErrorText(HCkTar cHandle);
CK_C_VISIBLE_PUBLIC void CkTar_getLastErrorXml(HCkTar cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkTar_lastErrorXml(HCkTar cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkTar_getLastMethodSuccess(HCkTar cHandle);
CK_C_VISIBLE_PUBLIC void CkTar_putLastMethodSuccess(HCkTar cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC BOOL CkTar_getMatchCaseSensitive(HCkTar cHandle);
CK_C_VISIBLE_PUBLIC void CkTar_putMatchCaseSensitive(HCkTar cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC void CkTar_getMustMatch(HCkTar cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void CkTar_putMustMatch(HCkTar cHandle, const char *newVal);
CK_C_VISIBLE_PUBLIC const char *CkTar_mustMatch(HCkTar cHandle);
CK_C_VISIBLE_PUBLIC void CkTar_getMustNotMatch(HCkTar cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void CkTar_putMustNotMatch(HCkTar cHandle, const char *newVal);
CK_C_VISIBLE_PUBLIC const char *CkTar_mustNotMatch(HCkTar cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkTar_getNoAbsolutePaths(HCkTar cHandle);
CK_C_VISIBLE_PUBLIC void CkTar_putNoAbsolutePaths(HCkTar cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC int CkTar_getNumDirRoots(HCkTar cHandle);
CK_C_VISIBLE_PUBLIC int CkTar_getPercentDoneScale(HCkTar cHandle);
CK_C_VISIBLE_PUBLIC void CkTar_putPercentDoneScale(HCkTar cHandle, int newVal);
CK_C_VISIBLE_PUBLIC int CkTar_getScriptFileMode(HCkTar cHandle);
CK_C_VISIBLE_PUBLIC void CkTar_putScriptFileMode(HCkTar cHandle, int newVal);
CK_C_VISIBLE_PUBLIC BOOL CkTar_getSuppressOutput(HCkTar cHandle);
CK_C_VISIBLE_PUBLIC void CkTar_putSuppressOutput(HCkTar cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC BOOL CkTar_getUntarCaseSensitive(HCkTar cHandle);
CK_C_VISIBLE_PUBLIC void CkTar_putUntarCaseSensitive(HCkTar cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC BOOL CkTar_getUntarDebugLog(HCkTar cHandle);
CK_C_VISIBLE_PUBLIC void CkTar_putUntarDebugLog(HCkTar cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC BOOL CkTar_getUntarDiscardPaths(HCkTar cHandle);
CK_C_VISIBLE_PUBLIC void CkTar_putUntarDiscardPaths(HCkTar cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC void CkTar_getUntarFromDir(HCkTar cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void CkTar_putUntarFromDir(HCkTar cHandle, const char *newVal);
CK_C_VISIBLE_PUBLIC const char *CkTar_untarFromDir(HCkTar cHandle);
CK_C_VISIBLE_PUBLIC void CkTar_getUntarMatchPattern(HCkTar cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void CkTar_putUntarMatchPattern(HCkTar cHandle, const char *newVal);
CK_C_VISIBLE_PUBLIC const char *CkTar_untarMatchPattern(HCkTar cHandle);
CK_C_VISIBLE_PUBLIC int CkTar_getUntarMaxCount(HCkTar cHandle);
CK_C_VISIBLE_PUBLIC void CkTar_putUntarMaxCount(HCkTar cHandle, int newVal);
CK_C_VISIBLE_PUBLIC int CkTar_getUserId(HCkTar cHandle);
CK_C_VISIBLE_PUBLIC void CkTar_putUserId(HCkTar cHandle, int newVal);
CK_C_VISIBLE_PUBLIC void CkTar_getUserName(HCkTar cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void CkTar_putUserName(HCkTar cHandle, const char *newVal);
CK_C_VISIBLE_PUBLIC const char *CkTar_userName(HCkTar cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkTar_getUtf8(HCkTar cHandle);
CK_C_VISIBLE_PUBLIC void CkTar_putUtf8(HCkTar cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC BOOL CkTar_getVerboseLogging(HCkTar cHandle);
CK_C_VISIBLE_PUBLIC void CkTar_putVerboseLogging(HCkTar cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC void CkTar_getVersion(HCkTar cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkTar_version(HCkTar cHandle);
CK_C_VISIBLE_PUBLIC void CkTar_getWriteFormat(HCkTar cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void CkTar_putWriteFormat(HCkTar cHandle, const char *newVal);
CK_C_VISIBLE_PUBLIC const char *CkTar_writeFormat(HCkTar cHandle);
CK_C_VISIBLE_PUBLIC void CkTar_getXmlListing(HCkTar cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void CkTar_putXmlListing(HCkTar cHandle, const char *newVal);
CK_C_VISIBLE_PUBLIC const char *CkTar_xmlListing(HCkTar cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkTar_AddDirRoot(HCkTar cHandle, const char *dirPath);
CK_C_VISIBLE_PUBLIC BOOL CkTar_AddDirRoot2(HCkTar cHandle, const char *rootPrefix, const char *rootPath);
CK_C_VISIBLE_PUBLIC BOOL CkTar_AddFile(HCkTar cHandle, const char *path);
CK_C_VISIBLE_PUBLIC BOOL CkTar_AddFile2(HCkTar cHandle, const char *filePath, const char *pathWithinTar);
CK_C_VISIBLE_PUBLIC BOOL CkTar_ClearDirRootsAndFiles(HCkTar cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkTar_CreateDeb(HCkTar cHandle, const char *controlPath, const char *dataPath, const char *debPath);
CK_C_VISIBLE_PUBLIC BOOL CkTar_GetDirRoot(HCkTar cHandle, int index, HCkString outStr);
CK_C_VISIBLE_PUBLIC const char *CkTar_getDirRoot(HCkTar cHandle, int index);
CK_C_VISIBLE_PUBLIC BOOL CkTar_ListXml(HCkTar cHandle, const char *tarPath, HCkString outStr);
CK_C_VISIBLE_PUBLIC const char *CkTar_listXml(HCkTar cHandle, const char *tarPath);
CK_C_VISIBLE_PUBLIC HCkTask CkTar_ListXmlAsync(HCkTar cHandle, const char *tarPath);
CK_C_VISIBLE_PUBLIC BOOL CkTar_LoadTaskCaller(HCkTar cHandle, HCkTask task);
CK_C_VISIBLE_PUBLIC BOOL CkTar_SaveLastError(HCkTar cHandle, const char *path);
CK_C_VISIBLE_PUBLIC BOOL CkTar_UnlockComponent(HCkTar cHandle, const char *unlockCode);
CK_C_VISIBLE_PUBLIC int CkTar_Untar(HCkTar cHandle, const char *tarPath);
CK_C_VISIBLE_PUBLIC HCkTask CkTar_UntarAsync(HCkTar cHandle, const char *tarPath);
CK_C_VISIBLE_PUBLIC BOOL CkTar_UntarBz2(HCkTar cHandle, const char *tarPath);
CK_C_VISIBLE_PUBLIC HCkTask CkTar_UntarBz2Async(HCkTar cHandle, const char *tarPath);
CK_C_VISIBLE_PUBLIC BOOL CkTar_UntarFirstMatchingToBd(HCkTar cHandle, const char *tarPath, const char *matchPattern, HCkBinData bd);
CK_C_VISIBLE_PUBLIC BOOL CkTar_UntarFirstMatchingToMemory(HCkTar cHandle, HCkByteData tarFileBytes, const char *matchPattern, HCkByteData outBytes);
CK_C_VISIBLE_PUBLIC int CkTar_UntarFromMemory(HCkTar cHandle, HCkByteData tarFileBytes);
CK_C_VISIBLE_PUBLIC HCkTask CkTar_UntarFromMemoryAsync(HCkTar cHandle, HCkByteData tarFileBytes);
CK_C_VISIBLE_PUBLIC BOOL CkTar_UntarGz(HCkTar cHandle, const char *tarPath);
CK_C_VISIBLE_PUBLIC HCkTask CkTar_UntarGzAsync(HCkTar cHandle, const char *tarPath);
CK_C_VISIBLE_PUBLIC BOOL CkTar_UntarZ(HCkTar cHandle, const char *tarPath);
CK_C_VISIBLE_PUBLIC HCkTask CkTar_UntarZAsync(HCkTar cHandle, const char *tarPath);
CK_C_VISIBLE_PUBLIC BOOL CkTar_VerifyTar(HCkTar cHandle, const char *tarPath);
CK_C_VISIBLE_PUBLIC HCkTask CkTar_VerifyTarAsync(HCkTar cHandle, const char *tarPath);
CK_C_VISIBLE_PUBLIC BOOL CkTar_WriteTar(HCkTar cHandle, const char *tarPath);
CK_C_VISIBLE_PUBLIC HCkTask CkTar_WriteTarAsync(HCkTar cHandle, const char *tarPath);
CK_C_VISIBLE_PUBLIC BOOL CkTar_WriteTarBz2(HCkTar cHandle, const char *bz2Path);
CK_C_VISIBLE_PUBLIC HCkTask CkTar_WriteTarBz2Async(HCkTar cHandle, const char *bz2Path);
CK_C_VISIBLE_PUBLIC BOOL CkTar_WriteTarGz(HCkTar cHandle, const char *gzPath);
CK_C_VISIBLE_PUBLIC HCkTask CkTar_WriteTarGzAsync(HCkTar cHandle, const char *gzPath);
#endif
