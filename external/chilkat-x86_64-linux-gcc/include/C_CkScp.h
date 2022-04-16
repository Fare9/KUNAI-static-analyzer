// This is a generated source file for Chilkat version 9.5.0.89
#ifndef _C_CkScp_H
#define _C_CkScp_H
#include "chilkatDefs.h"

#include "Chilkat_C.h"


CK_C_VISIBLE_PUBLIC void CkScp_setAbortCheck(HCkScp cHandle, BOOL (*fnAbortCheck)(void));
CK_C_VISIBLE_PUBLIC void CkScp_setPercentDone(HCkScp cHandle, BOOL (*fnPercentDone)(int pctDone));
CK_C_VISIBLE_PUBLIC void CkScp_setProgressInfo(HCkScp cHandle, void (*fnProgressInfo)(const char *name, const char *value));
CK_C_VISIBLE_PUBLIC void CkScp_setTaskCompleted(HCkScp cHandle, void (*fnTaskCompleted)(HCkTask hTask));

CK_C_VISIBLE_PUBLIC void CkScp_setAbortCheck2(HCkScp cHandle, BOOL (*fnAbortCheck2)(void *pContext));
CK_C_VISIBLE_PUBLIC void CkScp_setPercentDone2(HCkScp cHandle, BOOL (*fnPercentDone2)(int pctDone, void *pContext));
CK_C_VISIBLE_PUBLIC void CkScp_setProgressInfo2(HCkScp cHandle, void (*fnProgressInfo2)(const char *name, const char *value, void *pContext));
CK_C_VISIBLE_PUBLIC void CkScp_setTaskCompleted2(HCkScp cHandle, void (*fnTaskCompleted2)(HCkTask hTask, void *pContext));

// setExternalProgress is for C callback functions defined in the external programming language (such as Go)
CK_C_VISIBLE_PUBLIC void CkScp_setExternalProgress(HCkScp cHandle, BOOL on);
CK_C_VISIBLE_PUBLIC void CkScp_setCallbackContext(HCkScp cHandle, void *pContext);

CK_C_VISIBLE_PUBLIC HCkScp CkScp_Create(void);
CK_C_VISIBLE_PUBLIC void CkScp_Dispose(HCkScp handle);
CK_C_VISIBLE_PUBLIC BOOL CkScp_getAbortCurrent(HCkScp cHandle);
CK_C_VISIBLE_PUBLIC void CkScp_putAbortCurrent(HCkScp cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC void CkScp_getDebugLogFilePath(HCkScp cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void CkScp_putDebugLogFilePath(HCkScp cHandle, const char *newVal);
CK_C_VISIBLE_PUBLIC const char *CkScp_debugLogFilePath(HCkScp cHandle);
CK_C_VISIBLE_PUBLIC int CkScp_getHeartbeatMs(HCkScp cHandle);
CK_C_VISIBLE_PUBLIC void CkScp_putHeartbeatMs(HCkScp cHandle, int newVal);
CK_C_VISIBLE_PUBLIC void CkScp_getLastErrorHtml(HCkScp cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkScp_lastErrorHtml(HCkScp cHandle);
CK_C_VISIBLE_PUBLIC void CkScp_getLastErrorText(HCkScp cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkScp_lastErrorText(HCkScp cHandle);
CK_C_VISIBLE_PUBLIC void CkScp_getLastErrorXml(HCkScp cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkScp_lastErrorXml(HCkScp cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkScp_getLastMethodSuccess(HCkScp cHandle);
CK_C_VISIBLE_PUBLIC void CkScp_putLastMethodSuccess(HCkScp cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC int CkScp_getPercentDoneScale(HCkScp cHandle);
CK_C_VISIBLE_PUBLIC void CkScp_putPercentDoneScale(HCkScp cHandle, int newVal);
CK_C_VISIBLE_PUBLIC void CkScp_getSendEnv(HCkScp cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void CkScp_putSendEnv(HCkScp cHandle, const char *newVal);
CK_C_VISIBLE_PUBLIC const char *CkScp_sendEnv(HCkScp cHandle);
CK_C_VISIBLE_PUBLIC void CkScp_getSyncedFiles(HCkScp cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void CkScp_putSyncedFiles(HCkScp cHandle, const char *newVal);
CK_C_VISIBLE_PUBLIC const char *CkScp_syncedFiles(HCkScp cHandle);
CK_C_VISIBLE_PUBLIC void CkScp_getSyncMustMatch(HCkScp cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void CkScp_putSyncMustMatch(HCkScp cHandle, const char *newVal);
CK_C_VISIBLE_PUBLIC const char *CkScp_syncMustMatch(HCkScp cHandle);
CK_C_VISIBLE_PUBLIC void CkScp_getSyncMustMatchDir(HCkScp cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void CkScp_putSyncMustMatchDir(HCkScp cHandle, const char *newVal);
CK_C_VISIBLE_PUBLIC const char *CkScp_syncMustMatchDir(HCkScp cHandle);
CK_C_VISIBLE_PUBLIC void CkScp_getSyncMustNotMatch(HCkScp cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void CkScp_putSyncMustNotMatch(HCkScp cHandle, const char *newVal);
CK_C_VISIBLE_PUBLIC const char *CkScp_syncMustNotMatch(HCkScp cHandle);
CK_C_VISIBLE_PUBLIC void CkScp_getSyncMustNotMatchDir(HCkScp cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void CkScp_putSyncMustNotMatchDir(HCkScp cHandle, const char *newVal);
CK_C_VISIBLE_PUBLIC const char *CkScp_syncMustNotMatchDir(HCkScp cHandle);
CK_C_VISIBLE_PUBLIC void CkScp_getUncommonOptions(HCkScp cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void CkScp_putUncommonOptions(HCkScp cHandle, const char *newVal);
CK_C_VISIBLE_PUBLIC const char *CkScp_uncommonOptions(HCkScp cHandle);
CK_C_VISIBLE_PUBLIC void CkScp_getUnixPermOverride(HCkScp cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void CkScp_putUnixPermOverride(HCkScp cHandle, const char *newVal);
CK_C_VISIBLE_PUBLIC const char *CkScp_unixPermOverride(HCkScp cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkScp_getUtf8(HCkScp cHandle);
CK_C_VISIBLE_PUBLIC void CkScp_putUtf8(HCkScp cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC BOOL CkScp_getVerboseLogging(HCkScp cHandle);
CK_C_VISIBLE_PUBLIC void CkScp_putVerboseLogging(HCkScp cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC void CkScp_getVersion(HCkScp cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkScp_version(HCkScp cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkScp_DownloadBd(HCkScp cHandle, const char *remotePath, HCkBinData bd);
CK_C_VISIBLE_PUBLIC HCkTask CkScp_DownloadBdAsync(HCkScp cHandle, const char *remotePath, HCkBinData bd);
CK_C_VISIBLE_PUBLIC BOOL CkScp_DownloadBinary(HCkScp cHandle, const char *remotePath, HCkByteData outBytes);
CK_C_VISIBLE_PUBLIC HCkTask CkScp_DownloadBinaryAsync(HCkScp cHandle, const char *remotePath);
CK_C_VISIBLE_PUBLIC BOOL CkScp_DownloadBinaryEncoded(HCkScp cHandle, const char *remotePath, const char *encoding, HCkString outStr);
CK_C_VISIBLE_PUBLIC const char *CkScp_downloadBinaryEncoded(HCkScp cHandle, const char *remotePath, const char *encoding);
CK_C_VISIBLE_PUBLIC HCkTask CkScp_DownloadBinaryEncodedAsync(HCkScp cHandle, const char *remotePath, const char *encoding);
CK_C_VISIBLE_PUBLIC BOOL CkScp_DownloadFile(HCkScp cHandle, const char *remotePath, const char *localPath);
CK_C_VISIBLE_PUBLIC HCkTask CkScp_DownloadFileAsync(HCkScp cHandle, const char *remotePath, const char *localPath);
CK_C_VISIBLE_PUBLIC BOOL CkScp_DownloadString(HCkScp cHandle, const char *remotePath, const char *charset, HCkString outStr);
CK_C_VISIBLE_PUBLIC const char *CkScp_downloadString(HCkScp cHandle, const char *remotePath, const char *charset);
CK_C_VISIBLE_PUBLIC HCkTask CkScp_DownloadStringAsync(HCkScp cHandle, const char *remotePath, const char *charset);
CK_C_VISIBLE_PUBLIC BOOL CkScp_LoadTaskCaller(HCkScp cHandle, HCkTask task);
CK_C_VISIBLE_PUBLIC BOOL CkScp_SaveLastError(HCkScp cHandle, const char *path);
CK_C_VISIBLE_PUBLIC BOOL CkScp_SyncTreeDownload(HCkScp cHandle, const char *remoteRoot, const char *localRoot, int mode, BOOL bRecurse);
CK_C_VISIBLE_PUBLIC HCkTask CkScp_SyncTreeDownloadAsync(HCkScp cHandle, const char *remoteRoot, const char *localRoot, int mode, BOOL bRecurse);
CK_C_VISIBLE_PUBLIC BOOL CkScp_SyncTreeUpload(HCkScp cHandle, const char *localBaseDir, const char *remoteBaseDir, int mode, BOOL bRecurse);
CK_C_VISIBLE_PUBLIC HCkTask CkScp_SyncTreeUploadAsync(HCkScp cHandle, const char *localBaseDir, const char *remoteBaseDir, int mode, BOOL bRecurse);
CK_C_VISIBLE_PUBLIC BOOL CkScp_UploadBd(HCkScp cHandle, const char *remotePath, HCkBinData bd);
CK_C_VISIBLE_PUBLIC HCkTask CkScp_UploadBdAsync(HCkScp cHandle, const char *remotePath, HCkBinData bd);
CK_C_VISIBLE_PUBLIC BOOL CkScp_UploadBinary(HCkScp cHandle, const char *remotePath, HCkByteData binData);
CK_C_VISIBLE_PUBLIC HCkTask CkScp_UploadBinaryAsync(HCkScp cHandle, const char *remotePath, HCkByteData binData);
CK_C_VISIBLE_PUBLIC BOOL CkScp_UploadBinaryEncoded(HCkScp cHandle, const char *remotePath, const char *encodedData, const char *encoding);
CK_C_VISIBLE_PUBLIC HCkTask CkScp_UploadBinaryEncodedAsync(HCkScp cHandle, const char *remotePath, const char *encodedData, const char *encoding);
CK_C_VISIBLE_PUBLIC BOOL CkScp_UploadFile(HCkScp cHandle, const char *localPath, const char *remotePath);
CK_C_VISIBLE_PUBLIC HCkTask CkScp_UploadFileAsync(HCkScp cHandle, const char *localPath, const char *remotePath);
CK_C_VISIBLE_PUBLIC BOOL CkScp_UploadString(HCkScp cHandle, const char *remotePath, const char *textData, const char *charset);
CK_C_VISIBLE_PUBLIC HCkTask CkScp_UploadStringAsync(HCkScp cHandle, const char *remotePath, const char *textData, const char *charset);
CK_C_VISIBLE_PUBLIC BOOL CkScp_UseSsh(HCkScp cHandle, HCkSsh sshConnection);
#endif
